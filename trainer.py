# -*- coding: utf-8 -*-

import os
import os.path as osp
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from PIL import Image
from tqdm import tqdm

import torch.nn.functional as F

from utils import calc_wer, diagonal_attention_prior, attention_to_duration_targets

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from utils import *

class Trainer(object):
    def __init__(self,
                 model=None,
                 criterion=None,
                 optimizer=None,
                 scheduler=None,
                 config={},
                 device=torch.device("cpu"),
                 logger=logger,
                 sorted_train_dataloader=None,
                 shuffled_train_dataloader=None,
                 sorted_val_dataloader=None,
                 shuffled_val_dataloader=None,
                 initial_steps=0,
                 initial_epochs=0,
                 switch_sortagrad_dataset_epoch = 10,
                 use_diagonal_attention_prior = True,
                 diagonal_attention_prior_weight = 0.1,
                 ctc_weight=1.0,
                 s2s_weight=1.0,
                 aux_alignment_config=None):

        self.steps = initial_steps
        self.epochs = initial_epochs
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.sorted_train_dataloader = sorted_train_dataloader
        self.shuffled_train_dataloader = shuffled_train_dataloader
        self.train_dataloader = sorted_train_dataloader
        self.sorted_val_dataloader = sorted_val_dataloader
        self.shuffled_val_dataloader = shuffled_val_dataloader
        self.val_dataloader = sorted_val_dataloader
        self.config = config
        self.device = device
        self.finish_train = False
        self.logger = logger
        self.fp16_run = False
        self.switch_sortagrad_dataset_epoch = switch_sortagrad_dataset_epoch
        self.use_diagonal_attention_prior = use_diagonal_attention_prior
        self.diagonal_attention_prior_weight = diagonal_attention_prior_weight
        self.maxm_mem_usage = 0
        self.ctc_weight = ctc_weight
        self.s2s_weight = s2s_weight
        self.aux_alignment_config = aux_alignment_config or {}

    def save_checkpoint(self, checkpoint_path):
        """Save checkpoint.
        Args:
            checkpoint_path (str): Checkpoint path to be saved.
        """
        state_dict = {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "steps": self.steps,
            "epochs": self.epochs,
        }
        state_dict["model"] = self.model.state_dict()

        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path))
        torch.save(state_dict, checkpoint_path)

    def load_checkpoint(self, checkpoint_path, load_only_params=False):
        """Load checkpoint.

        Args:
            checkpoint_path (str): Checkpoint path to be loaded.
            load_only_params (bool): Whether to load only model parameters.

        """
        self.logger.info("Loading checkpoint from: %s" % checkpoint_path)
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        self._load(state_dict["model"], self.model)

        if not load_only_params:
            self.steps = state_dict["steps"]
            self.epochs = state_dict["epochs"]
            self.optimizer.load_state_dict(state_dict["optimizer"])
            self.logger.info("Starting training from epoch %i" % state_dict["epochs"])

            # overwrite schedular argument parameters
            state_dict["scheduler"].update(**self.config.get("scheduler_params", {}))
            self.scheduler.load_state_dict(state_dict["scheduler"])

        return self.epochs

    def _load(self, states, model, force_load=True):
        model_states = model.state_dict()
        for key, val in states.items():
            try:
                if key not in model_states:
                    continue
                if isinstance(val, nn.Parameter):
                    val = val.data

                if val.shape != model_states[key].shape:
                    self.logger.info("%s does not have same shape" % key)
                    print(val.shape, model_states[key].shape)
                    if not force_load:
                        continue

                    min_shape = np.minimum(np.array(val.shape), np.array(model_states[key].shape))
                    slices = [slice(0, min_index) for min_index in min_shape]
                    model_states[key][slices].copy_(val[slices])
                else:
                    model_states[key].copy_(val)
            except:
                self.logger.info("not exist :%s" % key)
                print("not exist ", key)

    @staticmethod
    def get_gradient_norm(model):
        total_norm = 0
        for p in model.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2

        total_norm = np.sqrt(total_norm)
        return total_norm

    @staticmethod
    def length_to_mask(lengths):
        mask = (
            torch.arange(lengths.max(), device=lengths.device)
            .unsqueeze(0)
            .expand(lengths.shape[0], -1)
            .to(lengths.device)
        )
        mask = torch.gt(mask + 1, lengths.unsqueeze(1))
        return mask.to(dtype=torch.bool)

    def _compute_duration_loss(self, duration_pred, alignments, text_lengths, mel_lengths, text_mask):
        if duration_pred is None or alignments is None:
            return None

        if not self.aux_alignment_config.get('enabled', False):
            return None

        with torch.no_grad():
            duration_targets = attention_to_duration_targets(
                alignments,
                text_lengths=text_lengths,
                mel_lengths=mel_lengths,
                detach_attention=self.aux_alignment_config.get('detach_attention', True)
            )

        if duration_targets is None:
            return None

        duration_targets = duration_targets.to(duration_pred.device)
        if text_mask is not None and duration_targets.size(1) > text_mask.size(1):
            text_mask = F.pad(text_mask, (0, duration_targets.size(1) - text_mask.size(1)), value=True)
        if text_mask is not None and duration_targets.size(1) < text_mask.size(1):
            text_mask = text_mask[:, :duration_targets.size(1)]

        valid_mask = None
        if text_mask is not None:
            valid_mask = ~text_mask

        offset = float(self.aux_alignment_config.get('duration_offset', 1.0))
        use_log = self.aux_alignment_config.get('use_log_duration_loss', True)

        target_tensor = duration_targets.float()
        if use_log:
            target_tensor = torch.log(target_tensor + offset)

        pred_tensor = duration_pred
        if pred_tensor.size(1) < target_tensor.size(1):
            target_tensor = target_tensor[:, :pred_tensor.size(1)]
            if valid_mask is not None:
                valid_mask = valid_mask[:, :pred_tensor.size(1)]
        elif pred_tensor.size(1) > target_tensor.size(1):
            pad_size = pred_tensor.size(1) - target_tensor.size(1)
            target_tensor = F.pad(target_tensor, (0, pad_size))
            if valid_mask is not None:
                valid_mask = F.pad(valid_mask, (0, pad_size), value=False)

        if valid_mask is not None:
            pred_flat = pred_tensor.masked_select(valid_mask)
            target_flat = target_tensor.masked_select(valid_mask)
        else:
            pred_flat = pred_tensor.reshape(-1)
            target_flat = target_tensor.reshape(-1)

        if pred_flat.numel() == 0:
            return None

        loss = F.mse_loss(pred_flat, target_flat)
        return loss

    def _get_lr(self):
        for param_group in self.optimizer.param_groups:
            lr = param_group['lr']
            break
        return lr

    @staticmethod
    def get_image(arrs):
        pil_images = []
        height = 0
        width = 0
        for arr in arrs:
            uint_arr = (((arr - arr.min()) / (arr.max() - arr.min())) * 255).astype(np.uint8)
            pil_image = Image.fromarray(uint_arr)
            pil_images.append(pil_image)
            height += uint_arr.shape[0]
            width = max(width, uint_arr.shape[1])

        palette = Image.new('L', (width, height))
        curr_heigth = 0
        for pil_image in pil_images:
            palette.paste(pil_image, (0, curr_heigth))
            curr_heigth += pil_image.size[1]

        return palette

    def run(self, batch):
        # FIX: trying to fix OOM errors
        #torch.cuda.empty_cache()
        self.optimizer.zero_grad()
        batch = [b.to(self.device) for b in batch]
        text_input, text_input_length, mel_input, mel_input_length = batch
        mel_input_length = mel_input_length // (2 ** self.model.n_down)
        mel_mask = self.model.length_to_mask(mel_input_length)
        text_mask = self.model.length_to_mask(text_input_length)
        model_outputs = self.model(
            mel_input,
            src_key_padding_mask=mel_mask,
            text_input=text_input,
            text_mask=text_mask,
        )

        if isinstance(model_outputs, dict):
            ppgs = model_outputs.get('logits_ctc')
            s2s_pred = model_outputs.get('logits_s2s')
            s2s_attn = model_outputs.get('alignments')
            duration_pred = model_outputs.get('duration_pred')
        else:
            ppgs, s2s_pred, s2s_attn = model_outputs
            duration_pred = None

        if ppgs is None or s2s_pred is None:
            raise ValueError("Model must return both CTC and seq2seq logits during training")

        if self.use_diagonal_attention_prior and s2s_attn is not None:
            loss_diagonal = diagonal_attention_prior(s2s_attn, text_input_length, mel_input_length)
        else:
            loss_diagonal = 0

        loss_ctc = self.criterion['ctc'](
            ppgs.log_softmax(dim=2).transpose(0, 1),
            text_input,
            mel_input_length,
            text_input_length,
        )

        loss_s2s = 0
        for _s2s_pred, _text_input, _text_length in zip(s2s_pred, text_input, text_input_length):
            loss_s2s += self.criterion['ce'](_s2s_pred[:_text_length], _text_input[:_text_length])
        loss_s2s /= text_input.size(0)

        duration_loss = self._compute_duration_loss(
            duration_pred,
            s2s_attn,
            text_lengths=text_input_length,
            mel_lengths=mel_input_length,
            text_mask=text_mask,
        )

        total_loss = self.ctc_weight * loss_ctc + self.s2s_weight * loss_s2s
        if self.use_diagonal_attention_prior and s2s_attn is not None:
            total_loss = total_loss + self.diagonal_attention_prior_weight * loss_diagonal

        if duration_loss is not None:
            weight = float(self.aux_alignment_config.get('loss_weight', 1.0))
            total_loss = total_loss + weight * duration_loss

        loss = total_loss
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.model.parameters(), 5)
        self.optimizer.step()
        self.scheduler.step()
        result = {
            'loss': loss.item(),
            'ctc': loss_ctc.item(),
            's2s': loss_s2s.item(),
        }
        if duration_loss is not None:
            result['duration'] = duration_loss.item()
        if self.use_diagonal_attention_prior and s2s_attn is not None:
            result['diagonal'] = loss_diagonal.item() if not isinstance(loss_diagonal, (int, float)) else float(loss_diagonal)
        return result

    def _train_epoch(self):
        # switch dataloader if we're above configured epoch
        if ( self.epochs == self.switch_sortagrad_dataset_epoch) or ( self.epochs == 0 and self.switch_sortagrad_dataset_epoch <= 0 ) :
            self.train_dataloader = self.shuffled_train_dataloader
            self.val_dataloader = self.shuffled_val_dataloader
            self.logger.info("")
            self.logger.info("[SortaGrad]: switching sorted to shuffled dataloader at configured epoch %i" % self.epochs)
            self.logger.info("")

        train_losses = defaultdict(list)
        self.model.train()
        for train_steps_per_epoch, batch in enumerate(tqdm(self.train_dataloader, desc="[train]"), 1):
            losses = self.run(batch)
            for key, value in losses.items():
                train_losses["train/%s" % key].append(value)

        train_losses = {key: np.mean(value) for key, value in train_losses.items()}
        train_losses['train/learning_rate'] = self._get_lr()
        self.epochs += 1

        gpu_id = 0
        allocated = torch.cuda.memory_allocated(gpu_id) / 1024**2
        if allocated > self.maxm_mem_usage:
            self.maxm_mem_usage = allocated

        train_losses['gpu/max_allocation_recorded'] = self.maxm_mem_usage
        train_losses['gpu/current_allocation'] = allocated
        return train_losses

    @torch.no_grad()
    def _eval_epoch(self):
        self.model.eval()
        eval_losses = defaultdict(list)
        eval_images = defaultdict(list)
        for eval_steps_per_epoch, batch in enumerate(tqdm(self.val_dataloader, desc="[eval]"), 1):
            batch = [b.to(self.device) for b in batch]
            text_input, text_input_length, mel_input, mel_input_length = batch
            mel_input_length = mel_input_length // (2 ** self.model.n_down)
            mel_mask = self.model.length_to_mask(mel_input_length)
            text_mask = self.model.length_to_mask(text_input_length)
            model_outputs = self.model(
                mel_input,
                src_key_padding_mask=mel_mask,
                text_input=text_input,
                text_mask=text_mask,
            )

            if isinstance(model_outputs, dict):
                ppgs = model_outputs.get('logits_ctc')
                s2s_pred = model_outputs.get('logits_s2s')
                s2s_attn = model_outputs.get('alignments')
                duration_pred = model_outputs.get('duration_pred')
            else:
                ppgs, s2s_pred, s2s_attn = model_outputs
                duration_pred = None

            loss_ctc = self.criterion['ctc'](ppgs.log_softmax(dim=2).transpose(0, 1),
                                          text_input, mel_input_length, text_input_length)
            loss_s2s = 0
            for _s2s_pred, _text_input, _text_length in zip(s2s_pred, text_input, text_input_length):
                loss_s2s += self.criterion['ce'](_s2s_pred[:_text_length], _text_input[:_text_length])
            loss_s2s /= text_input.size(0)

            duration_loss = self._compute_duration_loss(
                duration_pred,
                s2s_attn,
                text_lengths=text_input_length,
                mel_lengths=mel_input_length,
                text_mask=text_mask,
            )

            loss = self.ctc_weight * loss_ctc + self.s2s_weight * loss_s2s
            if duration_loss is not None:
                loss = loss + float(self.aux_alignment_config.get('loss_weight', 1.0)) * duration_loss

            eval_losses["eval/ctc"].append(loss_ctc.item())
            eval_losses["eval/s2s"].append(loss_s2s.item())
            eval_losses["eval/loss"].append(loss.item())
            if duration_loss is not None:
                eval_losses["eval/duration"].append(duration_loss.item())

            _, amax_ppgs = torch.max(ppgs, dim=2)
            wers = [calc_wer(target[:text_length],
                             pred[:mel_length],
                             ignore_indexes=list(range(5))) \
                    for target, pred, text_length, mel_length in zip(
                            text_input.cpu(), amax_ppgs.cpu(), text_input_length.cpu(), mel_input_length.cpu())]
            eval_losses["eval/wer"].extend(wers)

            _, amax_s2s = torch.max(s2s_pred, dim=2)
            acc = [torch.eq(target[:length], pred[:length]).float().mean().item() \
                   for target, pred, length in zip(text_input.cpu(), amax_s2s.cpu(), text_input_length.cpu())]
            eval_losses["eval/acc"].extend(acc)

            if eval_steps_per_epoch <= 2:
                eval_images["eval/image"].append(
                    self.get_image([s2s_attn[0].cpu().numpy()]))

        eval_losses = {key: np.mean(value) for key, value in eval_losses.items()}
        eval_losses.update(eval_images)
        return eval_losses