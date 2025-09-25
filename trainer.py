# -*- coding: utf-8 -*-

import os
import os.path as osp
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from utils import calc_wer

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
                 use_variance_adaptor=False,
                 variance_loss_weight=1.0,
                 variance_loss_type='mse',
                 detach_duration_targets=True):

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
        self.use_variance_adaptor = use_variance_adaptor
        self.variance_loss_weight = variance_loss_weight
        self.variance_loss_type = variance_loss_type
        self.detach_duration_targets = detach_duration_targets

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
            .type_as(lengths)
        )
        mask = torch.gt(mask + 1, lengths.unsqueeze(1))
        return mask

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
        future_mask = self.model.get_future_mask(
            mel_input.size(2)//(2**self.model.n_down), unmask_future_steps=0).to(self.device)
        mel_mask = self.model.length_to_mask(mel_input_length)
        text_mask = self.model.length_to_mask(text_input_length)
        model_outputs = self.model(
            mel_input, src_key_padding_mask=mel_mask, text_input=text_input)
        if isinstance(model_outputs, (list, tuple)) and len(model_outputs) == 4:
            ppgs, s2s_pred, s2s_attn, auxiliary_outputs = model_outputs
        else:
            ppgs, s2s_pred, s2s_attn = model_outputs
            auxiliary_outputs = {}

        if self.use_diagonal_attention_prior:
            loss_diagonal = diagonal_attention_prior(s2s_attn, text_input_length, mel_input_length)

        loss_ctc = self.criterion['ctc'](ppgs.log_softmax(dim=2).transpose(0, 1),
                                      text_input, mel_input_length, text_input_length)

        loss_s2s = 0
        for _s2s_pred, _text_input, _text_length in zip(s2s_pred, text_input, text_input_length):
            loss_s2s += self.criterion['ce'](_s2s_pred[:_text_length], _text_input[:_text_length])
        loss_s2s /= text_input.size(0)

        duration_loss = torch.tensor(0.0, device=self.device)
        if self.use_variance_adaptor and auxiliary_outputs:
            log_duration_pred = auxiliary_outputs.get('log_duration')
            if log_duration_pred is not None:
                attn_for_duration = s2s_attn.detach() if self.detach_duration_targets else s2s_attn
                duration_targets = self._compute_duration_targets(
                    attn_for_duration,
                    text_input_length,
                    mel_input_length)
                log_duration_target = torch.log1p(duration_targets)
                duration_loss = self._duration_loss(
                    log_duration_pred,
                    log_duration_target,
                    text_input_length)

        total_loss = self.ctc_weight * loss_ctc + self.s2s_weight * loss_s2s
        if self.use_diagonal_attention_prior:
            total_loss = total_loss + self.diagonal_attention_prior_weight * loss_diagonal
        if self.use_variance_adaptor and auxiliary_outputs:
            total_loss = total_loss + self.variance_loss_weight * duration_loss
        loss = total_loss
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.model.parameters(), 5)
        self.optimizer.step()
        self.scheduler.step()
        results = {'loss': loss.item(),
                   'ctc': loss_ctc.item(),
                   's2s': loss_s2s.item()}
        if self.use_variance_adaptor and auxiliary_outputs:
            results['duration'] = duration_loss.item()
        return results

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
            future_mask = self.model.get_future_mask(
                mel_input.size(2)//(2**self.model.n_down), unmask_future_steps=0).to(self.device)
            mel_mask = self.model.length_to_mask(mel_input_length)
            text_mask = self.model.length_to_mask(text_input_length)
            model_outputs = self.model(
                mel_input, src_key_padding_mask=mel_mask, text_input=text_input)
            if isinstance(model_outputs, (list, tuple)) and len(model_outputs) == 4:
                ppgs, s2s_pred, s2s_attn, auxiliary_outputs = model_outputs
            else:
                ppgs, s2s_pred, s2s_attn = model_outputs
                auxiliary_outputs = {}
            loss_ctc = self.criterion['ctc'](ppgs.log_softmax(dim=2).transpose(0, 1),
                                          text_input, mel_input_length, text_input_length)
            loss_s2s = 0
            for _s2s_pred, _text_input, _text_length in zip(s2s_pred, text_input, text_input_length):
                loss_s2s += self.criterion['ce'](_s2s_pred[:_text_length], _text_input[:_text_length])
            loss_s2s /= text_input.size(0)
            duration_loss = torch.tensor(0.0, device=self.device)
            if self.use_variance_adaptor and auxiliary_outputs:
                log_duration_pred = auxiliary_outputs.get('log_duration')
                if log_duration_pred is not None:
                    duration_targets = self._compute_duration_targets(
                        s2s_attn.detach(),
                        text_input_length,
                        mel_input_length)
                    log_duration_target = torch.log1p(duration_targets)
                    duration_loss = self._duration_loss(
                        log_duration_pred,
                        log_duration_target,
                        text_input_length)
            loss = loss_ctc + loss_s2s
            if self.use_variance_adaptor and auxiliary_outputs:
                loss = loss + self.variance_loss_weight * duration_loss

            eval_losses["eval/ctc"].append(loss_ctc.item())
            eval_losses["eval/s2s"].append(loss_s2s.item())
            eval_losses["eval/loss"].append(loss.item())
            if self.use_variance_adaptor and auxiliary_outputs:
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

    def _compute_duration_targets(self, attn, text_lengths, mel_lengths):
        """Convert attention maps into discrete duration targets."""
        batch_size = attn.size(0)
        max_text_len = int(text_lengths.max().item()) if text_lengths.numel() else 0
        device = attn.device
        durations = attn.new_zeros((batch_size, max_text_len))

        for idx in range(batch_size):
            text_len = int(text_lengths[idx].item())
            mel_len = int(mel_lengths[idx].item())
            if text_len == 0 or mel_len == 0:
                continue
            attn_slice = attn[idx, 1:1 + text_len, :mel_len]
            if attn_slice.numel() == 0:
                continue
            frame_assignments = attn_slice.argmax(dim=0)
            counts = torch.bincount(frame_assignments, minlength=text_len).float().to(device)
            durations[idx, :text_len] = counts

        return durations

    def _duration_loss(self, log_duration_pred, log_duration_target, text_lengths):
        """Compute masked error between predicted and target log durations."""
        if log_duration_pred.dim() == 3 and log_duration_pred.size(-1) == 1:
            log_duration_pred = log_duration_pred.squeeze(-1)

        mask = self.length_to_mask(text_lengths).to(log_duration_pred.device)
        mask = mask[:, :log_duration_pred.size(1)]

        if self.variance_loss_type == 'l1':
            loss = torch.abs(log_duration_pred - log_duration_target)
        elif self.variance_loss_type in ('smooth_l1', 'huber'):
            loss = F.smooth_l1_loss(log_duration_pred, log_duration_target, reduction='none')
        else:
            loss = (log_duration_pred - log_duration_target) ** 2

        loss = loss.masked_fill(mask, 0.0)
        denom = torch.clamp((~mask).sum(), min=1).float()
        return loss.sum() / denom