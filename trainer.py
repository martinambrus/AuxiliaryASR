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
                 train_dataloader=None,
                 val_dataloader=None,
                 initial_steps=0,
                 initial_epochs=0):

        self.steps = initial_steps
        self.epochs = initial_epochs
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config or {}
        self.device = device
        self.finish_train = False
        self.logger = logger
        self.fp16_run = False

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
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        self._load(state_dict["model"], self.model)

        if not load_only_params:
            self.steps = state_dict["steps"]
            self.epochs = state_dict["epochs"]
            self.optimizer.load_state_dict(state_dict["optimizer"])

            # overwrite schedular argument parameters
            state_dict["scheduler"].update(**self.config.get("scheduler_params", {}))
            self.scheduler.load_state_dict(state_dict["scheduler"])

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
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask+1, lengths.unsqueeze(1))
        return mask

    def _get_duration_config(self):
        aux_config = self.config.get('model_params', {}).get('auxiliary_models', {})
        variance_config = aux_config.get('variance_adaptor', {})
        enabled = variance_config.get('enabled', False)
        duration_config = variance_config.get('duration_predictor', {})
        duration_enabled = enabled and duration_config.get('enabled', True)
        return duration_enabled, duration_config

    def _compute_duration_targets(self, alignments, text_lengths, mel_lengths, detach=True):
        if alignments is None:
            return None
        if detach:
            alignments = alignments.detach()
        batch_size = alignments.size(0)
        max_text_len = alignments.size(1) - 1
        durations = alignments.new_zeros((batch_size, max_text_len))
        for idx in range(batch_size):
            text_len = int(text_lengths[idx].item())
            mel_len = int(mel_lengths[idx].item())
            if text_len <= 0 or mel_len <= 0:
                continue
            attn = alignments[idx, 1:text_len+1, :mel_len]
            if attn.numel() == 0:
                continue
            frame_to_token = torch.argmax(attn, dim=0)
            counts = torch.zeros(text_len, device=attn.device, dtype=attn.dtype)
            counts.scatter_add_(0, frame_to_token,
                                torch.ones_like(frame_to_token, dtype=attn.dtype))
            durations[idx, :text_len] = counts
        return durations

    def _compute_duration_loss(self, aux_outputs, alignments, text_mask, text_lengths, mel_lengths):
        duration_enabled, duration_config = self._get_duration_config()
        if not duration_enabled:
            return None
        if aux_outputs is None or 'duration_pred' not in aux_outputs:
            return None
        duration_pred = aux_outputs['duration_pred']
        if duration_pred is None:
            return None
        if text_mask is None:
            return None

        duration_targets = self._compute_duration_targets(
            alignments,
            text_lengths,
            mel_lengths,
            detach=duration_config.get('detach_alignment_grad', True))
        if duration_targets is None:
            return None
        duration_targets = duration_targets.to(device=duration_pred.device,
                                              dtype=duration_pred.dtype)

        duration_mask = text_mask.to(device=duration_pred.device, dtype=torch.bool)
        if duration_pred.size(1) != duration_mask.size(1):
            min_len = min(duration_pred.size(1), duration_mask.size(1))
            duration_pred = duration_pred[:, :min_len]
            duration_mask = duration_mask[:, :min_len]
            duration_targets = duration_targets[:, :min_len]

        valid_positions = ~duration_mask
        if valid_positions.sum() == 0:
            return None

        duration_target_log = torch.log(duration_targets + 1.0)
        duration_loss = F.mse_loss(
            duration_pred.masked_select(valid_positions),
            duration_target_log.masked_select(valid_positions))
        weight = duration_config.get('loss_weight', 0.0)
        if weight == 0.0:
            return None
        return duration_loss * weight

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
        self.optimizer.zero_grad()
        batch = [b.to(self.device) for b in batch]
        text_input, text_input_length, mel_input, mel_input_length = batch
        mel_input_length = mel_input_length // (2 ** self.model.n_down)
        future_mask = self.model.get_future_mask(
            mel_input.size(2)//(2**self.model.n_down), unmask_future_steps=0).to(self.device)
        mel_mask = self.model.length_to_mask(mel_input_length)
        text_mask = self.model.length_to_mask(text_input_length)
        model_outputs = self.model(
            mel_input,
            src_key_padding_mask=mel_mask,
            text_input=text_input,
            text_mask=text_mask)
        aux_outputs = {}
        if isinstance(model_outputs, tuple) and len(model_outputs) == 4:
            ppgs, s2s_pred, s2s_attn, aux_outputs = model_outputs
        else:
            ppgs, s2s_pred, s2s_attn = model_outputs
        
        loss_ctc = self.criterion['ctc'](ppgs.log_softmax(dim=2).transpose(0, 1),
                                      text_input, mel_input_length, text_input_length)

        loss_s2s = 0
        for _s2s_pred, _text_input, _text_length in zip(s2s_pred, text_input, text_input_length):
            loss_s2s += self.criterion['ce'](_s2s_pred[:_text_length], _text_input[:_text_length])
        loss_s2s /= text_input.size(0)

        loss = loss_ctc + loss_s2s

        aux_losses = {}
        duration_loss = self._compute_duration_loss(
            aux_outputs,
            s2s_attn,
            text_mask,
            text_input_length,
            mel_input_length)
        if duration_loss is not None:
            aux_losses['duration'] = duration_loss

        for value in aux_losses.values():
            loss = loss + value

        loss.backward()
        torch.nn.utils.clip_grad_value_(self.model.parameters(), 5)
        self.optimizer.step()
        self.scheduler.step()
        results = {'loss': loss.item(),
                   'ctc': loss_ctc.item(),
                   's2s': loss_s2s.item()}
        for key, value in aux_losses.items():
            results[f'aux/{key}'] = value.item()
        return results

    def _train_epoch(self):
        train_losses = defaultdict(list)
        self.model.train()
        for train_steps_per_epoch, batch in enumerate(tqdm(self.train_dataloader, desc="[train]"), 1):
            losses = self.run(batch)
            for key, value in losses.items():
                train_losses["train/%s" % key].append(value)

        train_losses = {key: np.mean(value) for key, value in train_losses.items()}
        train_losses['train/learning_rate'] = self._get_lr()
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
                mel_input,
                src_key_padding_mask=mel_mask,
                text_input=text_input,
                text_mask=text_mask)
            aux_outputs = {}
            if isinstance(model_outputs, tuple) and len(model_outputs) == 4:
                ppgs, s2s_pred, s2s_attn, aux_outputs = model_outputs
            else:
                ppgs, s2s_pred, s2s_attn = model_outputs
            loss_ctc = self.criterion['ctc'](ppgs.log_softmax(dim=2).transpose(0, 1),
                                          text_input, mel_input_length, text_input_length)
            loss_s2s = 0
            for _s2s_pred, _text_input, _text_length in zip(s2s_pred, text_input, text_input_length):
                loss_s2s += self.criterion['ce'](_s2s_pred[:_text_length], _text_input[:_text_length])
            loss_s2s /= text_input.size(0)
            loss = loss_ctc + loss_s2s

            duration_loss = self._compute_duration_loss(
                aux_outputs,
                s2s_attn,
                text_mask,
                text_input_length,
                mel_input_length)
            if duration_loss is not None:
                loss = loss + duration_loss
                eval_losses["eval/aux_duration"].append(duration_loss.item())

            eval_losses["eval/ctc"].append(loss_ctc.item())
            eval_losses["eval/s2s"].append(loss_s2s.item())
            eval_losses["eval/loss"].append(loss.item())

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