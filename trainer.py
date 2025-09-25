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

from utils import calc_wer

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from utils import *


class Trainer(object):
    def __init__(
        self,
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
        switch_sortagrad_dataset_epoch=10,
        use_diagonal_attention_prior=True,
        diagonal_attention_prior_weight=0.1,
        ctc_weight=1.0,
        s2s_weight=1.0,
        rnnt_weight=0.0,
        decoder_type="rnnt",
        blank_id=0,
    ):

        self.steps = initial_steps
        self.epochs = initial_epochs
        self.model = model
        self.criterion = criterion or {}
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
        self.decoder_type = (decoder_type or "rnnt").lower()
        self.use_diagonal_attention_prior = (
            bool(use_diagonal_attention_prior) and self.decoder_type == "s2s"
        )
        self.diagonal_attention_prior_weight = diagonal_attention_prior_weight
        self.maxm_mem_usage = 0
        self.ctc_weight = ctc_weight
        self.s2s_weight = s2s_weight
        self.rnnt_weight = rnnt_weight
        self.blank_id = blank_id

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
            except Exception:
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

    def _prepare_rnnt_inputs(self, text_input, text_lengths):
        batch_size = text_input.size(0)
        processed = []
        max_len = 0
        for idx in range(batch_size):
            length = int(text_lengths[idx].item())
            sequence = text_input[idx, :length].clone()
            if sequence.numel() > 0 and sequence[0].item() == self.blank_id:
                sequence = sequence[1:]
            if sequence.numel() > 0 and sequence[-1].item() == self.blank_id:
                sequence = sequence[:-1]
            processed.append(sequence)
            max_len = max(max_len, sequence.numel())

        max_len = max(max_len, 1)
        padded = text_input.new_full((batch_size, max_len), fill_value=self.blank_id)
        lengths = text_lengths.new_zeros(batch_size)
        for idx, seq in enumerate(processed):
            seq_len = seq.numel()
            if seq_len > 0:
                padded[idx, :seq_len] = seq
            lengths[idx] = seq_len
        return padded, lengths

    def _forward_model(self, mel_input, mel_lengths, text_input, text_lengths):
        mel_mask = self.model.length_to_mask(mel_lengths)
        model_kwargs = {"src_key_padding_mask": mel_mask}

        if self.decoder_type == "s2s":
            model_kwargs["text_input"] = text_input
            outputs = self.model(mel_input, **model_kwargs)
            return outputs, None, None

        if self.decoder_type == "rnnt":
            decoder_inputs, decoder_lengths = self._prepare_rnnt_inputs(text_input, text_lengths)
            model_kwargs.update({
                "decoder_inputs": decoder_inputs.to(self.device),
                "decoder_lengths": decoder_lengths.to(self.device),
            })
            outputs = self.model(mel_input, **model_kwargs)
            return outputs, decoder_inputs, decoder_lengths

        outputs = self.model(mel_input, **model_kwargs)
        return outputs, None, None

    def run(self, batch):
        self.optimizer.zero_grad()
        batch = [b.to(self.device) for b in batch]
        text_input, text_input_length, mel_input, mel_input_length = batch
        mel_input_length = mel_input_length // (2 ** self.model.n_down)

        outputs, decoder_inputs, decoder_lengths = self._forward_model(
            mel_input, mel_input_length, text_input, text_input_length
        )

        s2s_pred = None
        s2s_attn = None
        rnnt_logits = None
        if self.decoder_type == "s2s":
            ppgs, s2s_pred, s2s_attn = outputs
        elif self.decoder_type == "rnnt":
            ppgs, rnnt_logits = outputs
        else:
            ppgs = outputs

        loss_ctc = torch.tensor(0.0, device=self.device)
        if "ctc" in self.criterion and self.ctc_weight > 0:
            loss_ctc = self.criterion['ctc'](
                ppgs.log_softmax(dim=2).transpose(0, 1),
                text_input,
                mel_input_length,
                text_input_length,
            )

        loss_s2s = torch.tensor(0.0, device=self.device)
        if self.decoder_type == "s2s" and "ce" in self.criterion and self.s2s_weight > 0:
            for _s2s_pred, _text_input, _text_length in zip(s2s_pred, text_input, text_input_length):
                loss_s2s += self.criterion['ce'](
                    _s2s_pred[:_text_length], _text_input[:_text_length]
                )
            loss_s2s = loss_s2s / text_input.size(0)

        loss_rnnt = torch.tensor(0.0, device=self.device)
        if self.decoder_type == "rnnt" and "rnnt" in self.criterion and self.rnnt_weight > 0:
            if decoder_inputs is None or decoder_lengths is None:
                decoder_inputs, decoder_lengths = self._prepare_rnnt_inputs(text_input, text_input_length)
            loss_rnnt = self.criterion['rnnt'](
                rnnt_logits.log_softmax(dim=-1),
                decoder_inputs.to(self.device),
                mel_input_length,
                decoder_lengths.to(self.device),
            )

        total_loss = self.ctc_weight * loss_ctc
        if self.decoder_type == "s2s":
            total_loss = total_loss + self.s2s_weight * loss_s2s
        if self.decoder_type == "rnnt":
            total_loss = total_loss + self.rnnt_weight * loss_rnnt

        if self.decoder_type == "s2s" and self.use_diagonal_attention_prior:
            loss_diagonal = diagonal_attention_prior(s2s_attn, text_input_length, mel_input_length)
            total_loss = total_loss + self.diagonal_attention_prior_weight * loss_diagonal

        total_loss.backward()
        torch.nn.utils.clip_grad_value_(self.model.parameters(), 5)
        self.optimizer.step()
        self.scheduler.step()

        metrics = {'loss': total_loss.item()}
        if "ctc" in self.criterion and self.ctc_weight > 0:
            metrics['ctc'] = loss_ctc.item()
        if self.decoder_type == "s2s" and "ce" in self.criterion and self.s2s_weight > 0:
            metrics['s2s'] = loss_s2s.item()
        if self.decoder_type == "rnnt" and "rnnt" in self.criterion and self.rnnt_weight > 0:
            metrics['rnnt'] = loss_rnnt.item()
        return metrics

    def _train_epoch(self):
        if (self.epochs == self.switch_sortagrad_dataset_epoch) or (
            self.epochs == 0 and self.switch_sortagrad_dataset_epoch <= 0
        ):
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

            outputs, decoder_inputs, decoder_lengths = self._forward_model(
                mel_input, mel_input_length, text_input, text_input_length
            )

            s2s_pred = None
            s2s_attn = None
            rnnt_logits = None
            if self.decoder_type == "s2s":
                ppgs, s2s_pred, s2s_attn = outputs
            elif self.decoder_type == "rnnt":
                ppgs, rnnt_logits = outputs
            else:
                ppgs = outputs

            loss_ctc = torch.tensor(0.0, device=self.device)
            if "ctc" in self.criterion and self.ctc_weight > 0:
                loss_ctc = self.criterion['ctc'](
                    ppgs.log_softmax(dim=2).transpose(0, 1),
                    text_input,
                    mel_input_length,
                    text_input_length,
                )

            loss_s2s = torch.tensor(0.0, device=self.device)
            if self.decoder_type == "s2s" and "ce" in self.criterion and self.s2s_weight > 0:
                for _s2s_pred, _text_input, _text_length in zip(s2s_pred, text_input, text_input_length):
                    loss_s2s += self.criterion['ce'](
                        _s2s_pred[:_text_length], _text_input[:_text_length]
                    )
                loss_s2s = loss_s2s / text_input.size(0)

            loss_rnnt = torch.tensor(0.0, device=self.device)
            if self.decoder_type == "rnnt" and "rnnt" in self.criterion and self.rnnt_weight > 0:
                if decoder_inputs is None or decoder_lengths is None:
                    decoder_inputs, decoder_lengths = self._prepare_rnnt_inputs(text_input, text_input_length)
                loss_rnnt = self.criterion['rnnt'](
                    rnnt_logits.log_softmax(dim=-1),
                    decoder_inputs.to(self.device),
                    mel_input_length,
                    decoder_lengths.to(self.device),
                )

            total_loss = self.ctc_weight * loss_ctc
            if self.decoder_type == "s2s":
                total_loss = total_loss + self.s2s_weight * loss_s2s
            if self.decoder_type == "rnnt":
                total_loss = total_loss + self.rnnt_weight * loss_rnnt

            if self.decoder_type == "s2s" and self.use_diagonal_attention_prior:
                loss_diagonal = diagonal_attention_prior(
                    s2s_attn, text_input_length, mel_input_length
                )
                total_loss = total_loss + self.diagonal_attention_prior_weight * loss_diagonal

            if "ctc" in self.criterion and self.ctc_weight > 0:
                eval_losses["eval/ctc"].append(loss_ctc.item())
            if self.decoder_type == "s2s" and "ce" in self.criterion and self.s2s_weight > 0:
                eval_losses["eval/s2s"].append(loss_s2s.item())
            if self.decoder_type == "rnnt" and "rnnt" in self.criterion and self.rnnt_weight > 0:
                eval_losses["eval/rnnt"].append(loss_rnnt.item())
            eval_losses["eval/loss"].append(total_loss.item())

            _, amax_ppgs = torch.max(ppgs, dim=2)
            wers = [
                calc_wer(
                    target[:text_length],
                    pred[:mel_length],
                    ignore_indexes=list(range(5))
                )
                for target, pred, text_length, mel_length in zip(
                    text_input.cpu(),
                    amax_ppgs.cpu(),
                    text_input_length.cpu(),
                    mel_input_length.cpu(),
                )
            ]
            eval_losses["eval/wer"].extend(wers)

            if self.decoder_type == "s2s" and "ce" in self.criterion and self.s2s_weight > 0:
                _, amax_s2s = torch.max(s2s_pred, dim=2)
                acc = [
                    torch.eq(target[:length], pred[:length]).float().mean().item()
                    for target, pred, length in zip(
                        text_input.cpu(),
                        amax_s2s.cpu(),
                        text_input_length.cpu(),
                    )
                ]
                eval_losses["eval/acc"].extend(acc)

                if eval_steps_per_epoch <= 2:
                    eval_images["eval/image"].append(
                        self.get_image([s2s_attn[0].cpu().numpy()])
                    )

        eval_losses = {key: np.mean(value) for key, value in eval_losses.items()}
        eval_losses.update(eval_images)
        return eval_losses
