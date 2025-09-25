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
                 frame_weight=0.0,
                 speaker_weight=0.0,
                 pron_error_weight=0.0,
                 enable_frame_classifier=False,
                 enable_speaker=False,
                 enable_pronunciation_error=False):

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
        self.frame_weight = frame_weight
        self.speaker_weight = speaker_weight
        self.pron_error_weight = pron_error_weight
        self.enable_frame_classifier = enable_frame_classifier
        self.enable_speaker = enable_speaker
        self.enable_pronunciation_error = enable_pronunciation_error

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

    def _build_frame_targets(self, text_input, text_lengths, frame_lengths, max_frames):
        device = text_input.device
        batch_size = text_input.size(0)
        targets = torch.full((batch_size, max_frames), -1, device=device, dtype=torch.long)
        for idx in range(batch_size):
            text_len = int(text_lengths[idx].item())
            frame_len = int(frame_lengths[idx].item())
            if text_len <= 0 or frame_len <= 0:
                continue
            frame_len = min(frame_len, max_frames)
            effective_text = text_input[idx, :text_len]
            frame_indices = torch.linspace(0, text_len - 1, steps=frame_len, device=device)
            frame_indices = frame_indices.round().clamp(min=0, max=text_len - 1).long()
            targets[idx, :frame_len] = effective_text[frame_indices]
        return targets

    def _build_pronunciation_targets(self, text_input, text_lengths, max_length):
        device = text_input.device
        batch_size = text_input.size(0)
        targets = torch.full((batch_size, max_length), -1, device=device, dtype=torch.long)
        for idx in range(batch_size):
            length = int(text_lengths[idx].item())
            if length <= 0:
                continue
            max_assign = min(length, max_length)
            if max_assign <= 0:
                continue
            tokens = text_input[idx, :max_assign]
            pron_targets = torch.zeros(max_assign, device=device, dtype=torch.long)
            if max_assign > 1:
                repeated = tokens[1:] == tokens[:-1]
                pron_targets[1:][repeated] = 1
            targets[idx, :max_assign] = pron_targets
        return targets

    @staticmethod
    def _sequence_accuracy(predictions, targets):
        if predictions is None or targets is None:
            return 0.0
        mask = targets != -1
        if mask.sum().item() == 0:
            return 0.0
        correct = (predictions == targets) & mask
        return correct.float().sum().item() / mask.float().sum().item()

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
        processed_batch = []
        for element in batch:
            if hasattr(element, 'to'):
                processed_batch.append(element.to(self.device))
            else:
                processed_batch.append(element)
        batch = processed_batch

        text_input, text_input_length, mel_input, mel_input_length = batch[:4]
        if len(batch) > 4:
            speaker_ids = batch[4]
        else:
            speaker_ids = torch.zeros(text_input.size(0), device=self.device, dtype=torch.long)
        speaker_ids = speaker_ids.long()
        mel_input_length = mel_input_length // (2 ** self.model.n_down)
        future_mask = self.model.get_future_mask(
            mel_input.size(2)//(2**self.model.n_down), unmask_future_steps=0).to(self.device)
        mel_mask = self.model.length_to_mask(mel_input_length)
        text_mask = self.model.length_to_mask(text_input_length)
        model_outputs = self.model(
            mel_input, src_key_padding_mask=mel_mask, text_input=text_input)

        losses = {}
        total_loss = torch.zeros(1, device=self.device)

        ppgs = model_outputs.get('ctc_logits')
        if self.ctc_weight > 0 and ppgs is not None:
            loss_ctc = self.criterion['ctc'](ppgs.log_softmax(dim=2).transpose(0, 1),
                                          text_input, mel_input_length, text_input_length)
            total_loss = total_loss + self.ctc_weight * loss_ctc
            losses['ctc'] = loss_ctc.item()
        else:
            loss_ctc = torch.zeros(1, device=self.device)

        s2s_pred = model_outputs.get('s2s_logits')
        s2s_attn = model_outputs.get('s2s_attn')
        if self.s2s_weight > 0 and s2s_pred is not None:
            loss_s2s = 0
            for _s2s_pred, _text_input, _text_length in zip(s2s_pred, text_input, text_input_length):
                loss_s2s += self.criterion['ce'](_s2s_pred[:_text_length], _text_input[:_text_length])
            loss_s2s /= max(1, text_input.size(0))
            total_loss = total_loss + self.s2s_weight * loss_s2s
            losses['s2s'] = loss_s2s.item()
        else:
            loss_s2s = torch.zeros(1, device=self.device)

        if self.enable_frame_classifier and self.frame_weight > 0:
            frame_logits = model_outputs.get('frame_phoneme_logits')
            if frame_logits is not None:
                frame_targets = self._build_frame_targets(text_input, text_input_length, mel_input_length, frame_logits.size(1))
                frame_loss = self.criterion['frame_ce'](frame_logits.reshape(-1, frame_logits.size(2)),
                                                        frame_targets.view(-1))
                total_loss = total_loss + self.frame_weight * frame_loss
                losses['frame_phoneme'] = frame_loss.item()
            else:
                frame_loss = torch.zeros(1, device=self.device)
        else:
            frame_loss = torch.zeros(1, device=self.device)

        if self.enable_speaker and self.speaker_weight > 0:
            speaker_logits = model_outputs.get('speaker_logits')
            if speaker_logits is not None:
                loss_speaker = self.criterion['speaker_ce'](speaker_logits, speaker_ids)
                total_loss = total_loss + self.speaker_weight * loss_speaker
                speaker_pred = torch.argmax(speaker_logits, dim=1)
                speaker_acc = torch.eq(speaker_pred, speaker_ids).float().mean().item()
                losses['speaker'] = loss_speaker.item()
                losses['speaker_acc'] = speaker_acc
            else:
                loss_speaker = torch.zeros(1, device=self.device)
        else:
            loss_speaker = torch.zeros(1, device=self.device)

        if self.enable_pronunciation_error and self.pron_error_weight > 0:
            pron_logits = model_outputs.get('pron_error_logits')
            if pron_logits is not None:
                pron_targets = self._build_pronunciation_targets(text_input, text_input_length, pron_logits.size(1))
                pron_loss = self.criterion['pron_error_ce'](pron_logits.reshape(-1, pron_logits.size(2)),
                                                            pron_targets.view(-1))
                total_loss = total_loss + self.pron_error_weight * pron_loss
                pron_pred = torch.argmax(pron_logits, dim=2)
                pron_acc = self._sequence_accuracy(pron_pred, pron_targets)
                losses['pronunciation_error'] = pron_loss.item()
                losses['pronunciation_error_acc'] = pron_acc
            else:
                pron_loss = torch.zeros(1, device=self.device)
        else:
            pron_loss = torch.zeros(1, device=self.device)

        if self.use_diagonal_attention_prior and s2s_attn is not None:
            loss_diagonal = diagonal_attention_prior(s2s_attn, text_input_length, mel_input_length)
            total_loss = total_loss + self.diagonal_attention_prior_weight * loss_diagonal
            losses['diag_attn'] = loss_diagonal.item()
        
        loss = total_loss
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.model.parameters(), 5)
        self.optimizer.step()
        self.scheduler.step()
        losses['loss'] = loss.item()
        if 'ctc' not in losses:
            losses['ctc'] = loss_ctc.item()
        if 's2s' not in losses:
            losses['s2s'] = loss_s2s.item()
        return losses

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
            processed_batch = []
            for element in batch:
                if hasattr(element, 'to'):
                    processed_batch.append(element.to(self.device))
                else:
                    processed_batch.append(element)
            batch = processed_batch

            text_input, text_input_length, mel_input, mel_input_length = batch[:4]
            if len(batch) > 4:
                speaker_ids = batch[4].long()
            else:
                speaker_ids = torch.zeros(text_input.size(0), device=self.device, dtype=torch.long)
            mel_input_length = mel_input_length // (2 ** self.model.n_down)
            future_mask = self.model.get_future_mask(
                mel_input.size(2)//(2**self.model.n_down), unmask_future_steps=0).to(self.device)
            mel_mask = self.model.length_to_mask(mel_input_length)
            text_mask = self.model.length_to_mask(text_input_length)
            model_outputs = self.model(
                mel_input, src_key_padding_mask=mel_mask, text_input=text_input)

            ppgs = model_outputs.get('ctc_logits')
            s2s_pred = model_outputs.get('s2s_logits')
            s2s_attn = model_outputs.get('s2s_attn')

            if self.ctc_weight > 0 and ppgs is not None:
                loss_ctc = self.criterion['ctc'](ppgs.log_softmax(dim=2).transpose(0, 1),
                                              text_input, mel_input_length, text_input_length)
                eval_losses["eval/ctc"].append(loss_ctc.item())
            else:
                loss_ctc = torch.zeros(1, device=self.device)

            if self.s2s_weight > 0 and s2s_pred is not None:
                loss_s2s = 0
                for _s2s_pred, _text_input, _text_length in zip(s2s_pred, text_input, text_input_length):
                    loss_s2s += self.criterion['ce'](_s2s_pred[:_text_length], _text_input[:_text_length])
                loss_s2s /= max(1, text_input.size(0))
                eval_losses["eval/s2s"].append(loss_s2s.item())
            else:
                loss_s2s = torch.zeros(1, device=self.device)

            total_eval_loss = torch.zeros(1, device=self.device)
            if self.ctc_weight > 0:
                total_eval_loss = total_eval_loss + self.ctc_weight * loss_ctc
            if self.s2s_weight > 0:
                total_eval_loss = total_eval_loss + self.s2s_weight * loss_s2s

            if self.enable_frame_classifier and self.frame_weight > 0:
                frame_logits = model_outputs.get('frame_phoneme_logits')
                if frame_logits is not None:
                    frame_targets = self._build_frame_targets(text_input, text_input_length, mel_input_length, frame_logits.size(1))
                    frame_loss = self.criterion['frame_ce'](frame_logits.reshape(-1, frame_logits.size(2)),
                                                            frame_targets.view(-1))
                    total_eval_loss = total_eval_loss + self.frame_weight * frame_loss
                    eval_losses['eval/frame_phoneme'].append(frame_loss.item())

            if self.enable_speaker and self.speaker_weight > 0:
                speaker_logits = model_outputs.get('speaker_logits')
                if speaker_logits is not None:
                    loss_speaker = self.criterion['speaker_ce'](speaker_logits, speaker_ids)
                    total_eval_loss = total_eval_loss + self.speaker_weight * loss_speaker
                    eval_losses['eval/speaker'].append(loss_speaker.item())
                    speaker_pred = torch.argmax(speaker_logits, dim=1)
                    speaker_acc = torch.eq(speaker_pred, speaker_ids).float().mean().item()
                    eval_losses['eval/speaker_acc'].append(speaker_acc)

            if self.enable_pronunciation_error and self.pron_error_weight > 0:
                pron_logits = model_outputs.get('pron_error_logits')
                if pron_logits is not None:
                    pron_targets = self._build_pronunciation_targets(text_input, text_input_length, pron_logits.size(1))
                    pron_loss = self.criterion['pron_error_ce'](pron_logits.reshape(-1, pron_logits.size(2)),
                                                                pron_targets.view(-1))
                    total_eval_loss = total_eval_loss + self.pron_error_weight * pron_loss
                    eval_losses['eval/pronunciation_error'].append(pron_loss.item())
                    pron_pred = torch.argmax(pron_logits, dim=2)
                    pron_acc = self._sequence_accuracy(pron_pred, pron_targets)
                    eval_losses['eval/pronunciation_error_acc'].append(pron_acc)

            eval_losses["eval/loss"].append(total_eval_loss.item())

            if ppgs is not None:
                _, amax_ppgs = torch.max(ppgs, dim=2)
                wers = [calc_wer(target[:text_length],
                                 pred[:mel_length],
                                 ignore_indexes=list(range(5))) \
                        for target, pred, text_length, mel_length in zip(
                                text_input.cpu(), amax_ppgs.cpu(), text_input_length.cpu(), mel_input_length.cpu())]
                eval_losses["eval/wer"].extend(wers)

            if s2s_pred is not None:
                _, amax_s2s = torch.max(s2s_pred, dim=2)
                acc = [torch.eq(target[:length], pred[:length]).float().mean().item() \
                       for target, pred, length in zip(text_input.cpu(), amax_s2s.cpu(), text_input_length.cpu())]
                eval_losses["eval/acc"].extend(acc)

            if s2s_attn is not None and eval_steps_per_epoch <= 2:
                eval_images["eval/image"].append(
                    self.get_image([s2s_attn[0].cpu().numpy()]))

        eval_losses = {key: np.mean(value) for key, value in eval_losses.items()}
        eval_losses.update(eval_images)
        return eval_losses