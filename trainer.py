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
                 auxiliary_objectives=None,
                 auxiliary_loss_weights=None):

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
        self.auxiliary_objectives = auxiliary_objectives or {}
        self.auxiliary_loss_weights = auxiliary_loss_weights or {}

        self.auxiliary_loss_fns = {}
        if self._is_objective_enabled('frame_phoneme'):
            self.auxiliary_loss_fns['frame_phoneme'] = nn.CrossEntropyLoss(ignore_index=-100)
        if self._is_objective_enabled('speaker_embedding'):
            self.auxiliary_loss_fns['speaker_embedding'] = nn.MSELoss()
        if self._is_objective_enabled('pronunciation_error'):
            self.auxiliary_loss_fns['pronunciation_error'] = nn.CrossEntropyLoss()

    def _is_objective_enabled(self, name):
        if not isinstance(self.auxiliary_objectives, dict):
            return False
        cfg = self.auxiliary_objectives.get(name, {})
        return isinstance(cfg, dict) and cfg.get('enabled', False)

    def _extract_batch(self, batch):
        aux_targets = {}
        if isinstance(batch, dict):
            text_input = batch['texts']
            text_input_length = batch['text_lengths']
            mel_input = batch['mels']
            mel_input_length = batch['mel_lengths']
            aux_targets = batch.get('auxiliary', {})
        else:
            core = batch
            if isinstance(batch, (list, tuple)) and batch and isinstance(batch[-1], dict):
                aux_targets = batch[-1]
                core = batch[:-1]
            if len(core) < 4:
                raise ValueError("Batch must contain at least four tensors")
            text_input, text_input_length, mel_input, mel_input_length = core[:4]
        return text_input, text_input_length, mel_input, mel_input_length, aux_targets

    def _move_aux_to_device(self, aux_targets):
        if not isinstance(aux_targets, dict):
            return {}
        moved = {}
        for key, value in aux_targets.items():
            if torch.is_tensor(value):
                moved[key] = value.to(self.device)
            else:
                moved[key] = value
        return moved

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
        text_input, text_input_length, mel_input, mel_input_length, aux_targets = self._extract_batch(batch)
        text_input = text_input.to(self.device)
        text_input_length = text_input_length.to(self.device)
        mel_input = mel_input.to(self.device)
        mel_input_length = mel_input_length.to(self.device)
        aux_targets = self._move_aux_to_device(aux_targets)
        mel_input_length = mel_input_length // (2 ** self.model.n_down)
        future_mask = self.model.get_future_mask(
            mel_input.size(2)//(2**self.model.n_down), unmask_future_steps=0).to(self.device)
        mel_mask = self.model.length_to_mask(mel_input_length)
        text_mask = self.model.length_to_mask(text_input_length)
        model_outputs = self.model(
            mel_input, src_key_padding_mask=mel_mask, text_input=text_input)
        if isinstance(model_outputs, tuple) and len(model_outputs) == 4:
            ppgs, s2s_pred, s2s_attn, aux_outputs = model_outputs
        else:
            ppgs, s2s_pred, s2s_attn = model_outputs
            aux_outputs = None

        if self.use_diagonal_attention_prior:
            loss_diagonal = diagonal_attention_prior(s2s_attn, text_input_length, mel_input_length)

        loss_ctc = self.criterion['ctc'](ppgs.log_softmax(dim=2).transpose(0, 1),
                                      text_input, mel_input_length, text_input_length)

        loss_s2s = 0
        for _s2s_pred, _text_input, _text_length in zip(s2s_pred, text_input, text_input_length):
            loss_s2s += self.criterion['ce'](_s2s_pred[:_text_length], _text_input[:_text_length])
        loss_s2s /= text_input.size(0)

        total_loss = self.ctc_weight * loss_ctc + self.s2s_weight * loss_s2s
        aux_losses = {}

        if isinstance(aux_outputs, dict) and aux_targets:
            if self._is_objective_enabled('frame_phoneme') and 'frame_phoneme' in aux_outputs and 'frame_phoneme' in aux_targets:
                logits = aux_outputs['frame_phoneme']
                targets = aux_targets['frame_phoneme']
                aux_losses['frame_phoneme'] = self.auxiliary_loss_fns['frame_phoneme'](
                    logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            if self._is_objective_enabled('speaker_embedding') and 'speaker_embedding' in aux_outputs and 'speaker_ids' in aux_targets:
                embeddings = aux_outputs['speaker_embedding']
                speaker_ids = aux_targets['speaker_ids']
                target_embeddings = self.model.get_speaker_reference_embeddings(speaker_ids)
                aux_losses['speaker_embedding'] = self.auxiliary_loss_fns['speaker_embedding'](embeddings, target_embeddings)
            if self._is_objective_enabled('pronunciation_error') and 'pronunciation_error' in aux_outputs and 'pronunciation_error' in aux_targets:
                logits = aux_outputs['pronunciation_error']
                targets = aux_targets['pronunciation_error']
                aux_losses['pronunciation_error'] = self.auxiliary_loss_fns['pronunciation_error'](logits, targets)

        for name, aux_loss in aux_losses.items():
            weight = float(self.auxiliary_loss_weights.get(name, 1.0))
            total_loss = total_loss + weight * aux_loss
        if self.use_diagonal_attention_prior:
            total_loss = total_loss + self.diagonal_attention_prior_weight * loss_diagonal
        loss = total_loss
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.model.parameters(), 5)
        self.optimizer.step()
        self.scheduler.step()
        loss_dict = {'loss': loss.item(),
                     'ctc': loss_ctc.item(),
                     's2s': loss_s2s.item()}
        for name, aux_loss in aux_losses.items():
            loss_dict[f'aux/{name}'] = aux_loss.item()
        return loss_dict

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
            text_input, text_input_length, mel_input, mel_input_length, aux_targets = self._extract_batch(batch)
            text_input = text_input.to(self.device)
            text_input_length = text_input_length.to(self.device)
            mel_input = mel_input.to(self.device)
            mel_input_length = mel_input_length.to(self.device)
            aux_targets = self._move_aux_to_device(aux_targets)
            mel_input_length = mel_input_length // (2 ** self.model.n_down)
            future_mask = self.model.get_future_mask(
                mel_input.size(2)//(2**self.model.n_down), unmask_future_steps=0).to(self.device)
            mel_mask = self.model.length_to_mask(mel_input_length)
            text_mask = self.model.length_to_mask(text_input_length)
            model_outputs = self.model(
                mel_input, src_key_padding_mask=mel_mask, text_input=text_input)
            if isinstance(model_outputs, tuple) and len(model_outputs) == 4:
                ppgs, s2s_pred, s2s_attn, aux_outputs = model_outputs
            else:
                ppgs, s2s_pred, s2s_attn = model_outputs
                aux_outputs = None
            loss_ctc = self.criterion['ctc'](ppgs.log_softmax(dim=2).transpose(0, 1),
                                          text_input, mel_input_length, text_input_length)
            loss_s2s = 0
            for _s2s_pred, _text_input, _text_length in zip(s2s_pred, text_input, text_input_length):
                loss_s2s += self.criterion['ce'](_s2s_pred[:_text_length], _text_input[:_text_length])
            loss_s2s /= text_input.size(0)
            loss = loss_ctc + loss_s2s
            aux_losses = {}
            if isinstance(aux_outputs, dict) and aux_targets:
                if self._is_objective_enabled('frame_phoneme') and 'frame_phoneme' in aux_outputs and 'frame_phoneme' in aux_targets:
                    logits = aux_outputs['frame_phoneme']
                    targets = aux_targets['frame_phoneme']
                    aux_losses['frame_phoneme'] = self.auxiliary_loss_fns['frame_phoneme'](
                        logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                if self._is_objective_enabled('speaker_embedding') and 'speaker_embedding' in aux_outputs and 'speaker_ids' in aux_targets:
                    embeddings = aux_outputs['speaker_embedding']
                    speaker_ids = aux_targets['speaker_ids']
                    target_embeddings = self.model.get_speaker_reference_embeddings(speaker_ids)
                    aux_losses['speaker_embedding'] = self.auxiliary_loss_fns['speaker_embedding'](embeddings, target_embeddings)
                if self._is_objective_enabled('pronunciation_error') and 'pronunciation_error' in aux_outputs and 'pronunciation_error' in aux_targets:
                    logits = aux_outputs['pronunciation_error']
                    targets = aux_targets['pronunciation_error']
                    aux_losses['pronunciation_error'] = self.auxiliary_loss_fns['pronunciation_error'](logits, targets)

            for name, aux_loss in aux_losses.items():
                weight = float(self.auxiliary_loss_weights.get(name, 1.0))
                loss = loss + weight * aux_loss

            eval_losses["eval/ctc"].append(loss_ctc.item())
            eval_losses["eval/s2s"].append(loss_s2s.item())
            eval_losses["eval/loss"].append(loss.item())
            for name, aux_loss in aux_losses.items():
                eval_losses[f"eval/aux/{name}"].append(aux_loss.item())

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