# -*- coding: utf-8 -*-

import os
import os.path as osp
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
import inspect

import numpy as np
import torch
from torch import amp, nn
from PIL import Image
from tqdm import tqdm

from utils import calc_wer

import logging
from torch.distributions import Beta
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
                 accelerator=None,
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
                 diagonal_attention_prior_sigma = 0.5,
                 ctc_weight=1.0,
                 s2s_weight=1.0,
                 frame_weight=0.0,
                 speaker_weight=0.0,
                 pron_error_weight=0.0,
                 enable_frame_classifier=False,
                 enable_speaker=False,
                 enable_pronunciation_error=False,
                 ctc_blank_id=0,
                 ctc_logit_bias=0.0,
                 ctc_logit_temperature=1.0,
                 ctc_regularization_config=None,
                 mixspeech_config=None,
                 intermediate_ctc_config=None,
                 self_conditioned_ctc_config=None,
                 entropy_regularization_config=None,
                 memory_optimization_config=None,
                 steps_per_epoch=None):

        self.steps = initial_steps
        self.epochs = initial_epochs
        # Track whether the trainer is resuming from an existing checkpoint as
        # early as possible so downstream logic can safely query the flag even
        # if initialisation is interrupted before reaching the later
        # assignment. This also provides backwards compatibility with older
        # call-sites that expect the attribute to exist unconditionally.
        self._resumed_from_checkpoint = bool(initial_epochs)
        # Default SortaGrad state so callers can always rely on the attribute
        # existing even if initialisation aborts before the full dataloader
        # wiring runs.
        self._sortagrad_active = False
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
        self.accelerator = accelerator
        self.finish_train = False
        self.logger = logger
        self.fp16_run = False
        self.switch_sortagrad_dataset_epoch = switch_sortagrad_dataset_epoch
        self.use_diagonal_attention_prior = use_diagonal_attention_prior
        self._configure_diagonal_attention_weight(diagonal_attention_prior_weight)
        sigma = float(diagonal_attention_prior_sigma)
        self.diagonal_attention_prior_sigma = sigma if sigma > 0 else 0.5
        self.maxm_mem_usage = 0
        self.steps_per_epoch = steps_per_epoch if steps_per_epoch is None else int(steps_per_epoch)
        self.ctc_weight = ctc_weight
        self.s2s_weight = s2s_weight
        self.frame_weight = frame_weight
        self.speaker_weight = speaker_weight
        self.pron_error_weight = pron_error_weight
        self.enable_frame_classifier = enable_frame_classifier
        self.enable_speaker = enable_speaker
        self.enable_pronunciation_error = enable_pronunciation_error

        self.ctc_blank_id = int(ctc_blank_id)
        self.ctc_blank_logit_bias = float(ctc_logit_bias)
        temperature = float(ctc_logit_temperature)
        self.ctc_logit_temperature = temperature if temperature > 0 else 1.0

        ctc_reg_cfg = ctc_regularization_config or {}
        if not isinstance(ctc_reg_cfg, dict):
            ctc_reg_cfg = {}

        blank_reg_cfg = ctc_reg_cfg.get('blank_rate', {}) or {}
        if not isinstance(blank_reg_cfg, dict):
            blank_reg_cfg = {}
        self.ctc_blank_rate_regularization_enabled = bool(blank_reg_cfg.get('enabled', False))
        self.ctc_blank_rate_regularization_config = blank_reg_cfg

        coverage_reg_cfg = ctc_reg_cfg.get('coverage', {}) or {}
        if not isinstance(coverage_reg_cfg, dict):
            coverage_reg_cfg = {}
        self.ctc_coverage_regularization_enabled = bool(coverage_reg_cfg.get('enabled', False))
        self.ctc_coverage_regularization_config = coverage_reg_cfg

        self.mixspeech_config = mixspeech_config or {}
        self.mixspeech_enabled = bool(self.mixspeech_config.get('enabled', False))
        self.mixspeech_prob = float(self.mixspeech_config.get('prob', 0.5))
        self.mixspeech_alpha = float(self.mixspeech_config.get('alpha', 0.3))
        self.mixspeech_dominant = bool(self.mixspeech_config.get('dominant_mix', True))

        ictc_cfg = intermediate_ctc_config or {}
        self.intermediate_ctc_enabled = bool(ictc_cfg.get('enabled', False))
        self.intermediate_ctc_weight = float(ictc_cfg.get('loss_weight', 0.0))
        self.intermediate_ctc_layer_weights = self._parse_intermediate_layer_weights(ictc_cfg.get('layers'))

        sctc_cfg = self_conditioned_ctc_config or {}
        self.self_conditioned_ctc_enabled = bool(sctc_cfg.get('enabled', False))
        self.self_conditioned_ctc_weight = float(sctc_cfg.get('loss_weight', 0.0))
        self.self_conditioned_ctc_layer_weights = self._parse_intermediate_layer_weights(sctc_cfg.get('layers'))

        entropy_cfg = entropy_regularization_config or {}
        self.entropy_regularization_enabled = bool(entropy_cfg.get('enabled', False))
        mode = str(entropy_cfg.get('mode', 'minimize')).lower()
        if mode not in ('minimize', 'maximize'):
            mode = 'minimize'
        self.entropy_regularization_mode = mode
        self.entropy_regularization_eps = float(entropy_cfg.get('eps', 1.0e-6))
        targets_cfg = entropy_cfg.get('targets', {}) if isinstance(entropy_cfg, dict) else {}
        self.entropy_regularization_targets = self._parse_entropy_regularization_targets(targets_cfg, entropy_cfg)

        memopt_cfg = memory_optimization_config or {}
        if not isinstance(memopt_cfg, dict):
            memopt_cfg = {}
        lazy_masks_cfg = memopt_cfg.get('lazy_masks', {}) or {}
        if not isinstance(lazy_masks_cfg, dict):
            lazy_masks_cfg = {}
        lazy_enabled = bool(lazy_masks_cfg.get('enabled', True))
        self.skip_future_mask_allocation = lazy_enabled and bool(lazy_masks_cfg.get('future_mask', True))
        self.skip_text_mask_allocation = lazy_enabled and bool(lazy_masks_cfg.get('text_mask', True))

        precision_cfg = {}
        if isinstance(self.config, dict):
            precision_cfg = self.config.get('precision', {}) or {}
            if not isinstance(precision_cfg, dict):
                precision_cfg = {}

        mp_cfg = precision_cfg.get('mixed_precision', {}) if isinstance(precision_cfg, dict) else {}
        if not isinstance(mp_cfg, dict):
            mp_cfg = {}

        dtype_map = {
            'float16': torch.float16,
            'fp16': torch.float16,
            'half': torch.float16,
            'bfloat16': torch.bfloat16,
            'bf16': torch.bfloat16,
        }
        dtype_key = str(mp_cfg.get('dtype', 'float16')).lower()
        self.mixed_precision_dtype = dtype_map.get(dtype_key, torch.float16)

        autocast_requested = bool(mp_cfg.get('enabled', False))
        device_type = self.device.type if isinstance(self.device, torch.device) else str(self.device)
        if not isinstance(device_type, str):
            device_type = str(device_type)
        device_type = device_type.lower()
        is_cuda_device = device_type.startswith('cuda')
        cuda_available = torch.cuda.is_available()

        self.autocast_device_type = 'cuda' if is_cuda_device else device_type or 'cuda'
        self.autocast_enabled = autocast_requested and is_cuda_device and cuda_available

        grad_scaler_cfg = mp_cfg.get('grad_scaler', {}) if isinstance(mp_cfg, dict) else {}
        if not isinstance(grad_scaler_cfg, dict):
            grad_scaler_cfg = {}

        scaler_enabled = bool(grad_scaler_cfg.get('enabled', True)) and self.autocast_enabled
        if scaler_enabled:
            init_scale = float(grad_scaler_cfg.get('init_scale', 65536.0))
            growth_factor = float(grad_scaler_cfg.get('growth_factor', 2.0))
            backoff_factor = float(grad_scaler_cfg.get('backoff_factor', 0.5))
            growth_interval = int(grad_scaler_cfg.get('growth_interval', 2000))
            scaler_kwargs = dict(
                enabled=True,
                init_scale=init_scale,
                growth_factor=growth_factor,
                backoff_factor=backoff_factor,
                growth_interval=growth_interval,
            )

            try:
                scaler_params = inspect.signature(amp.GradScaler.__init__).parameters
            except (ValueError, TypeError):
                scaler_params = {}

            if 'device_type' in scaler_params:
                scaler_kwargs['device_type'] = self.autocast_device_type

            self.grad_scaler = amp.GradScaler(**scaler_kwargs)
        else:
            self.grad_scaler = None

    def _configure_diagonal_attention_weight(self, weight_config):
        schedule = None
        if isinstance(weight_config, dict):
            initial = float(weight_config.get('initial', weight_config.get('base', weight_config.get('start', 0.0))))
            target = float(weight_config.get('target', weight_config.get('final', initial)))
            warmup = int(weight_config.get('warmup_epochs', weight_config.get('ramp_epochs', 0)))
            hold = int(weight_config.get('hold_epochs', 0))
            schedule = {
                'initial': initial,
                'target': target,
                'warmup': max(0, warmup),
                'hold': max(0, hold),
            }
            base_weight = initial
        else:
            base_weight = float(weight_config) if weight_config is not None else 0.0

        self.diagonal_attention_prior_weight = float(base_weight)
        self.diagonal_attention_weight_schedule = schedule
        self._active_diagonal_attention_weight = float(base_weight)

    def _current_diagonal_attention_weight(self, epoch_index: int) -> float:
        schedule = self.diagonal_attention_weight_schedule
        if not schedule:
            return float(self.diagonal_attention_prior_weight)

        epoch = max(1, int(epoch_index))
        initial = float(schedule['initial'])
        target = float(schedule['target'])
        warmup = int(schedule['warmup'])
        hold = int(schedule['hold'])

        if warmup <= 0:
            weight = target
        else:
            progress = min(1.0, max(0.0, (epoch - 1) / float(max(1, warmup))))
            weight = initial + (target - initial) * progress

        if epoch > warmup + hold:
            weight = target

        return float(weight)

    def _set_active_diagonal_attention_weight(self, training: bool) -> None:
        if not self.use_diagonal_attention_prior:
            self._active_diagonal_attention_weight = 0.0
            return

        if training:
            epoch_index = self.epochs + 1
        else:
            epoch_index = max(1, self.epochs)

        weight = self._current_diagonal_attention_weight(epoch_index)
        self._active_diagonal_attention_weight = weight

    def _get_active_diagonal_attention_weight(self) -> float:
        return float(getattr(self, '_active_diagonal_attention_weight', self.diagonal_attention_prior_weight))

        self._scheduler_aligned = False
        self._optimizer_step_count = self._get_optimizer_step_count()
        self._sortagrad_active = bool(
            self.sorted_train_dataloader
            and self.shuffled_train_dataloader
            and self.switch_sortagrad_dataset_epoch is not None
            and self.switch_sortagrad_dataset_epoch > 0
        )

        if not self._sortagrad_active:
            if self.shuffled_train_dataloader is not None:
                self.train_dataloader = self.shuffled_train_dataloader
            if self.shuffled_val_dataloader is not None:
                self.val_dataloader = self.shuffled_val_dataloader

    def _adjust_ctc_logits(self, logits):
        """Apply optional blank bias and temperature scaling to CTC logits."""
        if logits is None:
            return logits

        bias = self.ctc_blank_logit_bias
        temperature = self.ctc_logit_temperature

        adjusted = logits
        if bias != 0.0 and 0 <= self.ctc_blank_id < logits.size(-1):
            adjusted = logits.clone()
            adjusted[..., self.ctc_blank_id] = adjusted[..., self.ctc_blank_id] - bias
        if temperature != 1.0:
            adjusted = adjusted / temperature

        return adjusted

    def _ctc_regularization_warmup_passed(self, cfg):
        if cfg is None:
            return False
        warmup = int(cfg.get('warmup_epochs', 0))
        if warmup <= 0:
            return True
        return self.epochs >= max(warmup, 0)

    def _compute_ctc_alignment_regularization(self, logits, input_lengths, target_lengths):
        if logits is None:
            return {}, {}

        enabled = self.ctc_blank_rate_regularization_enabled or self.ctc_coverage_regularization_enabled
        if not enabled:
            return {}, {}

        adjusted = self._adjust_ctc_logits(logits)
        probs = torch.softmax(adjusted.float(), dim=-1)

        reg_losses = {}
        reg_stats = {}

        if self.ctc_blank_rate_regularization_enabled:
            blank_cfg = self.ctc_blank_rate_regularization_config
            if self._ctc_regularization_warmup_passed(blank_cfg):
                blank_result = self._compute_ctc_blank_rate_regularization(probs, input_lengths, blank_cfg)
                if blank_result is not None:
                    loss_value, stats = blank_result
                    reg_losses['ctc/blank_rate_reg'] = loss_value
                    reg_stats.update(stats)

        if self.ctc_coverage_regularization_enabled:
            coverage_cfg = self.ctc_coverage_regularization_config
            if self._ctc_regularization_warmup_passed(coverage_cfg):
                coverage_result = self._compute_ctc_coverage_regularization(
                    probs, input_lengths, target_lengths, coverage_cfg
                )
                if coverage_result is not None:
                    loss_value, stats = coverage_result
                    reg_losses['ctc/coverage_reg'] = loss_value
                    reg_stats.update(stats)

        return reg_losses, reg_stats

    def _compute_ctc_blank_rate_regularization(self, probs, input_lengths, cfg):
        blank_id = self.ctc_blank_id
        if blank_id < 0 or blank_id >= probs.size(-1):
            return None

        weight = float(cfg.get('weight', 0.0))
        if weight == 0.0:
            return None

        blank_probs = probs[..., blank_id]
        if blank_probs.dim() != 2:
            return None

        mask = None
        if input_lengths is not None:
            if not torch.is_tensor(input_lengths):
                input_lengths = torch.as_tensor(input_lengths, device=blank_probs.device)
            input_lengths = input_lengths.to(device=blank_probs.device, dtype=torch.long)
            max_len = blank_probs.size(1)
            mask = torch.arange(max_len, device=blank_probs.device).unsqueeze(0)
            mask = mask < input_lengths.unsqueeze(1)
            mask = mask.to(blank_probs.dtype)

        if mask is not None:
            effective = mask.sum(dim=1).clamp_min(1.0)
            blank_rate = (blank_probs * mask).sum(dim=1) / effective
        else:
            blank_rate = blank_probs.mean(dim=1)

        target = float(cfg.get('target', cfg.get('target_blank_rate', 0.65)))
        tolerance = float(cfg.get('tolerance', 0.0))
        upper = min(max(target + tolerance, 0.0), 1.0)
        lower = max(min(target - tolerance, 1.0), 0.0)

        over_penalty = torch.clamp(blank_rate - upper, min=0.0)
        if bool(cfg.get('penalize_low_blank', False)):
            under_penalty = torch.clamp(lower - blank_rate, min=0.0)
        else:
            under_penalty = torch.zeros_like(over_penalty)

        penalty = over_penalty + under_penalty
        loss_value = penalty.mean() * weight

        stats = {
            'diagnostics/ctc_blank_rate': float(blank_rate.mean().detach().item()),
        }
        return loss_value, stats

    def _compute_ctc_coverage_regularization(self, probs, input_lengths, target_lengths, cfg):
        blank_id = self.ctc_blank_id
        if blank_id < 0 or blank_id >= probs.size(-1):
            return None

        weight = float(cfg.get('weight', 0.0))
        if weight == 0.0:
            return None

        if target_lengths is None:
            return None
        if not torch.is_tensor(target_lengths):
            target_lengths = torch.as_tensor(target_lengths, device=probs.device)
        target_lengths = target_lengths.to(device=probs.device, dtype=torch.float32)

        non_blank = 1.0 - probs[..., blank_id]
        if non_blank.dim() != 2:
            return None

        mask = None
        if input_lengths is not None:
            if not torch.is_tensor(input_lengths):
                input_lengths = torch.as_tensor(input_lengths, device=non_blank.device)
            input_lengths = input_lengths.to(device=non_blank.device, dtype=torch.long)
            max_len = non_blank.size(1)
            mask = torch.arange(max_len, device=non_blank.device).unsqueeze(0)
            mask = mask < input_lengths.unsqueeze(1)
            mask = mask.to(non_blank.dtype)

        if mask is not None:
            coverage_mass = (non_blank * mask).sum(dim=1)
        else:
            coverage_mass = non_blank.sum(dim=1)

        denom = target_lengths.clamp_min(1.0)
        coverage_ratio = coverage_mass / denom

        min_ratio = float(cfg.get('min_ratio', 1.0))
        tolerance = float(cfg.get('tolerance', 0.0))
        lower_bound = max(min_ratio - tolerance, 0.0)
        penalty = torch.clamp(lower_bound - coverage_ratio, min=0.0)

        max_ratio = cfg.get('max_ratio', None)
        if max_ratio is not None:
            max_ratio = float(max_ratio)
            upper_bound = max_ratio + max(0.0, tolerance)
            penalty = penalty + torch.clamp(coverage_ratio - upper_bound, min=0.0)

        loss_value = penalty.mean() * weight

        stats = {
            'diagnostics/ctc_coverage_ratio': float(coverage_ratio.mean().detach().item()),
        }
        return loss_value, stats

    def _get_target_model(self):
        """Return the underlying model, unwrapping accelerator/DDP wrappers."""
        if self.accelerator is not None:
            try:
                return self.accelerator.unwrap_model(self.model)
            except (AttributeError, ValueError):
                pass
        if hasattr(self.model, 'module'):
            return self.model.module
        return self.model

    def _get_optimizer_step_count(self):
        if self.optimizer is None:
            return 0
        try:
            return max(int(getattr(self.optimizer, "_step_count", 0)), 0)
        except (TypeError, ValueError):
            return 0

    def update_dataloaders(
        self,
        sorted_train=None,
        shuffled_train=None,
        sorted_val=None,
        shuffled_val=None,
        steps_per_epoch=None,
    ):
        """Update dataloaders when curriculum settings change."""

        if sorted_train is not None:
            self.sorted_train_dataloader = sorted_train
            if self._sortagrad_active:
                self.train_dataloader = sorted_train

        if shuffled_train is not None:
            self.shuffled_train_dataloader = shuffled_train
            if not self._sortagrad_active:
                self.train_dataloader = shuffled_train

        if sorted_val is not None:
            self.sorted_val_dataloader = sorted_val
            if self._sortagrad_active:
                self.val_dataloader = sorted_val

        if shuffled_val is not None:
            self.shuffled_val_dataloader = shuffled_val
            if not self._sortagrad_active:
                self.val_dataloader = shuffled_val

        if steps_per_epoch is not None:
            self.steps_per_epoch = int(steps_per_epoch)

    def _reduce_scalar(self, value):
        if self.accelerator is None:
            return float(value)
        tensor = torch.tensor([float(value)], device=self.device, dtype=torch.float32)
        reduced = self.accelerator.reduce(tensor, reduction="mean")
        return reduced.item()

    def _gather_metric_list(self, values):
        values = list(values)
        if self.accelerator is None:
            return values
        local_count = torch.tensor([len(values)], device=self.device, dtype=torch.long)
        counts = self.accelerator.gather(local_count)
        max_count = int(counts.max().item())
        if max_count == 0:
            return []
        padded = torch.zeros(max_count, device=self.device, dtype=torch.float32)
        if values:
            padded[:len(values)] = torch.tensor(values, device=self.device, dtype=torch.float32)
        gathered = self.accelerator.gather(padded)
        results = []
        offset = 0
        for count in counts.cpu().tolist():
            if count > 0:
                segment = gathered[offset:offset + max_count][:count]
                results.extend(segment.cpu().tolist())
            offset += max_count
        return results

    @staticmethod
    def _parse_intermediate_layer_weights(layers_config):
        weights = {}
        if layers_config is None:
            return weights

        if isinstance(layers_config, dict):
            iterator = layers_config.items()
        else:
            iterator = []
            for entry in layers_config:
                if isinstance(entry, dict):
                    idx = entry.get('index', entry.get('layer'))
                    weight = entry.get('weight', entry.get('loss_weight', 1.0))
                else:
                    idx = entry
                    weight = 1.0
                iterator.append((idx, weight))

        for idx, weight in iterator:
            try:
                layer_idx = int(idx)
                weights[str(layer_idx)] = float(weight)
            except (TypeError, ValueError):
                continue

        return weights

    @staticmethod
    def _parse_entropy_regularization_targets(targets_config, root_config):
        supported = ('ctc', 's2s')
        parsed = {}
        for key in supported:
            cfg = {}
            if isinstance(targets_config, dict):
                raw = targets_config.get(key, {})
            else:
                raw = {}
            if isinstance(raw, bool):
                cfg['enabled'] = raw
            elif isinstance(raw, (int, float)):
                cfg['weight'] = float(raw)
            elif isinstance(raw, dict):
                cfg.update(raw)
            weight = float(cfg.get('weight', root_config.get(f'{key}_weight', 0.0) if isinstance(root_config, dict) else 0.0))
            enabled = cfg.get('enabled', None)
            if enabled is None:
                enabled = bool(weight)
            parsed[key] = {
                'enabled': bool(enabled),
                'weight': weight,
                'length_normalize': bool(cfg.get('length_normalize', True)),
                'reduction': str(cfg.get('reduction', 'mean')).lower(),
            }
        return parsed

    @staticmethod
    def _collapse_tokens(tokens, ignore_indexes=None):
        if ignore_indexes is None:
            ignore_indexes = []
        filtered = [int(tok) for tok in tokens if int(tok) not in ignore_indexes]
        if not filtered:
            return []
        collapsed = [filtered[0]]
        for tok in filtered[1:]:
            if tok != collapsed[-1]:
                collapsed.append(tok)
        return collapsed

    def _entropy_target_config(self, key):
        if not self.entropy_regularization_enabled:
            return None
        cfg = self.entropy_regularization_targets.get(key, {})
        if not cfg.get('enabled', False):
            return None
        weight = float(cfg.get('weight', 0.0))
        if weight == 0.0:
            return None
        return cfg

    def _maybe_create_future_mask(self, length: int):
        if self.skip_future_mask_allocation:
            return None
        if length is None or length <= 0:
            return None
        target_model = self._get_target_model()
        mask = target_model.get_future_mask(length, unmask_future_steps=0)
        if mask is None:
            return None
        return mask.to(self.device) if hasattr(mask, 'to') else mask

    def _maybe_create_text_mask(self, lengths):
        if self.skip_text_mask_allocation:
            return None
        if lengths is None:
            return None
        target_model = self._get_target_model()
        return target_model.length_to_mask(lengths)

    def _compute_entropy_regularization(self, logits, lengths=None, key='ctc'):
        cfg = self._entropy_target_config(key)
        if cfg is None:
            return None

        if key == 'ctc':
            logits = self._adjust_ctc_logits(logits)

        logits = logits.float()
        probs = torch.softmax(logits, dim=-1)
        eps = max(self.entropy_regularization_eps, 0.0)
        if eps > 0.0:
            probs = probs.clamp_min(eps)
            probs = probs / probs.sum(dim=-1, keepdim=True)
        log_probs = torch.log(probs)
        entropy = -(probs * log_probs).sum(dim=-1)

        if lengths is not None:
            if lengths.dim() == 1:
                max_steps = entropy.size(1)
                mask = torch.arange(max_steps, device=entropy.device).unsqueeze(0)
                mask = mask < lengths.unsqueeze(1)
            else:
                mask = lengths
                if mask.dim() < entropy.dim():
                    mask = mask.unsqueeze(-1)
            mask = mask.to(entropy.dtype)
            entropy = entropy * mask
            if cfg.get('length_normalize', True):
                reduce_dims = tuple(range(1, entropy.dim()))
                denom = mask.sum(dim=reduce_dims)
                denom = denom.clamp_min(1.0)
                entropy = entropy.sum(dim=reduce_dims) / denom
            else:
                entropy = entropy.sum(dim=tuple(range(1, entropy.dim())))
        else:
            entropy = entropy.mean(dim=tuple(range(1, entropy.dim())))

        reduction = cfg.get('reduction', 'mean')
        if reduction == 'sum':
            entropy_value = entropy.sum()
        elif reduction == 'none':
            entropy_value = entropy
        else:
            entropy_value = entropy.mean()

        if self.entropy_regularization_mode == 'maximize':
            entropy_value = -entropy_value

        if isinstance(entropy_value, torch.Tensor) and entropy_value.dim() > 0:
            entropy_value = entropy_value.mean()

        reg_weight = float(cfg.get('weight', 0.0))
        reg_loss = entropy_value * reg_weight
        if not torch.is_tensor(reg_loss):
            reg_loss = torch.tensor(reg_loss, device=logits.device, dtype=logits.dtype)
        else:
            reg_loss = reg_loss.to(device=logits.device, dtype=logits.dtype)

        return reg_loss

    def _switch_to_shuffled(self, reason=None):
        if self.shuffled_train_dataloader is None:
            return

        self.train_dataloader = self.shuffled_train_dataloader
        if self.shuffled_val_dataloader is not None:
            self.val_dataloader = self.shuffled_val_dataloader

        if self._sortagrad_active:
            if reason:
                self.logger.info("")
                self.logger.info(reason)
                self.logger.info("")
            else:
                self.logger.info("")
                self.logger.info("[SortaGrad]: switching sorted to shuffled dataloader at configured epoch %i" % self.epochs)
                self.logger.info("")

        self._sortagrad_active = False

    def handle_sortagrad_after_resume(self):
        if not self._sortagrad_active:
            return

        if self.switch_sortagrad_dataset_epoch is None:
            self._switch_to_shuffled("[SortaGrad]: resuming with shuffled dataloader (no switch epoch configured)")
            return

        if self.switch_sortagrad_dataset_epoch <= 0 or self.epochs >= self.switch_sortagrad_dataset_epoch:
            self._switch_to_shuffled(
                "[SortaGrad]: resuming from epoch %i, switching to shuffled dataloader immediately" % self.epochs
            )

    def sync_scheduler_to_progress(self, completed_steps=None):
        if self._scheduler_aligned:
            return

        if not self._resumed_from_checkpoint:
            return

        if self.scheduler is None:
            return

        if completed_steps is None:
            if self.steps_per_epoch is None or self.steps_per_epoch <= 0:
                return
            completed_steps = int(self.epochs * self.steps_per_epoch)
        else:
            completed_steps = int(completed_steps)
        if completed_steps <= 0:
            return

        target_step = completed_steps - 1
        if target_step < 0:
            return

        total_steps = getattr(self.scheduler, 'total_steps', None)
        if isinstance(total_steps, int) and total_steps > 0:
            target_step = min(target_step, max(total_steps - 1, 0))

        if hasattr(self.scheduler, '_step_count'):
            self.scheduler._step_count = max(target_step, 1)

        recorded_steps = getattr(self.optimizer, '_step_count', 0)
        if recorded_steps < target_step + 1:
            setattr(self.optimizer, '_step_count', target_step + 1)
        self._optimizer_step_count = max(self._optimizer_step_count, target_step + 1)

        try:
            self.scheduler.step(target_step)
        except TypeError:
            if hasattr(self.scheduler, 'last_epoch'):
                self.scheduler.last_epoch = target_step
            if hasattr(self.scheduler, '_get_lr_called_within_step'):
                previous_flag = self.scheduler._get_lr_called_within_step
                self.scheduler._get_lr_called_within_step = True
            else:
                previous_flag = None

            try:
                lrs = self.scheduler.get_lr()
            finally:
                if previous_flag is not None:
                    self.scheduler._get_lr_called_within_step = previous_flag

            if hasattr(self.scheduler, '_last_lr'):
                self.scheduler._last_lr = lrs
            for param_group, lr in zip(self.optimizer.param_groups, lrs):
                param_group['lr'] = lr

        self._scheduler_aligned = True
        if self.logger:
            self.logger.info("")
            self.logger.info(
                "[Scheduler]: resumed from epoch %i, skipping warm-up by advancing to step %i" % (self.epochs, target_step)
            )
            self.logger.info("")

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
        model_to_save = self.model
        if self.accelerator is not None:
            model_to_save = self.accelerator.unwrap_model(self.model)
        state_dict["model"] = model_to_save.state_dict()

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
        target_model = self.accelerator.unwrap_model(self.model) if self.accelerator is not None else self.model
        self._load(state_dict["model"], target_model)

        if not load_only_params:
            self.steps = state_dict["steps"]
            self.epochs = state_dict["epochs"]
            self.optimizer.load_state_dict(state_dict["optimizer"])
            self.logger.info("Starting training from epoch %i" % state_dict["epochs"])

            # overwrite schedular argument parameters
            state_dict["scheduler"].update(**self.config.get("scheduler_params", {}))
            self.scheduler.load_state_dict(state_dict["scheduler"])

        self._optimizer_step_count = self._get_optimizer_step_count()

        if not load_only_params:
            self._resumed_from_checkpoint = True

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

    def _apply_mixspeech(self, mel_input):
        if not self.mixspeech_enabled or mel_input.size(0) < 2 or self.mixspeech_prob <= 0.0:
            return mel_input, None

        device = mel_input.device
        batch_size = mel_input.size(0)
        mix_mask = torch.rand(batch_size, device=device) < self.mixspeech_prob
        if not mix_mask.any():
            return mel_input, None

        perm = torch.randperm(batch_size, device=device)
        if self.mixspeech_alpha > 0.0:
            beta = Beta(self.mixspeech_alpha, self.mixspeech_alpha)
            lam = beta.sample((batch_size,)).to(device=device, dtype=mel_input.dtype)
        else:
            lam = torch.full((batch_size,), 0.5, device=device, dtype=mel_input.dtype)

        mixed = mel_input.clone()
        lam_primary = torch.ones(batch_size, device=device, dtype=mel_input.dtype)
        lam_secondary = torch.zeros(batch_size, device=device, dtype=mel_input.dtype)
        secondary_indices = torch.full((batch_size,), -1, device=device, dtype=torch.long)
        applied = False

        for idx in range(batch_size):
            if not mix_mask[idx] or perm[idx].item() == idx:
                continue
            partner = int(perm[idx].item())
            lam_val = lam[idx]
            if self.mixspeech_dominant:
                lam_val = torch.max(lam_val, 1.0 - lam_val)
            mixed[idx] = lam_val * mel_input[idx] + (1.0 - lam_val) * mel_input[partner]
            lam_primary[idx] = lam_val
            lam_secondary[idx] = 1.0 - lam_val
            secondary_indices[idx] = partner
            applied = True

        if not applied:
            return mel_input, None

        metadata = {
            'lam_primary': lam_primary,
            'lam_secondary': lam_secondary,
            'secondary_indices': secondary_indices,
            'avg_lambda': float(lam_primary.mean().item()),
        }
        return mixed, metadata

    def _compute_ctc_loss(self, logits, targets, input_lengths, target_lengths, mix_metadata=None):
        logits = self._adjust_ctc_logits(logits)
        ctc = self.criterion['ctc']
        log_probs = logits.log_softmax(dim=2).transpose(0, 1)
        primary_loss = ctc(log_probs, targets, input_lengths, target_lengths)
        if primary_loss.dim() == 0:
            primary_loss = primary_loss.unsqueeze(0)

        target_lengths = target_lengths.to(primary_loss.device)
        normalizer = target_lengths.to(primary_loss.dtype).clamp_min(1)
        primary_loss = primary_loss / normalizer

        total_loss = primary_loss
        if mix_metadata and mix_metadata.get('lam_secondary') is not None:
            lam_primary = mix_metadata['lam_primary'].to(primary_loss.dtype)
            lam_secondary = mix_metadata['lam_secondary'].to(primary_loss.dtype)
            secondary_indices = mix_metadata['secondary_indices']
            if torch.any(lam_secondary > 0):
                fallback = torch.arange(targets.size(0), device=targets.device, dtype=torch.long)
                secondary_indices = torch.where(secondary_indices >= 0, secondary_indices, fallback)
                secondary_targets = targets.index_select(0, secondary_indices)
                secondary_lengths = target_lengths.index_select(0, secondary_indices)
                secondary_loss = ctc(log_probs, secondary_targets, input_lengths, secondary_lengths)
                if secondary_loss.dim() == 0:
                    secondary_loss = secondary_loss.unsqueeze(0)
                secondary_lengths = secondary_lengths.to(primary_loss.device)
                secondary_norm = secondary_lengths.to(primary_loss.dtype).clamp_min(1)
                secondary_loss = secondary_loss / secondary_norm
            else:
                secondary_loss = torch.zeros_like(primary_loss)

            total_loss = primary_loss * lam_primary + secondary_loss * lam_secondary

        return total_loss.mean()

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

        mix_metadata = None
        if self.mixspeech_enabled and self.model.training:
            mel_input, mix_metadata = self._apply_mixspeech(mel_input)

        target_model = self._get_target_model()
        downsample_factor = 2 ** getattr(target_model, 'n_down', 1)
        mel_input_length = mel_input_length // downsample_factor
        future_mask = self._maybe_create_future_mask(
            mel_input.size(2) // downsample_factor
        )
        mel_mask = target_model.length_to_mask(mel_input_length)
        text_mask = self._maybe_create_text_mask(text_input_length)

        autocast_dtype = self.mixed_precision_dtype if self.autocast_enabled else None
        grad_scaler = self.grad_scaler if self.grad_scaler is not None else None

        if self.autocast_enabled:
            autocast_cm = amp.autocast(
                device_type=self.autocast_device_type,
                dtype=autocast_dtype,
                enabled=True,
            )
        else:
            autocast_cm = nullcontext()

        with autocast_cm:
            model_outputs = self.model(
                mel_input, src_key_padding_mask=mel_mask, text_input=text_input)

            losses = {}
            if mix_metadata:
                losses['mixspeech/avg_lambda'] = mix_metadata['avg_lambda']
                active_ratio = (mix_metadata['secondary_indices'] >= 0).float().mean().item()
                losses['mixspeech/applicable_ratio'] = active_ratio
            total_loss = torch.zeros(1, device=self.device)

            ppgs = model_outputs.get('ctc_logits')
            if self.ctc_weight > 0 and ppgs is not None:
                loss_ctc = self._compute_ctc_loss(ppgs, text_input, mel_input_length, text_input_length, mix_metadata)
                total_loss = total_loss + self.ctc_weight * loss_ctc
                losses['ctc'] = loss_ctc.item()
            else:
                loss_ctc = torch.zeros(1, device=self.device)

            reg_losses, reg_stats = self._compute_ctc_alignment_regularization(
                ppgs, mel_input_length, text_input_length
            )
            for key, value in reg_losses.items():
                total_loss = total_loss + value
                losses[key] = value.item() if torch.is_tensor(value) else float(value)
            for key, value in reg_stats.items():
                losses[key] = float(value)

            s2s_pred = model_outputs.get('s2s_logits')
            s2s_attn = model_outputs.get('s2s_attn')
            if self.s2s_weight > 0 and s2s_pred is not None:
                loss_s2s = torch.zeros(1, device=self.device)
                total_samples = text_input.size(0)
                lam_primary = mix_metadata['lam_primary'] if mix_metadata else None
                lam_secondary = mix_metadata['lam_secondary'] if mix_metadata else None
                secondary_indices = mix_metadata['secondary_indices'] if mix_metadata else None
                fallback_indices = torch.arange(total_samples, device=self.device, dtype=torch.long)

                for idx, (_s2s_pred, _text_input, _text_length) in enumerate(zip(s2s_pred, text_input, text_input_length)):
                    main_len = int(_text_length.item())
                    main_len = min(main_len, _s2s_pred.size(0))
                    if main_len <= 0:
                        continue
                    ce_loss = self.criterion['ce'](_s2s_pred[:main_len], _text_input[:main_len])
                    weight_primary = lam_primary[idx] if lam_primary is not None else 1.0
                    sample_loss = ce_loss * weight_primary

                    if lam_secondary is not None and lam_secondary[idx] > 0:
                        partner_idx = secondary_indices[idx] if secondary_indices[idx] >= 0 else fallback_indices[idx]
                        partner_idx = int(partner_idx.item())
                        partner_len = int(text_input_length[partner_idx].item())
                        partner_len = min(partner_len, _s2s_pred.size(0))
                        if partner_len > 0:
                            partner_target = text_input[partner_idx, :partner_len]
                            ce_partner = self.criterion['ce'](_s2s_pred[:partner_len], partner_target)
                            sample_loss = sample_loss + ce_partner * lam_secondary[idx]

                    loss_s2s = loss_s2s + sample_loss

                loss_s2s = loss_s2s / max(1, text_input.size(0))
                total_loss = total_loss + self.s2s_weight * loss_s2s
                losses['s2s'] = loss_s2s.item()
            else:
                loss_s2s = torch.zeros(1, device=self.device)

            if self.entropy_regularization_enabled:
                if ppgs is not None:
                    entropy_ctc = self._compute_entropy_regularization(ppgs, lengths=mel_input_length, key='ctc')
                    if entropy_ctc is not None:
                        total_loss = total_loss + entropy_ctc
                        losses['entropy/ctc'] = entropy_ctc.item() if torch.is_tensor(entropy_ctc) else float(entropy_ctc)
                if s2s_pred is not None:
                    entropy_s2s = self._compute_entropy_regularization(s2s_pred, lengths=text_input_length, key='s2s')
                    if entropy_s2s is not None:
                        total_loss = total_loss + entropy_s2s
                        losses['entropy/s2s'] = entropy_s2s.item() if torch.is_tensor(entropy_s2s) else float(entropy_s2s)

            intermediate_outputs = model_outputs.get('intermediate_ctc_logits') or {}
            if (
                self.intermediate_ctc_enabled
                and self.intermediate_ctc_weight > 0.0
                and isinstance(intermediate_outputs, dict)
                and intermediate_outputs
            ):
                layer_losses = torch.zeros(1, device=self.device)
                for layer_key, logits in intermediate_outputs.items():
                    if not isinstance(logits, torch.Tensor):
                        continue
                    layer_loss = self._compute_ctc_loss(logits, text_input, mel_input_length, text_input_length, mix_metadata)
                    weight = float(self.intermediate_ctc_layer_weights.get(str(layer_key), 1.0))
                    layer_losses = layer_losses + weight * layer_loss
                    losses[f'intermediate_ctc/layer_{layer_key}'] = layer_loss.item()

                total_loss = total_loss + self.intermediate_ctc_weight * layer_losses
                losses['intermediate_ctc'] = layer_losses.item()

            self_conditioned_outputs = model_outputs.get('self_conditioned_ctc_logits') or {}
            if (
                self.self_conditioned_ctc_enabled
                and self.self_conditioned_ctc_weight > 0.0
                and isinstance(self_conditioned_outputs, dict)
                and self_conditioned_outputs
            ):
                sc_losses = torch.zeros(1, device=self.device)
                for layer_key, logits in self_conditioned_outputs.items():
                    if not isinstance(logits, torch.Tensor):
                        continue
                    sc_loss = self._compute_ctc_loss(logits, text_input, mel_input_length, text_input_length, mix_metadata)
                    weight = float(self.self_conditioned_ctc_layer_weights.get(str(layer_key), 1.0))
                    sc_losses = sc_losses + weight * sc_loss
                    losses[f'self_conditioned_ctc/layer_{layer_key}'] = sc_loss.item()

                total_loss = total_loss + self.self_conditioned_ctc_weight * sc_losses
                losses['self_conditioned_ctc'] = sc_losses.item()

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
                diag_weight = self._get_active_diagonal_attention_weight()
                if diag_weight > 0.0:
                    loss_diagonal = diagonal_attention_prior(
                        s2s_attn,
                        text_input_length,
                        mel_input_length,
                        sigma=self.diagonal_attention_prior_sigma,
                    )
                    total_loss = total_loss + diag_weight * loss_diagonal
                    losses['diag_attn'] = loss_diagonal.item()
                    losses['diag_attn_weight'] = float(diag_weight)

            loss = total_loss

        optimizer_step_ran = False
        if grad_scaler is not None:
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 5)
            grad_scaler.step(self.optimizer)

            found_inf_tensors = getattr(grad_scaler, '_found_inf_per_device', None)
            if isinstance(found_inf_tensors, dict):
                overflow_detected = any(
                    hasattr(found_inf, 'item') and found_inf.item() != 0
                    for found_inf in found_inf_tensors.values()
                )
                optimizer_step_ran = not overflow_detected
            else:
                optimizer_step_ran = True

            grad_scaler.update()
        else:
            if self.accelerator is not None:
                self.accelerator.backward(loss)
            else:
                loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 5)
            self.optimizer.step()
            optimizer_step_ran = True

        if optimizer_step_ran:
            self._optimizer_step_count += 1
            recorded_steps = getattr(self.optimizer, "_step_count", 0)
            if recorded_steps < self._optimizer_step_count:
                setattr(self.optimizer, "_step_count", self._optimizer_step_count)

        if (
            self.scheduler is not None
            and optimizer_step_ran
        ):
            self.scheduler.step()
        losses['loss'] = loss.item()
        if 'ctc' not in losses:
            losses['ctc'] = loss_ctc.item()
        if 's2s' not in losses:
            losses['s2s'] = loss_s2s.item()
        reduced_losses = {}
        for key, value in losses.items():
            reduced_losses[key] = self._reduce_scalar(value)
        return reduced_losses

    def _train_epoch(self):
        if self._sortagrad_active:
            should_switch_now = False
            if self.switch_sortagrad_dataset_epoch is None:
                should_switch_now = True
            elif self.switch_sortagrad_dataset_epoch <= 0 and self.epochs == 0:
                should_switch_now = True
            elif self.epochs == self.switch_sortagrad_dataset_epoch:
                should_switch_now = True

            if should_switch_now:
                self._switch_to_shuffled(None)

        train_losses = defaultdict(list)
        self.model.train()
        self._set_active_diagonal_attention_weight(training=True)
        if self.accelerator is not None and not self.accelerator.is_local_main_process:
            data_iterator = self.train_dataloader
        else:
            data_iterator = tqdm(self.train_dataloader, desc="[train]")

        for train_steps_per_epoch, batch in enumerate(data_iterator, 1):
            losses = self.run(batch)
            for key, value in losses.items():
                train_losses["train/%s" % key].append(value)

        train_losses = {key: np.mean(value) for key, value in train_losses.items()}
        train_losses['train/learning_rate'] = self._get_lr()
        self.epochs += 1

        if torch.cuda.is_available():
            gpu_id = torch.cuda.current_device()
            allocated = torch.cuda.memory_allocated(gpu_id) / 1024**2
        else:
            allocated = 0.0
        if allocated > self.maxm_mem_usage:
            self.maxm_mem_usage = allocated

        train_losses['gpu/max_allocation_recorded'] = self.maxm_mem_usage
        train_losses['gpu/current_allocation'] = allocated

        for key in list(train_losses.keys()):
            if key.startswith('gpu/'):
                continue
            train_losses[key] = self._reduce_scalar(train_losses[key])
        return train_losses

    @torch.no_grad()
    def _eval_epoch(self):
        self.model.eval()
        self._set_active_diagonal_attention_weight(training=False)
        eval_losses = defaultdict(list)
        eval_images = defaultdict(list)
        if self.accelerator is not None and not self.accelerator.is_local_main_process:
            data_iterator = self.val_dataloader
        else:
            data_iterator = tqdm(self.val_dataloader, desc="[eval]")

        for eval_steps_per_epoch, batch in enumerate(data_iterator, 1):
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
            target_model = self._get_target_model()
            downsample_factor = 2 ** getattr(target_model, 'n_down', 1)
            mel_input_length = mel_input_length // downsample_factor
            future_mask = self._maybe_create_future_mask(
                mel_input.size(2) // downsample_factor
            )
            mel_mask = target_model.length_to_mask(mel_input_length)
            text_mask = self._maybe_create_text_mask(text_input_length)
            model_outputs = self.model(
                mel_input, src_key_padding_mask=mel_mask, text_input=text_input)

            ppgs = model_outputs.get('ctc_logits')
            s2s_pred = model_outputs.get('s2s_logits')
            s2s_attn = model_outputs.get('s2s_attn')

            if self.ctc_weight > 0 and ppgs is not None:
                loss_ctc = self._compute_ctc_loss(
                    ppgs,
                    text_input,
                    mel_input_length,
                    text_input_length,
                    mix_metadata=None,
                )
                eval_losses["eval/ctc"].append(self._reduce_scalar(loss_ctc.item()))
            else:
                loss_ctc = torch.zeros(1, device=self.device)

            if self.s2s_weight > 0 and s2s_pred is not None:
                loss_s2s = 0
                for _s2s_pred, _text_input, _text_length in zip(s2s_pred, text_input, text_input_length):
                    loss_s2s += self.criterion['ce'](_s2s_pred[:_text_length], _text_input[:_text_length])
                loss_s2s /= max(1, text_input.size(0))
                eval_losses["eval/s2s"].append(self._reduce_scalar(loss_s2s.item()))
            else:
                loss_s2s = torch.zeros(1, device=self.device)

            total_eval_loss = torch.zeros(1, device=self.device)
            if self.ctc_weight > 0:
                total_eval_loss = total_eval_loss + self.ctc_weight * loss_ctc
            if self.s2s_weight > 0:
                total_eval_loss = total_eval_loss + self.s2s_weight * loss_s2s

            if self.entropy_regularization_enabled:
                if ppgs is not None:
                    entropy_ctc = self._compute_entropy_regularization(ppgs, lengths=mel_input_length, key='ctc')
                    if entropy_ctc is not None:
                        total_eval_loss = total_eval_loss + entropy_ctc
                        eval_losses['eval/entropy_ctc'].append(self._reduce_scalar(entropy_ctc.item()))
                if s2s_pred is not None:
                    entropy_s2s = self._compute_entropy_regularization(s2s_pred, lengths=text_input_length, key='s2s')
                    if entropy_s2s is not None:
                        total_eval_loss = total_eval_loss + entropy_s2s
                        eval_losses['eval/entropy_s2s'].append(self._reduce_scalar(entropy_s2s.item()))

            intermediate_outputs = model_outputs.get('intermediate_ctc_logits') or {}
            if (
                self.intermediate_ctc_enabled
                and self.intermediate_ctc_weight > 0.0
                and isinstance(intermediate_outputs, dict)
                and intermediate_outputs
            ):
                layer_losses = torch.zeros(1, device=self.device)
                for layer_key, logits in intermediate_outputs.items():
                    if not isinstance(logits, torch.Tensor):
                        continue
                    layer_loss = self._compute_ctc_loss(
                        logits,
                        text_input,
                        mel_input_length,
                        text_input_length,
                        mix_metadata=None,
                    )
                    weight = float(self.intermediate_ctc_layer_weights.get(str(layer_key), 1.0))
                    layer_losses = layer_losses + weight * layer_loss
                    eval_losses[f'eval/intermediate_ctc/layer_{layer_key}'].append(self._reduce_scalar(layer_loss.item()))

                total_eval_loss = total_eval_loss + self.intermediate_ctc_weight * layer_losses
                eval_losses['eval/intermediate_ctc'].append(self._reduce_scalar(layer_losses.item()))

            self_conditioned_outputs = model_outputs.get('self_conditioned_ctc_logits') or {}
            if (
                self.self_conditioned_ctc_enabled
                and self.self_conditioned_ctc_weight > 0.0
                and isinstance(self_conditioned_outputs, dict)
                and self_conditioned_outputs
            ):
                sc_losses = torch.zeros(1, device=self.device)
                for layer_key, logits in self_conditioned_outputs.items():
                    if not isinstance(logits, torch.Tensor):
                        continue
                    sc_loss = self._compute_ctc_loss(
                        logits,
                        text_input,
                        mel_input_length,
                        text_input_length,
                        mix_metadata=None,
                    )
                    weight = float(self.self_conditioned_ctc_layer_weights.get(str(layer_key), 1.0))
                    sc_losses = sc_losses + weight * sc_loss
                    eval_losses[f'eval/self_conditioned_ctc/layer_{layer_key}'].append(self._reduce_scalar(sc_loss.item()))

                total_eval_loss = total_eval_loss + self.self_conditioned_ctc_weight * sc_losses
                eval_losses['eval/self_conditioned_ctc'].append(self._reduce_scalar(sc_losses.item()))

            if self.enable_frame_classifier and self.frame_weight > 0:
                frame_logits = model_outputs.get('frame_phoneme_logits')
                if frame_logits is not None:
                    frame_targets = self._build_frame_targets(text_input, text_input_length, mel_input_length, frame_logits.size(1))
                    frame_loss = self.criterion['frame_ce'](frame_logits.reshape(-1, frame_logits.size(2)),
                                                            frame_targets.view(-1))
                    total_eval_loss = total_eval_loss + self.frame_weight * frame_loss
                    eval_losses['eval/frame_phoneme'].append(self._reduce_scalar(frame_loss.item()))

            if self.enable_speaker and self.speaker_weight > 0:
                speaker_logits = model_outputs.get('speaker_logits')
                if speaker_logits is not None:
                    loss_speaker = self.criterion['speaker_ce'](speaker_logits, speaker_ids)
                    total_eval_loss = total_eval_loss + self.speaker_weight * loss_speaker
                    eval_losses['eval/speaker'].append(self._reduce_scalar(loss_speaker.item()))
                    speaker_pred = torch.argmax(speaker_logits, dim=1)
                    speaker_acc = torch.eq(speaker_pred, speaker_ids).float().mean().item()
                    eval_losses['eval/speaker_acc'].append(self._reduce_scalar(speaker_acc))

            if self.enable_pronunciation_error and self.pron_error_weight > 0:
                pron_logits = model_outputs.get('pron_error_logits')
                if pron_logits is not None:
                    pron_targets = self._build_pronunciation_targets(text_input, text_input_length, pron_logits.size(1))
                    pron_loss = self.criterion['pron_error_ce'](pron_logits.reshape(-1, pron_logits.size(2)),
                                                                pron_targets.view(-1))
                    total_eval_loss = total_eval_loss + self.pron_error_weight * pron_loss
                    eval_losses['eval/pronunciation_error'].append(self._reduce_scalar(pron_loss.item()))
                    pron_pred = torch.argmax(pron_logits, dim=2)
                    pron_acc = self._sequence_accuracy(pron_pred, pron_targets)
                    eval_losses['eval/pronunciation_error_acc'].append(self._reduce_scalar(pron_acc))

            eval_losses["eval/loss"].append(self._reduce_scalar(total_eval_loss.item()))

            if ppgs is not None:
                _, amax_ppgs = torch.max(ppgs, dim=2)
                wers = []
                len_diffs = []
                len_diff_norms = []
                ignore_indexes = list(range(5))
                for target, pred, text_length, mel_length in zip(
                    text_input.cpu(),
                    amax_ppgs.cpu(),
                    text_input_length.cpu(),
                    mel_input_length.cpu(),
                ):
                    target_seq = target[:text_length]
                    pred_seq = pred[:mel_length]
                    wers.append(
                        calc_wer(
                            target_seq,
                            pred_seq,
                            ignore_indexes=ignore_indexes,
                        )
                    )
                    collapsed_target = self._collapse_tokens(target_seq.tolist(), ignore_indexes)
                    collapsed_pred = self._collapse_tokens(pred_seq.tolist(), ignore_indexes)
                    diff = float(len(collapsed_pred) - len(collapsed_target))
                    len_diffs.append(diff)
                    ref_len = max(1.0, float(len(collapsed_target)))
                    len_diff_norms.append(diff / ref_len)

                wers = self._gather_metric_list(wers)
                len_diffs = self._gather_metric_list(len_diffs)
                len_diff_norms = self._gather_metric_list(len_diff_norms)
                eval_losses["eval/wer"].extend(wers)
                eval_losses["eval/ctc_len_diff"].extend(len_diffs)
                eval_losses["eval/ctc_len_diff_norm"].extend(len_diff_norms)

            if s2s_pred is not None:
                _, amax_s2s = torch.max(s2s_pred, dim=2)
                acc = [torch.eq(target[:length], pred[:length]).float().mean().item() \
                       for target, pred, length in zip(text_input.cpu(), amax_s2s.cpu(), text_input_length.cpu())]
                acc = self._gather_metric_list(acc)
                eval_losses["eval/acc"].extend(acc)

            if s2s_attn is not None:
                diag_scores = diagonal_attention_coherence(
                    s2s_attn,
                    text_input_length,
                    mel_input_length,
                    sigma=self.diagonal_attention_prior_sigma,
                )
                if isinstance(diag_scores, torch.Tensor):
                    diag_scores = diag_scores.detach().cpu().tolist()
                diag_scores = self._gather_metric_list(diag_scores)
                eval_losses["eval/diag_coherence"].extend(diag_scores)

                if eval_steps_per_epoch <= 2:
                    if self.accelerator is None or self.accelerator.is_main_process:
                        eval_images["eval/image"].append(
                            self.get_image([s2s_attn[0].cpu().numpy()]))

        eval_losses = {key: np.mean(value) for key, value in eval_losses.items()}
        eval_losses.setdefault('eval/diag_weight', self._get_active_diagonal_attention_weight())
        eval_losses.update(eval_images)
        for key in list(eval_losses.keys()):
            if key.startswith('eval/image'):
                continue
            eval_losses[key] = self._reduce_scalar(eval_losses[key])
        return eval_losses
