#coding:utf-8
import math
from functools import reduce

import torch
from torch.optim import AdamW

class MultiOptimizer:
    def __init__(self, optimizers={}, schedulers={}):
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.keys = list(optimizers.keys())
        self.param_groups = reduce(lambda x,y: x+y, [v.param_groups for v in self.optimizers.values()])

    def state_dict(self):
        state_dicts = [(key, self.optimizers[key].state_dict())\
                       for key in self.keys]
        return state_dicts

    def load_state_dict(self, state_dict):
        for key, val in state_dict:
            try:
                self.optimizers[key].load_state_dict(val)
            except:
                print("Unloaded %s" % key)


    def step(self, key=None):
        if key is not None:
            self.optimizers[key].step()
        else:
            _ = [self.optimizers[key].step() for key in self.keys]

    def zero_grad(self, key=None):
        if key is not None:
            self.optimizers[key].zero_grad()
        else:
            _ = [self.optimizers[key].zero_grad() for key in self.keys]

    def scheduler(self, *args, key=None):
        if key is not None:
            scheduler = self.schedulers.get(key)
            if scheduler is not None:
                scheduler.step(*args)
        else:
            for scheduler in self.schedulers.values():
                if scheduler is not None:
                    scheduler.step(*args)


class CosineWarmupRestarts(torch.optim.lr_scheduler._LRScheduler):
    """Cosine annealing with warm restarts and optional linear warm-up."""

    def __init__(
        self,
        optimizer,
        T_0,
        T_mult=1,
        eta_min=0.0,
        warmup_steps=0,
        warmup_start_lr=0.0,
        last_epoch=-1,
    ):
        self.T_0 = max(1, int(T_0))
        self.T_mult = max(1, int(T_mult))
        self.eta_min = float(eta_min)
        self.warmup_steps = max(0, int(warmup_steps))
        self.warmup_start_lr = float(warmup_start_lr)
        super().__init__(optimizer, last_epoch)

    def _current_cycle_length(self, step_in_cycle):
        cycle_length = self.T_0
        if self.T_mult == 1:
            if cycle_length <= 0:
                cycle_length = 1
            return cycle_length, step_in_cycle % cycle_length

        remaining = step_in_cycle
        cycle_length = max(1, self.T_0)
        while remaining >= cycle_length:
            remaining -= cycle_length
            cycle_length = max(1, int(cycle_length * self.T_mult))
        return cycle_length, remaining

    def get_lr(self):
        step = self.last_epoch
        if self.warmup_steps > 0 and step < self.warmup_steps:
            progress = (step + 1) / float(self.warmup_steps)
            return [
                self.warmup_start_lr + (base_lr - self.warmup_start_lr) * progress
                for base_lr in self.base_lrs
            ]

        step_in_cycle = max(0, step - self.warmup_steps)
        cycle_length, cycle_progress = self._current_cycle_length(step_in_cycle)
        cycle_length = max(1, cycle_length)
        cosine_value = math.cos(math.pi * cycle_progress / float(cycle_length))
        scale = (1.0 + cosine_value) / 2.0
        return [
            self.eta_min + (base_lr - self.eta_min) * scale
            for base_lr in self.base_lrs
        ]


def build_optimizer(parameters, runtime_params=None):
    optimizer, scheduler = _define_optimizer(parameters, runtime_params or {})
    return optimizer, scheduler

def _define_optimizer(params, runtime_params):
    optimizer_params = params.get('optimizer_params', {}) or {}
    betas = optimizer_params.get('betas', (0.9, 0.98))
    if isinstance(betas, (list, tuple)) and len(betas) == 2:
        betas = tuple(betas)
    else:
        betas = (0.9, 0.98)
    optimizer = AdamW(
        params['params'],
        lr=optimizer_params.get('lr', 1e-4),
        weight_decay=optimizer_params.get('weight_decay', 5e-4),
        betas=betas,
        eps=optimizer_params.get('eps', 1e-9))
    scheduler = _define_scheduler(optimizer, optimizer_params, runtime_params)
    return optimizer, scheduler

def _define_scheduler(optimizer, optimizer_params, runtime_params):
    scheduler_config = optimizer_params.get('scheduler')
    legacy_pct_start = optimizer_params.get('pct_start')
    legacy_final_div = optimizer_params.get('final_div_factor', 5.0)
    if scheduler_config is None and legacy_pct_start is not None:
        scheduler_config = {
            'enabled': True,
            'type': 'one_cycle',
            'one_cycle': {
                'pct_start': legacy_pct_start,
                'final_div_factor': legacy_final_div,
            },
        }

    if not scheduler_config:
        return None

    if not scheduler_config.get('enabled', True):
        return None

    scheduler_type = scheduler_config.get('type', 'one_cycle')
    max_lr = optimizer_params.get('max_lr', optimizer_params.get('lr', 5e-4))

    if scheduler_type == 'one_cycle':
        one_cycle_cfg = scheduler_config.get('one_cycle', {}) or {}
        pct_start = float(one_cycle_cfg.get('pct_start', legacy_pct_start if legacy_pct_start is not None else 0.1))
        final_div_factor = float(one_cycle_cfg.get('final_div_factor', legacy_final_div))
        total_steps = one_cycle_cfg.get('total_steps', runtime_params.get('total_steps'))
        if total_steps is not None:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=max_lr,
                total_steps=int(total_steps),
                pct_start=pct_start,
                final_div_factor=final_div_factor,
            )
        else:
            epochs = one_cycle_cfg.get('epochs', runtime_params.get('epochs', 200))
            steps_per_epoch = one_cycle_cfg.get('steps_per_epoch', runtime_params.get('steps_per_epoch', 1000))
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=max_lr,
                epochs=int(epochs),
                steps_per_epoch=int(steps_per_epoch),
                pct_start=pct_start,
                final_div_factor=final_div_factor,
            )
        scheduler.total_steps = total_steps if total_steps is not None else int(runtime_params.get('total_steps', 0) or 0)
        scheduler.scheduler_type = 'one_cycle'
        scheduler.pct_start = pct_start
        scheduler.final_div_factor = final_div_factor
        return scheduler

    if scheduler_type == 'cosine_warm_restarts':
        cosine_cfg = scheduler_config.get('cosine_warm_restarts', {}) or {}
        steps_per_epoch = runtime_params.get('steps_per_epoch')
        T_0_steps = cosine_cfg.get('T_0_steps')
        if T_0_steps is None:
            T_0_epochs = cosine_cfg.get('T_0_epochs', 1)
            if steps_per_epoch is None:
                raise ValueError(
                    "`steps_per_epoch` must be provided in runtime_params to derive `T_0_steps` for the cosine warm restart scheduler."
                )
            T_0_steps = max(1, int(round(float(T_0_epochs) * steps_per_epoch)))
        else:
            T_0_steps = max(1, int(T_0_steps))

        T_mult = int(cosine_cfg.get('T_mult', 1))
        eta_min = float(cosine_cfg.get('eta_min', 0.0))
        warmup_pct = cosine_cfg.get('warmup_pct', cosine_cfg.get('pct_start', legacy_pct_start if legacy_pct_start is not None else 0.0))
        warmup_steps = cosine_cfg.get('warmup_steps')
        if warmup_steps is None:
            warmup_steps = int(round(T_0_steps * float(warmup_pct or 0.0)))
        else:
            warmup_steps = int(warmup_steps)
        warmup_start_lr = float(cosine_cfg.get('warmup_start_lr', eta_min))

        scheduler = CosineWarmupRestarts(
            optimizer,
            T_0=T_0_steps,
            T_mult=T_mult,
            eta_min=eta_min,
            warmup_steps=warmup_steps,
            warmup_start_lr=warmup_start_lr,
        )
        scheduler.total_steps = runtime_params.get('total_steps')
        scheduler.scheduler_type = 'cosine_warm_restarts'
        scheduler.max_lr = max_lr
        scheduler.T_0 = T_0_steps
        scheduler.T_mult = T_mult
        scheduler.eta_min = eta_min
        scheduler.warmup_steps = warmup_steps
        scheduler.warmup_start_lr = warmup_start_lr
        scheduler.warmup_pct = warmup_pct
        return scheduler

    raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

def build_multi_optimizer(parameters_dict, optimizer_params, runtime_params=None):
    runtime_params = runtime_params or {}
    optim = dict([
        (
            key,
            AdamW(
                params,
                lr=optimizer_params.get('lr', 1e-4),
                weight_decay=optimizer_params.get('weight_decay', 1e-6),
                betas=tuple(optimizer_params.get('betas', (0.9, 0.98))),
                eps=optimizer_params.get('eps', 1e-9),
            ),
        )
        for key, params in parameters_dict.items()
    ])

    schedulers = dict([
        (key, _define_scheduler(opt, optimizer_params, runtime_params))
        for key, opt in optim.items()
    ])

    multi_optim = MultiOptimizer(optim, schedulers)
    return multi_optim
