#coding:utf-8
import os, sys
import os.path as osp
import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from functools import reduce
from torch.optim import AdamW
import importlib
import warnings

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
            for key in self.keys:
                scheduler = self.schedulers.get(key)
                if scheduler is not None:
                    scheduler.step(*args)


def build_optimizer(parameters):
    optimizer, scheduler = _define_optimizer(parameters)
    return optimizer, scheduler

def _define_optimizer(params):
    optimizer_params = params.get('optimizer_params') or {}
    sch_params = params.get('scheduler_params') or {}
    optimizer = _create_optimizer(params['params'], optimizer_params)
    scheduler = _define_scheduler(optimizer, sch_params) if sch_params else None
    return optimizer, scheduler


def _create_optimizer(param_groups, optimizer_params):
    lr = float(optimizer_params.get('lr', 1e-4))
    weight_decay = float(optimizer_params.get('weight_decay', 5e-4))
    betas = optimizer_params.get('betas', (0.9, 0.98))
    if isinstance(betas, (list, tuple)):
        betas = tuple(float(b) for b in betas)
    eps = float(optimizer_params.get('eps', 1e-9))

    fused_config = optimizer_params.get('fused_optimizers', {}) or {}
    implementation_priority = optimizer_params.get('implementation_priority')
    if isinstance(implementation_priority, (str, bytes)):
        implementation_priority = [implementation_priority]
    if not implementation_priority:
        implementation_priority = [
            'torch_fused_adamw',
            'apex_fused_adam',
            'adamw_foreach',
        ]
    allow_unfused_fallback = bool(optimizer_params.get('fallback_to_unfused', True))

    attempted = []
    for implementation in implementation_priority:
        impl_cfg = fused_config.get(implementation, {}) if isinstance(fused_config, dict) else {}
        if impl_cfg and not bool(impl_cfg.get('enabled', False)):
            continue
        if not impl_cfg and implementation in ('adamw_foreach',) and not bool(fused_config.get('enabled', True) if isinstance(fused_config, dict) and 'enabled' in fused_config else True):
            # Support the legacy pattern fused_optimizers: {enabled: bool}
            continue
        try:
            if implementation == 'torch_fused_adamw':
                optimizer = _try_torch_fused_adamw(param_groups, lr, betas, eps, weight_decay, impl_cfg)
            elif implementation == 'apex_fused_adam':
                optimizer = _try_apex_fused_adam(param_groups, lr, betas, eps, weight_decay, impl_cfg)
            elif implementation == 'adamw_foreach':
                optimizer = _try_adamw_foreach(param_groups, lr, betas, eps, weight_decay, impl_cfg)
            else:
                attempted.append((implementation, 'unsupported implementation'))
                continue
        except RuntimeError as runtime_error:
            attempted.append((implementation, str(runtime_error)))
            continue
        except Exception as exc:
            attempted.append((implementation, str(exc)))
            continue

        if optimizer is not None:
            print(f"Using optimizer implementation: {implementation}")
            return optimizer

    if allow_unfused_fallback:
        print("Falling back to standard AdamW optimizer")
        return AdamW(param_groups, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

    attempted_info = ', '.join([f"{name} (error: {err})" for name, err in attempted]) or 'none'
    raise RuntimeError(f"Unable to construct a fused optimizer. Attempts: {attempted_info}")


def _try_torch_fused_adamw(param_groups, lr, betas, eps, weight_decay, impl_cfg):
    if not hasattr(AdamW, '__init__'):
        raise RuntimeError('Torch AdamW constructor not available')
    fused_kwargs = {}
    if isinstance(impl_cfg, dict) and 'use_foreach' in impl_cfg:
        fused_kwargs['foreach'] = bool(impl_cfg.get('use_foreach'))
    try:
        optimizer = AdamW(param_groups, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, fused=True, **fused_kwargs)
    except TypeError as exc:
        raise RuntimeError(str(exc))
    return optimizer


def _try_apex_fused_adam(param_groups, lr, betas, eps, weight_decay, impl_cfg):
    fused_adam_cls = _import_apex_fused_adam()
    if fused_adam_cls is None:
        raise RuntimeError('apex.optimizers.FusedAdam is not available')

    adamw_mode = bool((impl_cfg or {}).get('adamw_mode', True))
    optimizer = fused_adam_cls(
        param_groups,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        adam_w_mode=adamw_mode,
    )
    return optimizer


def _try_adamw_foreach(param_groups, lr, betas, eps, weight_decay, impl_cfg):
    foreach = True if impl_cfg is None else bool(impl_cfg.get('enabled', True))
    if not foreach:
        raise RuntimeError('adamw_foreach implementation explicitly disabled')
    try:
        optimizer = AdamW(param_groups, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, foreach=True)
    except TypeError as exc:
        raise RuntimeError(str(exc))
    return optimizer


def _import_apex_fused_adam():
    try:
        apex_optimizers = importlib.import_module('apex.optimizers')
    except ModuleNotFoundError:
        return None
    except Exception as exc:
        warnings.warn(f"Failed to import apex.optimizers due to: {exc}")
        return None

    return getattr(apex_optimizers, 'FusedAdam', None)

def _define_scheduler(optimizer, params):
    if not params:
        raise ValueError('Scheduler parameters must be provided for OneCycleLR.')
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=params.get('max_lr', 5e-4),
        epochs=params.get('epochs', 200),
        steps_per_epoch=params.get('steps_per_epoch', 1000),
        pct_start=params.get('pct_start', 0.0),
        final_div_factor=5)

    return scheduler


def build_multi_optimizer(parameters_dict, scheduler_params, optimizer_params=None):
    optimizer_params = optimizer_params or {}
    optim = {}
    schedulers = {}

    for key, params in parameters_dict.items():
        optim[key] = _create_optimizer(params, optimizer_params)
        schedulers[key] = _define_scheduler(optim[key], scheduler_params) if scheduler_params else None

    multi_optim = MultiOptimizer(optim, schedulers)
    return multi_optim
