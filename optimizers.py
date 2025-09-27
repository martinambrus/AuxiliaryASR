#coding:utf-8
import os, sys
import os.path as osp
import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from functools import reduce
from torch.optim import AdamW

try:
    from typing import Iterable, List, Sequence, Tuple, Union, Dict, Any
except ImportError:  # pragma: no cover - typing is available in stdlib, defensive only
    Iterable = List = Sequence = Tuple = Union = Dict = Any = None

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
            self.schedulers[key].step(*args)
        else:
            _ = [self.schedulers[key].step(*args) for key in self.keys]


def _as_parameter_list(params):
    if isinstance(params, (list, tuple)):
        return list(params)
    if isinstance(params, nn.Parameter):
        return [params]
    return list(params)


def _materialize_param_groups(param_spec):
    if isinstance(param_spec, dict):
        param_spec = [param_spec]
    elif not isinstance(param_spec, (list, tuple)):
        param_spec = list(param_spec)

    materialized = []
    for group in param_spec:
        if isinstance(group, dict):
            new_group = dict(group)
            new_group['params'] = _as_parameter_list(new_group.get('params', []))
            materialized.append(new_group)
        else:
            materialized.append(group)
    return materialized


def _clone_param_groups(materialized):
    cloned = []
    for group in materialized:
        if isinstance(group, dict):
            cloned_group = dict(group)
            cloned_group['params'] = list(cloned_group.get('params', []))
            cloned.append(cloned_group)
        else:
            cloned.append(group)
    return cloned


def _resolve_backend_config(fused_cfg, backend, global_default):
    entry = fused_cfg.get(backend)
    if isinstance(entry, dict):
        enabled = entry.get('enabled', global_default)
        extra = {k: v for k, v in entry.items() if k != 'enabled'}
        return bool(enabled), extra
    if entry is None:
        return bool(global_default), {}
    return bool(entry), {}


def _attempt_torch_fused_adamw(param_groups, *, lr, weight_decay, betas, eps):
    kwargs = dict(lr=lr, weight_decay=weight_decay, betas=betas, eps=eps, fused=True)
    try:
        return AdamW(param_groups, **kwargs)
    except TypeError as exc:
        print(f"Torch fused AdamW is unavailable: {exc}")
    except RuntimeError as exc:
        print(f"Failed to initialise torch fused AdamW: {exc}")
    return None


def _attempt_torch_foreach_adamw(param_groups, *, lr, weight_decay, betas, eps):
    kwargs = dict(lr=lr, weight_decay=weight_decay, betas=betas, eps=eps, foreach=True)
    try:
        return AdamW(param_groups, **kwargs)
    except TypeError as exc:
        print(f"AdamW foreach implementation is unavailable: {exc}")
    except RuntimeError as exc:
        print(f"Failed to initialise AdamW with foreach=True: {exc}")
    return None


def _attempt_apex_fused_adam(param_groups, *, lr, weight_decay, betas, eps, extra_kwargs=None):
    try:
        from apex.optimizers import FusedAdam  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"Apex FusedAdam could not be imported: {exc}")
        return None

    fused_kwargs = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, adam_w_mode=True)
    if isinstance(extra_kwargs, dict):
        fused_kwargs.update(extra_kwargs)

    try:
        return FusedAdam(param_groups, **fused_kwargs)
    except Exception as exc:
        print(f"Failed to initialise Apex FusedAdam: {exc}")
    return None


def _select_optimizer_backend(materialized_groups, optimizer_params):
    fused_cfg = optimizer_params.get('fused_optimizers', {}) if isinstance(optimizer_params, dict) else {}
    fused_enabled = bool(fused_cfg.get('enabled', False))

    priority = fused_cfg.get('priority', []) if isinstance(fused_cfg, dict) else []
    if isinstance(priority, str):
        priority = [priority]

    if not priority:
        priority = [backend for backend in ('torch', 'apex', 'foreach')
                    if bool(_resolve_backend_config(fused_cfg, backend, fused_enabled)[0])]

    lr = optimizer_params.get('lr', 1e-4)
    weight_decay = optimizer_params.get('weight_decay', 5e-4)
    betas = optimizer_params.get('betas', (0.9, 0.98))
    if isinstance(betas, list):
        betas = tuple(betas)
    eps = optimizer_params.get('eps', 1e-9)

    for backend in priority:
        backend = str(backend).lower()
        enabled, backend_cfg = _resolve_backend_config(fused_cfg, backend, fused_enabled)
        if not enabled:
            continue

        param_groups = _clone_param_groups(materialized_groups)

        if backend == 'torch':
            optimizer = _attempt_torch_fused_adamw(param_groups, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
        elif backend == 'apex':
            optimizer = _attempt_apex_fused_adam(param_groups, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps, extra_kwargs=backend_cfg)
        elif backend == 'foreach':
            optimizer = _attempt_torch_foreach_adamw(param_groups, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
        else:
            print(f"Unknown fused optimizer backend '{backend}' requested; skipping.")
            optimizer = None

        if optimizer is not None:
            print(f"Using fused optimizer backend: {backend}")
            return optimizer

    return None


def build_optimizer(parameters):
    optimizer, scheduler = _define_optimizer(parameters)
    return optimizer, scheduler

def _define_optimizer(params):
    optimizer_params = params.get('optimizer_params', {}) or {}
    sch_params = params['scheduler_params']
    materialized_groups = _materialize_param_groups(params['params'])

    optimizer = _select_optimizer_backend(materialized_groups, optimizer_params)
    if optimizer is None:
        lr = optimizer_params.get('lr', 1e-4)
        weight_decay = optimizer_params.get('weight_decay', 5e-4)
        betas = optimizer_params.get('betas', (0.9, 0.98))
        if isinstance(betas, list):
            betas = tuple(betas)
        eps = optimizer_params.get('eps', 1e-9)
        optimizer = AdamW(
            _clone_param_groups(materialized_groups),
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps)
        print("Using standard AdamW optimizer")

    scheduler = _define_scheduler(optimizer, sch_params)
    return optimizer, scheduler

def _define_scheduler(optimizer, params):
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=params.get('max_lr', 5e-4),
        epochs=params.get('epochs', 200),
        steps_per_epoch=params.get('steps_per_epoch', 1000),
        pct_start=params.get('pct_start', 0.0),
        final_div_factor=5)

    return scheduler

def build_multi_optimizer(parameters_dict, scheduler_params):
    optim = dict([(key, AdamW(params, lr=1e-4, weight_decay=1e-6, betas=(0.9, 0.98), eps=1e-9))
                   for key, params in parameters_dict.items()])

    schedulers = dict([(key, _define_scheduler(opt, scheduler_params)) \
                       for key, opt in optim.items()])

    multi_optim = MultiOptimizer(optim, schedulers)
    return multi_optim
