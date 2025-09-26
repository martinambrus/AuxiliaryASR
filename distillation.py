"""Utilities for configuring and computing ensemble distillation losses."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class TargetConfig:
    enabled: bool = True
    weight: float = 1.0
    loss: str = "kl"


@dataclass
class TeacherConfig:
    name: str
    storage_path: str
    file_suffix: str = ".pt"
    file_format: str = "pt"
    path_style: str = "basename"
    file_prefix: str = ""
    include_extension: bool = False
    weight: float = 1.0
    temperature: float = 1.0
    logits_format: str = "logits"
    optional: bool = False
    keys: Dict[str, Optional[str]] = None

    def __post_init__(self):
        self.file_format = str(self.file_format or "pt").lower()
        self.path_style = str(self.path_style or "basename").lower()
        self.logits_format = str(self.logits_format or "logits").lower()
        self.keys = dict(self.keys or {})


@dataclass
class DistillationConfig:
    enabled: bool = False
    temperature: float = 1.0
    loss_weight: float = 1.0
    strict_loading: bool = False
    missing_strategy: str = "skip"
    targets: Dict[str, TargetConfig] = None
    teachers: List[TeacherConfig] = None

    def __post_init__(self):
        self.targets = dict(self.targets or {})
        self.teachers = list(self.teachers or [])
        self.missing_strategy = str(self.missing_strategy or "skip").lower()
        if self.missing_strategy not in {"skip", "error", "warn"}:
            self.missing_strategy = "skip"


_DEFAULT_TARGETS = {
    "ctc": TargetConfig(enabled=True, weight=1.0, loss="kl"),
    "s2s": TargetConfig(enabled=True, weight=1.0, loss="kl"),
    "attention": TargetConfig(enabled=False, weight=1.0, loss="kl"),
}


def prepare_distillation_config(raw_config: Optional[Dict[str, Any]]) -> DistillationConfig:
    """Return a normalised :class:`DistillationConfig` from raw YAML data."""
    if not isinstance(raw_config, dict):
        return DistillationConfig(enabled=False, targets=_DEFAULT_TARGETS)

    enabled = bool(raw_config.get("enabled", False))
    temperature = float(raw_config.get("temperature", 1.0))
    loss_weight = float(raw_config.get("loss_weight", 1.0))
    strict_loading = bool(raw_config.get("strict_loading", False))
    missing_strategy = str(raw_config.get("missing_strategy", "skip")).lower()

    targets_cfg = {}
    raw_targets = raw_config.get("targets") or raw_config.get("apply_to") or {}
    for key, default in _DEFAULT_TARGETS.items():
        raw_target = raw_targets.get(key, {}) if isinstance(raw_targets, dict) else {}
        if isinstance(raw_target, (int, float)):
            raw_target = {"weight": float(raw_target), "enabled": True}
        if isinstance(raw_target, bool):
            raw_target = {"enabled": raw_target}
        target = TargetConfig(
            enabled=bool(raw_target.get("enabled", default.enabled)),
            weight=float(raw_target.get("weight", default.weight)),
            loss=str(raw_target.get("loss", default.loss)).lower(),
        )
        targets_cfg[key] = target

    teachers_cfg: List[TeacherConfig] = []
    raw_teachers = raw_config.get("teachers", [])
    if isinstance(raw_teachers, dict):
        raw_teachers = [raw_teachers]
    for idx, teacher in enumerate(raw_teachers):
        if not isinstance(teacher, dict):
            continue
        storage = teacher.get("storage", {})
        if not isinstance(storage, dict):
            storage = {}
        storage_path = storage.get("path") or teacher.get("path")
        if not storage_path:
            continue
        name = teacher.get("name") or f"teacher_{idx+1}"
        weight = float(teacher.get("weight", 1.0))
        teacher_temp = float(teacher.get("temperature", temperature))
        logits_format = teacher.get("logits_format", teacher.get("format", "logits"))
        optional = bool(teacher.get("optional", False))
        file_suffix = storage.get("file_suffix", storage.get("suffix", ".pt"))
        file_format = storage.get("file_format", storage.get("format", "pt"))
        path_style = storage.get("path_style", storage.get("style", "basename"))
        file_prefix = storage.get("file_prefix", storage.get("prefix", ""))
        include_extension = bool(storage.get("include_extension", False))
        keys = teacher.get("keys", {})
        teachers_cfg.append(TeacherConfig(
            name=name,
            storage_path=storage_path,
            file_suffix=file_suffix,
            file_format=file_format,
            path_style=path_style,
            file_prefix=file_prefix,
            include_extension=include_extension,
            weight=weight,
            temperature=teacher_temp,
            logits_format=logits_format,
            optional=optional,
            keys=keys,
        ))

    return DistillationConfig(
        enabled=enabled,
        temperature=temperature,
        loss_weight=loss_weight,
        strict_loading=strict_loading,
        missing_strategy=missing_strategy,
        targets=targets_cfg,
        teachers=teachers_cfg,
    )


class TeacherLogitsLoader:
    """Load per-sample teacher logits from disk according to a config."""

    def __init__(self, config: DistillationConfig):
        self.config = config or DistillationConfig(enabled=False, targets=_DEFAULT_TARGETS)
        self.enabled = bool(self.config.enabled and self.config.teachers)

    def _resolve_sample_stem(self, audio_path: str, teacher: TeacherConfig) -> str:
        if teacher.path_style == "relative":
            base = os.path.splitext(audio_path)[0]
            base = base.replace("\\", os.sep)
        else:
            base = os.path.splitext(os.path.basename(audio_path))[0]
        if teacher.include_extension:
            base = os.path.basename(audio_path)
        return f"{teacher.file_prefix}{base}{teacher.file_suffix}"

    def _load_payload(self, file_path: str, teacher: TeacherConfig):
        fmt = teacher.file_format
        if fmt == "pt":
            return torch.load(file_path, map_location="cpu")
        if fmt == "pth":
            return torch.load(file_path, map_location="cpu")
        if fmt == "npz":
            return np.load(file_path, allow_pickle=True)
        if fmt == "npy":
            return np.load(file_path, allow_pickle=True)
        raise ValueError(f"Unsupported teacher file format: {fmt}")

    def _extract_value(self, payload, key: Optional[str]):
        if key is None:
            return None
        if isinstance(payload, dict):
            return payload.get(key)
        if isinstance(payload, np.lib.npyio.NpzFile):
            if key in payload.files:
                return payload[key]
            return None
        if hasattr(payload, key):
            return getattr(payload, key)
        if isinstance(payload, (list, tuple)):
            try:
                key_int = int(key)
            except (TypeError, ValueError):
                return None
            if 0 <= key_int < len(payload):
                return payload[key_int]
        return None

    def fetch(self, audio_path: str, mel_frames: int = None, text_tokens: int = None):
        if not self.enabled:
            return None

        aggregated = {"ctc": [], "s2s": [], "attention": []}
        for teacher in self.config.teachers:
            sample_name = self._resolve_sample_stem(audio_path, teacher)
            file_path = os.path.join(teacher.storage_path, sample_name)
            if not os.path.exists(file_path):
                if self.config.strict_loading and not teacher.optional:
                    raise FileNotFoundError(f"Teacher logits missing: {file_path}")
                if self.config.missing_strategy == "warn" and not teacher.optional:
                    print(f"[distillation] Missing teacher logits for {audio_path} -> {file_path}")
                continue
            try:
                payload = self._load_payload(file_path, teacher)
            except Exception as exc:  # pylint: disable=broad-except
                if self.config.strict_loading and not teacher.optional:
                    raise
                print(f"[distillation] Failed to load {file_path}: {exc}")
                continue

            keys = teacher.keys or {}
            entry = {}
            for target_key in ("ctc", "s2s", "attention"):
                key_name = keys.get(target_key)
                if not key_name:
                    continue
                raw_value = self._extract_value(payload, key_name)
                if raw_value is None:
                    continue
                tensor = torch.as_tensor(raw_value).float()
                if tensor.dim() >= 3 and tensor.size(0) == 1:
                    tensor = tensor.squeeze(0)
                length_key = keys.get(f"{target_key}_length")
                target_length = self._extract_value(payload, length_key) if length_key else None
                if isinstance(target_length, (list, tuple, np.ndarray)):
                    if len(target_length) > 0:
                        target_length = int(target_length[0])
                    else:
                        target_length = None
                if torch.is_tensor(target_length):
                    target_length = int(target_length.item())
                entry[target_key] = {
                    "logits": tensor,
                    "length": target_length,
                    "name": teacher.name,
                    "weight": float(teacher.weight),
                    "temperature": float(teacher.temperature),
                    "format": teacher.logits_format,
                }
            for key, value in entry.items():
                aggregated[key].append(value)

        if not any(aggregated[key] for key in aggregated):
            return None
        return aggregated


class EnsembleDistillationHelper:
    """Compute ensemble distillation losses for the trainer."""

    def __init__(self, config: DistillationConfig, device: torch.device):
        self.config = config
        self.device = device
        self.temperature = float(config.temperature)
        self.loss_weight = float(config.loss_weight)

    def is_enabled(self, key: str) -> bool:
        target_cfg = self.config.targets.get(key)
        return bool(self.config.enabled and target_cfg and target_cfg.enabled and target_cfg.weight > 0.0)

    def target_weight(self, key: str) -> float:
        target_cfg = self.config.targets.get(key)
        if target_cfg is None:
            return 0.0
        return float(target_cfg.weight)

    def target_loss_type(self, key: str) -> str:
        target_cfg = self.config.targets.get(key)
        if target_cfg is None:
            return "kl"
        return target_cfg.loss

    def _aggregate_probabilities(
        self,
        teachers: Sequence[Dict[str, Any]],
        reference: torch.Tensor,
        key: str,
    ) -> Optional[torch.Tensor]:
        if not teachers:
            return None

        valid_tensors: List[torch.Tensor] = []
        weights: List[float] = []
        min_time = reference.size(0)
        vocab_size = reference.size(-1)
        for entry in teachers:
            tensor = entry.get("logits")
            if tensor is None:
                continue
            tensor = torch.as_tensor(tensor).float()
            if tensor.dim() == 3 and tensor.size(0) == 1:
                tensor = tensor.squeeze(0)
            if tensor.size(-1) != vocab_size:
                continue
            entry_length = entry.get("length")
            if entry_length is None:
                entry_length = tensor.size(0)
            min_time = min(min_time, int(entry_length), tensor.size(0))
            weights.append(float(entry.get("weight", 1.0)))
            valid_tensors.append(tensor)

        if not valid_tensors or min_time <= 0:
            return None

        stacked = []
        for tensor, weight, entry in zip(valid_tensors, weights, teachers):
            tensor = tensor[:min_time].to(self.device)
            fmt = entry.get("format", "logits")
            teacher_temp = float(entry.get("temperature", self.temperature))
            if fmt == "probs":
                probs = tensor
                probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            else:
                probs = torch.softmax(tensor / max(teacher_temp, 1e-6), dim=-1)
            stacked.append(probs * weight)
        summed = torch.stack(stacked, dim=0).sum(dim=0)
        weight_total = sum(weights)
        if weight_total <= 0:
            weight_total = len(stacked)
        aggregated = summed / max(weight_total, 1e-6)
        aggregated = aggregated / aggregated.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return aggregated

    def _aggregate_raw(self, teachers: Sequence[Dict[str, Any]], reference: torch.Tensor) -> Optional[torch.Tensor]:
        if not teachers:
            return None
        valid_tensors: List[torch.Tensor] = []
        weights: List[float] = []
        min_time = reference.size(0)
        feature_size = reference.size(-1)
        for entry in teachers:
            tensor = entry.get("logits")
            if tensor is None:
                continue
            tensor = torch.as_tensor(tensor).float()
            if tensor.dim() == 3 and tensor.size(0) == 1:
                tensor = tensor.squeeze(0)
            if tensor.size(-1) != feature_size:
                continue
            entry_length = entry.get("length")
            if entry_length is None:
                entry_length = tensor.size(0)
            min_time = min(min_time, int(entry_length), tensor.size(0))
            weights.append(float(entry.get("weight", 1.0)))
            valid_tensors.append(tensor)

        if not valid_tensors or min_time <= 0:
            return None

        stacked = []
        for tensor, weight in zip(valid_tensors, weights):
            stacked.append(tensor[:min_time].to(self.device) * weight)
        summed = torch.stack(stacked, dim=0).sum(dim=0)
        weight_total = sum(weights)
        if weight_total <= 0:
            weight_total = len(stacked)
        aggregated = summed / max(weight_total, 1e-6)
        return aggregated

    def _sequence_mask(self, length: int, max_len: int) -> torch.Tensor:
        return torch.arange(max_len, device=self.device) < max(0, int(length))

    def _kl_loss(
        self,
        student_logits: torch.Tensor,
        teacher_entries: Sequence[Dict[str, Any]],
        length: int,
        key: str,
    ) -> Optional[torch.Tensor]:
        aggregated = self._aggregate_probabilities(teacher_entries, student_logits, key)
        if aggregated is None:
            return None
        max_time = aggregated.size(0)
        student_slice = student_logits[:max_time]
        length = min(int(length), max_time)
        if length <= 0:
            return None
        mask = self._sequence_mask(length, max_time).float()
        student_log_probs = torch.log_softmax(student_slice / max(self.temperature, 1e-6), dim=-1)
        per_step = F.kl_div(student_log_probs, aggregated, reduction="none").sum(dim=-1)
        per_step = per_step * mask
        normalizer = mask.sum().clamp_min(1.0)
        loss = per_step.sum() / normalizer
        loss = loss * (self.temperature ** 2)
        return loss

    def _mse_loss(
        self,
        student_tensor: torch.Tensor,
        teacher_entries: Sequence[Dict[str, Any]],
        length: int,
    ) -> Optional[torch.Tensor]:
        aggregated = self._aggregate_raw(teacher_entries, student_tensor)
        if aggregated is None:
            return None
        max_time = aggregated.size(0)
        student_slice = student_tensor[:max_time]
        length = min(int(length), max_time)
        if length <= 0:
            return None
        mask = self._sequence_mask(length, max_time).float().unsqueeze(-1)
        squared = (student_slice - aggregated) ** 2
        squared = squared * mask
        normalizer = mask.sum().clamp_min(1.0)
        loss = squared.sum() / normalizer
        return loss

    def compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        teacher_batch: Optional[Sequence[Optional[Dict[str, Any]]]],
        mel_lengths: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
        if not self.config.enabled:
            return None, {}
        if teacher_batch is None:
            return None, {}

        batch_size = len(teacher_batch)
        losses: Dict[str, float] = {}
        total_loss = None

        # CTC distillation
        ctc_logits = outputs.get("ctc_logits")
        if self.is_enabled("ctc") and isinstance(ctc_logits, torch.Tensor):
            per_sample_losses = []
            for idx in range(batch_size):
                teachers = (teacher_batch[idx] or {}).get("ctc", [])
                if not teachers:
                    continue
                sample_loss = self._kl_loss(ctc_logits[idx], teachers, mel_lengths[idx], "ctc")
                if sample_loss is not None:
                    per_sample_losses.append(sample_loss)
            if per_sample_losses:
                stacked = torch.stack(per_sample_losses)
                loss_ctc = stacked.mean() * self.target_weight("ctc")
                losses["distillation/ctc"] = float(loss_ctc.item())
                total_loss = loss_ctc if total_loss is None else total_loss + loss_ctc

        # Seq2Seq distillation
        s2s_logits = outputs.get("s2s_logits")
        if self.is_enabled("s2s") and isinstance(s2s_logits, torch.Tensor):
            per_sample_losses = []
            for idx in range(batch_size):
                teachers = (teacher_batch[idx] or {}).get("s2s", [])
                if not teachers:
                    continue
                sample_loss = self._kl_loss(s2s_logits[idx], teachers, text_lengths[idx], "s2s")
                if sample_loss is not None:
                    per_sample_losses.append(sample_loss)
            if per_sample_losses:
                stacked = torch.stack(per_sample_losses)
                loss_s2s = stacked.mean() * self.target_weight("s2s")
                losses["distillation/s2s"] = float(loss_s2s.item())
                total_loss = loss_s2s if total_loss is None else total_loss + loss_s2s

        # Attention distillation
        s2s_attn = outputs.get("s2s_attn")
        if self.is_enabled("attention") and isinstance(s2s_attn, torch.Tensor):
            per_sample_losses = []
            loss_type = self.target_loss_type("attention")
            for idx in range(batch_size):
                teachers = (teacher_batch[idx] or {}).get("attention", [])
                if not teachers:
                    continue
                if loss_type == "mse":
                    sample_loss = self._mse_loss(s2s_attn[idx], teachers, text_lengths[idx])
                else:
                    sample_loss = self._kl_loss(s2s_attn[idx], teachers, text_lengths[idx], "attention")
                if sample_loss is not None:
                    per_sample_losses.append(sample_loss)
            if per_sample_losses:
                stacked = torch.stack(per_sample_losses)
                loss_attn = stacked.mean() * self.target_weight("attention")
                losses["distillation/attention"] = float(loss_attn.item())
                total_loss = loss_attn if total_loss is None else total_loss + loss_attn

        if total_loss is None:
            return None, {}
        losses["distillation/total_unscaled"] = float(total_loss.item())
        total_loss = total_loss * self.loss_weight
        losses["distillation/total"] = float(total_loss.item())
        return total_loss, losses
