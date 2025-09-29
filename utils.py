import math
import os
import os.path as osp
import sys
import time
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import numpy as np
import soundfile as sf
import torch
from torch import nn
import jiwer

import matplotlib.pylab as plt
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from decoding import build_decoder_from_config


def select_logits_from_output(model_output, preferred_order=(
    "primary_logits",
    "ctc_logits",
    "s2s_logits",
    "logits",
)):
    """Return the primary logit tensor from a model forward pass.

    Recent multi-task changes make :class:`~models.ASRCNN` return a dictionary
    containing logits for every enabled objective. The utility notebooks were
    written against the previous behaviour where ``model(mels)`` yielded a
    tensor, so helpers that consume model outputs now need a consistent way to
    retrieve the main ASR logits.  This function inspects the output structure
    and returns the first tensor that matches the preferred key order.  If a
    tensor is passed directly it is returned unchanged.

    Args:
        model_output: The object returned by ``model.forward``.
        preferred_order: Sequence of dictionary keys to probe.  The first
            existing tensor value is returned.

    Returns:
        torch.Tensor: The logits tensor suitable for decoding.

    Raises:
        TypeError: If ``model_output`` is neither a tensor nor a mapping.
        KeyError: If no tensor is found for any of the preferred keys.
    """

    if isinstance(model_output, torch.Tensor):
        return model_output

    if isinstance(model_output, dict):
        for key in preferred_order:
            tensor = model_output.get(key)
            if isinstance(tensor, torch.Tensor):
                return tensor
        raise KeyError(
            "Could not find logits in model output. Available keys: "
            + ", ".join(model_output.keys())
        )

    raise TypeError(
        "Expected model output to be a tensor or dict, got "
        f"{type(model_output)!r} instead"
    )

def calc_wer(target, pred, ignore_indexes=[0]):
    target_chars = drop_duplicated(list(filter(lambda x: x not in ignore_indexes, map(str, list(target)))))
    pred_chars = drop_duplicated(list(filter(lambda x: x not in ignore_indexes, map(str, list(pred)))))
    target_str = ' '.join(target_chars)
    pred_str = ' '.join(pred_chars)
    error = jiwer.wer(target_str, pred_str)
    return error

def drop_duplicated(chars):
    ret_chars = [chars[0]]
    for prev, curr in zip(chars[:-1], chars[1:]):
        if prev != curr:
            ret_chars.append(curr)
    return ret_chars

def build_criterion(critic_params={}, entropy_params={}, multi_task_config=None):
    multi_task_config = multi_task_config or {}

    ctc_params = dict(critic_params.get('ctc', {}))
    ctc_params.setdefault('reduction', 'none')

    criterion = {
        "ce": nn.CrossEntropyLoss(ignore_index=-1, **entropy_params),
        "ctc": torch.nn.CTCLoss(**ctc_params),
    }

    frame_cfg = multi_task_config.get('frame_phoneme', {}) or {}
    if frame_cfg.get('enabled', False):
        criterion["frame_ce"] = nn.CrossEntropyLoss(ignore_index=-1, **entropy_params)

    speaker_cfg = multi_task_config.get('speaker', {}) or {}
    if speaker_cfg.get('enabled', False):
        criterion["speaker_ce"] = nn.CrossEntropyLoss()

    pron_cfg = multi_task_config.get('pronunciation_error', {}) or {}
    if pron_cfg.get('enabled', False):
        criterion["pron_error_ce"] = nn.CrossEntropyLoss(ignore_index=-1, **entropy_params)

    return criterion

def get_data_path_list(train_path=None, val_path=None, return_paths=False):
    """Return the metadata entries for the train/validation splits.

    Args:
        train_path: Optional path to the training metadata file.  When ``None``
            the default ``Data/train_list.txt`` is used.
        val_path: Optional path to the validation metadata file.  When ``None``
            the default ``Data/val_list.txt`` is used.
        return_paths: If ``True`` the resolved metadata file paths are returned
            alongside the contents.  This is useful for caching layers that
            need to reason about file modification times.

    Returns:
        Tuple containing the training and validation metadata lines.  When
        ``return_paths`` is enabled, the resolved file system paths are appended
        to the tuple.
    """

    train_path = Path(train_path) if train_path is not None else Path("Data") / "train_list.txt"
    val_path = Path(val_path) if val_path is not None else Path("Data") / "val_list.txt"

    with train_path.open('r', encoding='utf-8') as f:
        train_list = f.readlines()
    with val_path.open('r', encoding='utf-8') as f:
        val_list = f.readlines()

    if return_paths:
        return train_list, val_list, str(train_path), str(val_path)
    return train_list, val_list


def build_beam_search_decoder(config=None, vocab_size=None):
    """Construct a :class:`~decoding.CTCBeamSearchDecoder` from a config dict."""

    if config is None:
        return None
    try:
        if (
            vocab_size is not None
            and isinstance(config, dict)
            and "decoding" in config
            and "shallow_fusion" in config["decoding"]
        ):
            config = dict(config)
            decoding_cfg = dict(config.get("decoding", {}))
            shallow_cfg = dict(decoding_cfg.get("shallow_fusion", {}))
            shallow_cfg["vocab_size"] = int(vocab_size)
            decoding_cfg["shallow_fusion"] = shallow_cfg
            config["decoding"] = decoding_cfg
        decoder = build_decoder_from_config(config)
    except Exception:
        return None
    return decoder


def plot_image(image):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(image, aspect="auto", origin="lower",
                   interpolation='none')

    fig.canvas.draw()
    plt.close(fig)

    return fig

def diagonal_attention_prior(attn, text_lengths, mel_lengths, sigma=0.5):
    """Calculate diagonal attention loss."""
    B, T_text, T_mel = attn.size()  # usually [B, T_text, T_mel]
    device = attn.device

    # Normalize indices
    text_pos = torch.arange(T_text, device=device).unsqueeze(1).float() / T_text
    mel_pos = torch.arange(T_mel, device=device).unsqueeze(0).float() / T_mel
    expected = torch.exp(-((text_pos - mel_pos) ** 2) / (2 * sigma ** 2))  # [T_text, T_mel]
    expected = expected / expected.max()  # Normalize

    expected = expected.unsqueeze(0).expand(B, -1, -1)  # [B, T_text, T_mel]
    loss = torch.mean(attn * (1.0 - expected))  # Encourage attention mass to lie on the diagonal
    return loss


class BatchSizeScheduler:
    """Utility to manage curriculum batch-size schedules.

    The scheduler consumes the ``training_curriculum.batch_size_schedule``
    section from the configuration dictionary and produces the batch size that
    should be used for each epoch.  Two complementary strategies are supported:

    * ``milestones`` – Specify explicit ``epoch`` → ``batch_size`` pairs.  The
      batch size is held constant between milestones.
    * ``linear`` – Optionally enable linear interpolation between the milestone
      anchors.  The interpolation frequency can be controlled with
      ``update_interval`` so that the batch size only changes every ``n``
      epochs.

    Both strategies can be enabled at the same time.  In that case milestones
    provide the anchor points and the linear strategy interpolates between
    them.  When no milestones are supplied the linear configuration is used to
    derive the start and end anchors, falling back to ``initial_batch_size`` and
    ``base_batch_size`` respectively.

    The scheduler is deterministic and pre-computes the curriculum for all
    epochs, making it safe to query in notebooks or in the training loop.
    """

    def __init__(
        self,
        schedule_config: Optional[Dict] = None,
        default_batch_size: int = 1,
        total_epochs: int = 1,
    ) -> None:
        schedule_config = schedule_config or {}
        if not isinstance(schedule_config, dict):
            schedule_config = {}

        self.config = schedule_config
        self.enabled = bool(schedule_config.get("enabled", False))
        self.total_epochs = max(1, int(total_epochs))
        self.default_batch_size = max(1, int(default_batch_size))
        self.base_batch_size = max(
            1,
            int(schedule_config.get("base_batch_size", self.default_batch_size)),
        )
        self.initial_batch_size = max(
            1,
            int(schedule_config.get("initial_batch_size", self.default_batch_size)),
        )
        self.apply_to_validation = bool(schedule_config.get("apply_to_validation", False))

        eval_epoch = schedule_config.get("evaluation_epoch")
        try:
            self.evaluation_epoch = int(eval_epoch) if eval_epoch is not None else None
        except (TypeError, ValueError):
            self.evaluation_epoch = None

        strategies = schedule_config.get("strategies", {})
        if not isinstance(strategies, dict):
            strategies = {}
        self._milestone_cfg = strategies.get("milestones", {}) or {}
        if not isinstance(self._milestone_cfg, dict):
            self._milestone_cfg = {}
        self._linear_cfg = strategies.get("linear", {}) or {}
        if not isinstance(self._linear_cfg, dict):
            self._linear_cfg = {}

        self.linear_enabled = bool(self._linear_cfg.get("enabled", False))
        self.linear_update_interval = max(
            1,
            int(self._linear_cfg.get("update_interval", 1)),
        )

        self._anchors = self._build_anchors()
        self._epoch_schedule = self._build_epoch_schedule()

    def _build_anchors(self) -> List[Tuple[int, int]]:
        if not self.enabled:
            return [(1, self.default_batch_size), (self.total_epochs, self.default_batch_size)]

        anchors: Dict[int, int] = {}
        schedule_entries: Iterable = []
        if isinstance(self._milestone_cfg, dict):
            entries = self._milestone_cfg.get("schedule", [])
            if isinstance(entries, dict):
                schedule_entries = entries.items()
            else:
                schedule_entries = entries

        for entry in schedule_entries:
            if isinstance(entry, dict):
                epoch = entry.get("epoch", entry.get("start_epoch"))
                batch_size = entry.get("batch_size", entry.get("value"))
            elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                epoch, batch_size = entry[0], entry[1]
            else:
                continue

            try:
                epoch = int(epoch)
                batch_size = int(batch_size)
            except (TypeError, ValueError):
                continue

            if epoch < 1:
                epoch = 1

            anchors[epoch] = max(1, batch_size)

        if not anchors and self.linear_enabled:
            start_epoch = max(1, int(self._linear_cfg.get("start_epoch", 1)))
            end_epoch = max(start_epoch, int(self._linear_cfg.get("end_epoch", self.total_epochs)))
            start_bs = max(1, int(self._linear_cfg.get("start_batch_size", self.initial_batch_size)))
            end_bs = max(1, int(self._linear_cfg.get("end_batch_size", self.base_batch_size)))
            anchors[start_epoch] = start_bs
            anchors[end_epoch] = end_bs

        if 1 not in anchors:
            anchors[1] = self.initial_batch_size

        final_epoch = self.total_epochs
        final_value = anchors.get(final_epoch, self.base_batch_size)
        if self.linear_enabled:
            final_value = int(self._linear_cfg.get("end_batch_size", final_value))
        anchors[final_epoch] = max(1, final_value)

        filtered: Dict[int, int] = {}
        for epoch, batch_size in anchors.items():
            epoch = int(epoch)
            if epoch < 1:
                continue
            if epoch > self.total_epochs:
                epoch = self.total_epochs
            filtered[epoch] = max(1, int(batch_size))

        return sorted(filtered.items(), key=lambda item: item[0])

    def _build_epoch_schedule(self) -> List[int]:
        schedule = [self.default_batch_size] * (self.total_epochs + 1)
        if not self.enabled:
            for epoch in range(1, self.total_epochs + 1):
                schedule[epoch] = self.default_batch_size
            return schedule

        anchors = self._anchors
        if len(anchors) == 1:
            value = max(1, anchors[0][1])
            for epoch in range(1, self.total_epochs + 1):
                schedule[epoch] = value
            return schedule

        last_value = self.initial_batch_size
        for epoch in range(1, self.total_epochs + 1):
            prev_anchor = anchors[0]
            next_anchor = anchors[-1]
            for anchor in anchors:
                if anchor[0] <= epoch:
                    prev_anchor = anchor
                if anchor[0] >= epoch:
                    next_anchor = anchor
                    break

            if self.linear_enabled and next_anchor[0] > prev_anchor[0]:
                span = max(1, next_anchor[0] - prev_anchor[0])
                progress = (epoch - prev_anchor[0]) / span
                progress = min(max(progress, 0.0), 1.0)
                interpolated = prev_anchor[1] + progress * (next_anchor[1] - prev_anchor[1])
                value = int(round(interpolated))
                if (
                    self.linear_update_interval > 1
                    and epoch != prev_anchor[0]
                    and (epoch - prev_anchor[0]) % self.linear_update_interval != 0
                ):
                    value = last_value
            else:
                value = prev_anchor[1]

            value = max(1, int(value))
            schedule[epoch] = value
            last_value = value

        return schedule

    def batch_size_for_epoch(self, epoch: int) -> int:
        epoch = int(epoch)
        if epoch < 1:
            epoch = 1
        if epoch > self.total_epochs:
            epoch = self.total_epochs
        return self._epoch_schedule[epoch]

    def epoch_schedule(self) -> Dict[int, int]:
        return {epoch: self._epoch_schedule[epoch] for epoch in range(1, self.total_epochs + 1)}

    def summary(self, max_entries: int = 10) -> str:
        """Return a readable summary of schedule transitions."""

        transitions: List[Tuple[int, int]] = []
        last_value = None
        for epoch in range(1, self.total_epochs + 1):
            value = self._epoch_schedule[epoch]
            if value != last_value:
                transitions.append((epoch, value))
                last_value = value

        if len(transitions) > max_entries:
            head = transitions[: max_entries - 1]
            tail = transitions[-1:]
            summary_parts = ["%i→%i" % (e, v) for e, v in head]
            summary_parts.append("…")
            summary_parts.extend("%i→%i" % (e, v) for e, v in tail)
        else:
            summary_parts = ["%i→%i" % (e, v) for e, v in transitions]

        return ", ".join(summary_parts)

    def final_batch_size(self) -> int:
        return self.batch_size_for_epoch(self.total_epochs)

    def expected_total_steps(
        self,
        dataset_size: int,
        *,
        world_size: int = 1,
        per_rank_samples: Optional[int] = None,
    ) -> int:
        """Return the number of optimiser steps implied by the schedule.

        Args:
            dataset_size: Total number of training items across all processes.
            world_size: Number of data-parallel workers processing the dataset.
            per_rank_samples: Optional override for the number of samples seen by
                each rank in an epoch.  When provided, ``world_size`` is ignored
                and the step estimation is derived directly from this value.
        """

        samples_per_rank: int
        if per_rank_samples is not None:
            samples_per_rank = max(1, int(per_rank_samples))
        else:
            dataset_size = max(1, int(dataset_size))
            world_size = max(1, int(world_size))
            samples_per_rank = max(1, math.ceil(dataset_size / float(world_size)))

        total_steps = 0
        for epoch in range(1, self.total_epochs + 1):
            batch_size = max(1, self.batch_size_for_epoch(epoch))
            # ``len(dataloader)`` effectively performs a ``math.ceil`` over the
            # batch size, so mirror that behaviour here using the per-rank
            # dataset size.  This keeps OneCycle/linear warm-up schedulers in
            # sync with the actual number of optimisation steps executed on each
            # worker.
            steps = max(1, math.ceil(samples_per_rank / float(batch_size)))
            total_steps += steps
        return total_steps
