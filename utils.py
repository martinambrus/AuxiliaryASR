import os
import os.path as osp
import sys
import time
from collections import defaultdict

import matplotlib
import numpy as np
import soundfile as sf
import torch
from torch import nn
import jiwer

import matplotlib.pylab as plt
from pathlib import Path

try:
    from torchaudio.functional import rnnt_loss as torchaudio_rnnt_loss
except ImportError:  # pragma: no cover - torchaudio might be unavailable during testing
    torchaudio_rnnt_loss = None

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

class RNNTLossWrapper(nn.Module):
    def __init__(
        self,
        blank=0,
        reduction="mean",
        clamp=-1,
        fused_log_softmax=True,
        input_is_log_probs=False,
        normalize_by_target_length=True,
        normalize_by_logit_length=False,
        length_norm_epsilon=1e-6,
    ):
        super().__init__()
        if torchaudio_rnnt_loss is None:
            raise ImportError("torchaudio is required to compute RNNT loss but is not available.")
        self.blank = blank
        self.reduction = reduction
        self.clamp = clamp
        self.fused_log_softmax = fused_log_softmax
        self.input_is_log_probs = input_is_log_probs
        self.normalize_by_target_length = normalize_by_target_length
        self.normalize_by_logit_length = normalize_by_logit_length
        self.length_norm_epsilon = length_norm_epsilon

        if self.input_is_log_probs and self.fused_log_softmax:
            raise ValueError(
                "RNNTLossWrapper cannot receive pre-normalized log probabilities "
                "when fused_log_softmax is enabled."
            )

    def forward(self, logits, targets, logit_lengths, target_lengths):
        if self.normalize_by_target_length or self.normalize_by_logit_length:
            raw_reduction = "none"
        else:
            raw_reduction = self.reduction

        if self.input_is_log_probs:
            loss_input = logits
            fused_log_softmax = False
        else:
            if self.fused_log_softmax:
                loss_input = logits
                fused_log_softmax = True
            else:
                loss_input = logits.log_softmax(dim=-1)
                fused_log_softmax = False

        losses = torchaudio_rnnt_loss(
            loss_input,
            targets,
            logit_lengths,
            target_lengths,
            blank=self.blank,
            reduction=raw_reduction,
            clamp=self.clamp,
            fused_log_softmax=fused_log_softmax,
        )

        if not (self.normalize_by_target_length or self.normalize_by_logit_length):
            return losses

        # convert to float tensors on the same device for safe normalization
        if self.normalize_by_target_length:
            norm = target_lengths.to(device=losses.device, dtype=losses.dtype)
            norm = norm.clamp_min(self.length_norm_epsilon)
            losses = losses / norm

        if self.normalize_by_logit_length:
            norm = logit_lengths.to(device=losses.device, dtype=losses.dtype)
            norm = norm.clamp_min(self.length_norm_epsilon)
            losses = losses / norm

        if self.reduction == "sum":
            return losses.sum()
        if self.reduction == "mean":
            return losses.mean()
        if self.reduction == "none":
            return losses

        raise ValueError(f"Unsupported reduction: {self.reduction}")


def build_criterion(critic_params=None, entropy_params=None):
    critic_params = critic_params or {}
    entropy_params = entropy_params or {}
    criterion = {}

    if "ce" in critic_params:
        ce_config = critic_params.get("ce") or {}
        ce_kwargs = dict(ce_config)
        ce_kwargs.setdefault("ignore_index", -1)
        ce_kwargs.update(entropy_params)
        criterion["ce"] = nn.CrossEntropyLoss(**ce_kwargs)

    if "ctc" in critic_params:
        ctc_kwargs = critic_params.get("ctc") or {}
        criterion["ctc"] = torch.nn.CTCLoss(**ctc_kwargs)

    if "rnnt" in critic_params:
        rnnt_kwargs = critic_params.get("rnnt") or {}
        criterion["rnnt"] = RNNTLossWrapper(**rnnt_kwargs)

    return criterion

def get_data_path_list(train_path=None, val_path=None):
    train_path = Path(train_path) if train_path is not None else Path("Data") / "train_list.txt"
    val_path = Path(val_path) if val_path is not None else Path("Data") / "val_list.txt"

    with train_path.open('r') as f:
        train_list = f.readlines()
    with val_path.open('r') as f:
        val_list = f.readlines()

    return train_list, val_list


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