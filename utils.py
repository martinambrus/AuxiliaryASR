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

def build_criterion(critic_params={}, entropy_params={}):
    criterion = {
        "ce": nn.CrossEntropyLoss(ignore_index=-1, **entropy_params),
        "ctc": torch.nn.CTCLoss(**critic_params.get('ctc', {})),
    }
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


def attention_to_duration_targets(alignments,
                                  text_lengths,
                                  mel_lengths,
                                  detach_attention: bool = True):
    """Convert attention matrices to integer duration targets per token."""

    if alignments is None:
        return None

    if detach_attention:
        alignments = alignments.detach()

    alignments = alignments.float()
    batch_size = alignments.size(0)
    max_text_len = int(text_lengths.max().item()) if text_lengths.numel() > 0 else 0
    if max_text_len == 0:
        return None

    durations = alignments.new_zeros((batch_size, max_text_len))

    for idx in range(batch_size):
        text_len = int(text_lengths[idx].item())
        mel_len = int(mel_lengths[idx].item())
        if text_len <= 0 or mel_len <= 0:
            continue

        attn = alignments[idx]
        if attn.size(0) > text_len:
            attn = attn[1:text_len + 1, :mel_len]
        else:
            attn = attn[:text_len, :mel_len]

        if attn.numel() == 0:
            continue

        token_ids = torch.argmax(attn, dim=0)
        counts = torch.bincount(token_ids, minlength=text_len).float()
        durations[idx, :text_len] = counts

    return durations
