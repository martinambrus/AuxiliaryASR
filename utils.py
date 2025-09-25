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

def build_criterion(critic_params={}, entropy_params={}, multi_task_config=None):
    multi_task_config = multi_task_config or {}

    criterion = {
        "ce": nn.CrossEntropyLoss(ignore_index=-1, **entropy_params),
        "ctc": torch.nn.CTCLoss(**critic_params.get('ctc', {})),
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