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

def get_data_path_list(train_path=None, val_path=None):
    train_path = Path(train_path) if train_path is not None else Path("Data") / "train_list.txt"
    val_path = Path(val_path) if val_path is not None else Path("Data") / "val_list.txt"

    with train_path.open('r') as f:
        train_list = f.readlines()
    with val_path.open('r') as f:
        val_list = f.readlines()

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