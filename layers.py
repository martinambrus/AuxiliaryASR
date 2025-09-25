import math
import torch
from torch import nn
from typing import Optional
from torch import Tensor
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as audio_F

import random
random.seed(0)


def _get_activation_fn(activ):
    if activ == 'relu':
        return nn.ReLU()
    elif activ == 'lrelu':
        return nn.LeakyReLU(0.2)
    elif activ == 'swish':
        return lambda x: x*torch.sigmoid(x)
    else:
        raise RuntimeError('Unexpected activ type %s, expected [relu, lrelu, swish]' % activ)

class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear', param=None):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain, param=param))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal

class CausualConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=1, dilation=1, bias=True, w_init_gain='linear', param=None):
        super(CausualConv, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2) * 2
        else:
            self.padding = padding * 2
        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=self.padding,
                              dilation=dilation,
                              bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain, param=param))

    def forward(self, x):
        x = self.conv(x)
        x = x[:, :, :-self.padding]
        return x

class CausualBlock(nn.Module):
    def __init__(self, hidden_dim, n_conv=3, dropout_p=0.2, activ='lrelu'):
        super(CausualBlock, self).__init__()
        self.blocks = nn.ModuleList([
            self._get_conv(hidden_dim, dilation=3**i, activ=activ, dropout_p=dropout_p)
            for i in range(n_conv)])

    def forward(self, x):
        for block in self.blocks:
            res = x
            x = block(x)
            x += res
        return x

    def _get_conv(self, hidden_dim, dilation, activ='lrelu', dropout_p=0.2):
        layers = [
            CausualConv(hidden_dim, hidden_dim, kernel_size=3, padding=dilation, dilation=dilation),
            _get_activation_fn(activ),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p=dropout_p),
            CausualConv(hidden_dim, hidden_dim, kernel_size=3, padding=1, dilation=1),
            _get_activation_fn(activ),
            nn.Dropout(p=dropout_p)
        ]
        return nn.Sequential(*layers)

class ConvBlock(nn.Module):
    def __init__(self, hidden_dim, n_conv=3, dropout_p=0.2, activ='relu'):
        super().__init__()
        self._n_groups = 8
        self.blocks = nn.ModuleList([
            self._get_conv(hidden_dim, dilation=3**i, activ=activ, dropout_p=dropout_p)
            for i in range(n_conv)])


    def forward(self, x):
        for block in self.blocks:
            res = x
            x = block(x)
            x += res
        return x

    def _get_conv(self, hidden_dim, dilation, activ='relu', dropout_p=0.2):
        layers = [
            ConvNorm(hidden_dim, hidden_dim, kernel_size=3, padding=dilation, dilation=dilation),
            _get_activation_fn(activ),
            nn.GroupNorm(num_groups=self._n_groups, num_channels=hidden_dim),
            nn.Dropout(p=dropout_p),
            ConvNorm(hidden_dim, hidden_dim, kernel_size=3, padding=1, dilation=1),
            _get_activation_fn(activ),
            nn.Dropout(p=dropout_p)
        ]
        return nn.Sequential(*layers)

class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size,
                 attention_dim):
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(2, attention_n_filters,
                                      kernel_size=attention_kernel_size,
                                      padding=padding, bias=False, stride=1,
                                      dilation=1)
        self.location_dense = LinearNorm(attention_n_filters, attention_dim,
                                         bias=False, w_init_gain='tanh')

    def forward(self, attention_weights_cat):
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention


class Attention(nn.Module):
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size):
        super(Attention, self).__init__()
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')
        self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False,
                                       w_init_gain='tanh')
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim)
        self.score_mask_value = -float("inf")

    def get_alignment_energies(self, query, processed_memory,
                               attention_weights_cat):
        """
        PARAMS
        ------
        query: decoder output (batch, n_mel_channels * n_frames_per_step)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)
        RETURNS
        -------
        alignment (batch, max_time)
        """

        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory))

        energies = energies.squeeze(-1)
        return energies

    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """
        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat)

        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights


def _monotonic_attention(p_select: Tensor, prev_attention: Tensor, mask: Optional[Tensor] = None,
                         eps: float = 1e-8) -> Tensor:
    """Compute expected monotonic attention distribution.

    Args:
        p_select: Selection probabilities for the current decoding step (B, T).
        prev_attention: Attention distribution from the previous decoding step (B, T).
        mask: Optional boolean mask where ``True`` indicates padded positions (B, T).
        eps: Small constant for numerical stability.

    Returns:
        Tensor: Normalized attention weights with shape (B, T).
    """

    if mask is not None:
        p_select = p_select.masked_fill(mask, 0.0)
        prev_attention = prev_attention.masked_fill(mask, 0.0)

    batch, seq_len = p_select.size()
    cumulative = torch.zeros(batch, device=p_select.device, dtype=p_select.dtype)
    attention = []

    for step in range(seq_len):
        prev = prev_attention[:, step]
        cumulative = cumulative * (1.0 - p_select[:, step]) + prev
        attn_step = p_select[:, step] * cumulative
        attention.append(attn_step)

    attention = torch.stack(attention, dim=-1)

    if mask is not None:
        attention = attention.masked_fill(mask, 0.0)

    denom = attention.sum(-1, keepdim=True)
    normalized = attention / (denom + eps)

    zero_mask = denom.squeeze(-1) <= eps
    if zero_mask.any():
        prev_norm = prev_attention / (prev_attention.sum(-1, keepdim=True) + eps)
        prev_norm = prev_norm.masked_fill(mask, 0.0) if mask is not None else prev_norm
        normalized = torch.where(zero_mask.unsqueeze(-1), prev_norm, normalized)

    return normalized


def _compute_chunkwise_attention(alpha: Tensor, chunk_energy: Tensor, chunk_size: int,
                                 mask: Optional[Tensor] = None,
                                 score_mask_value: float = -1e4,
                                 eps: float = 1e-8) -> Tensor:
    """Distribute monotonic attention mass over chunk windows.

    Args:
        alpha: Monotonic attention distribution (B, T).
        chunk_energy: Chunk energy scores (B, T).
        chunk_size: Size of the chunk window.
        mask: Optional boolean mask for padded positions (B, T).
        score_mask_value: Mask value for invalid locations.
        eps: Numerical stability constant.

    Returns:
        Tensor: Chunk-wise attention distribution (B, T).
    """

    if chunk_size <= 1:
        beta = alpha.clone()
        if mask is not None:
            beta = beta.masked_fill(mask, 0.0)
        denom = beta.sum(-1, keepdim=True)
        return beta / (denom + eps)

    batch, seq_len = alpha.size()
    chunk_size = min(chunk_size, seq_len)

    device = alpha.device
    idx = torch.arange(seq_len, device=device)
    end_idx = idx.unsqueeze(1)
    start_idx = idx.unsqueeze(0)
    base_mask = (start_idx <= end_idx) & ((end_idx - start_idx) < chunk_size)
    base_mask = base_mask.unsqueeze(0).expand(batch, -1, -1)

    energy_matrix = chunk_energy.unsqueeze(1).expand(-1, seq_len, -1)
    energy_matrix = energy_matrix.masked_fill(~base_mask, score_mask_value)

    if mask is not None:
        expanded_mask = mask.unsqueeze(1).expand(-1, seq_len, -1)
        energy_matrix = energy_matrix.masked_fill(expanded_mask, score_mask_value)
        alpha = alpha.masked_fill(mask, 0.0)

    chunk_weights = torch.softmax(energy_matrix, dim=-1)
    chunk_weights = chunk_weights * base_mask.to(chunk_weights.dtype)

    denom = chunk_weights.sum(-1, keepdim=True).clamp_min(eps)
    chunk_weights = chunk_weights / denom

    beta = torch.bmm(alpha.unsqueeze(1), chunk_weights).squeeze(1)

    if mask is not None:
        beta = beta.masked_fill(mask, 0.0)

    beta_sum = beta.sum(-1, keepdim=True)
    normalized_beta = beta / (beta_sum.clamp_min(eps))

    zero_mask = beta_sum.squeeze(-1) <= eps
    if zero_mask.any():
        fallback = alpha / (alpha.sum(-1, keepdim=True).clamp_min(eps))
        fallback = fallback.masked_fill(mask, 0.0) if mask is not None else fallback
        normalized_beta = torch.where(zero_mask.unsqueeze(-1), fallback, normalized_beta)

    return normalized_beta


class MonotonicAttention(nn.Module):
    def __init__(self,
                 attention_rnn_dim,
                 embedding_dim,
                 attention_dim,
                 attention_location_n_filters=32,
                 attention_location_kernel_size=31,
                 use_location_features=True,
                 noise_std=0.0):
        super().__init__()
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')
        self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False,
                                       w_init_gain='tanh')
        self.energy_layer = LinearNorm(attention_dim, 1, bias=False)
        self.use_location_features = use_location_features
        if self.use_location_features:
            self.location_layer = LocationLayer(
                attention_location_n_filters,
                attention_location_kernel_size,
                attention_dim,
            )
        else:
            self.location_layer = None
        self.score_mask_value = -1e4
        self.noise_std = noise_std

    def _compute_energy(self, query: Tensor, processed_memory: Tensor,
                         attention_weights_cat: Optional[Tensor]) -> Tensor:
        processed_query = self.query_layer(query).unsqueeze(1)
        energies = processed_query + processed_memory
        if self.use_location_features and attention_weights_cat is not None:
            processed_attention = self.location_layer(attention_weights_cat)
            energies = energies + processed_attention
        return torch.tanh(energies)

    def forward(self, attention_hidden_state: Tensor, memory: Tensor,
                processed_memory: Tensor, attention_weights_cat: Optional[Tensor],
                mask: Optional[Tensor]):
        energies = self._compute_energy(attention_hidden_state,
                                        processed_memory,
                                        attention_weights_cat)
        energy_scores = self.energy_layer(energies).squeeze(-1)
        if mask is not None:
            energy_scores = energy_scores.masked_fill(mask, self.score_mask_value)
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(energy_scores) * self.noise_std
            energy_scores = energy_scores + noise
        p_select = torch.sigmoid(energy_scores)
        prev_attention = attention_weights_cat[:, 0, :] if attention_weights_cat is not None else None
        if prev_attention is None:
            prev_attention = torch.zeros_like(p_select)
            prev_attention[:, 0] = 1.0
        attention_weights = _monotonic_attention(p_select, prev_attention, mask)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory).squeeze(1)
        return attention_context, attention_weights


class MoChAAttention(MonotonicAttention):
    def __init__(self,
                 attention_rnn_dim,
                 embedding_dim,
                 attention_dim,
                 attention_location_n_filters=32,
                 attention_location_kernel_size=31,
                 use_location_features=True,
                 noise_std=0.0,
                 chunk_size=4):
        super().__init__(attention_rnn_dim,
                         embedding_dim,
                         attention_dim,
                         attention_location_n_filters,
                         attention_location_kernel_size,
                         use_location_features,
                         noise_std)
        self.chunk_size = chunk_size
        self.chunk_energy_layer = LinearNorm(attention_dim, 1, bias=False)

    def forward(self, attention_hidden_state: Tensor, memory: Tensor,
                processed_memory: Tensor, attention_weights_cat: Optional[Tensor],
                mask: Optional[Tensor]):
        energies = self._compute_energy(attention_hidden_state,
                                        processed_memory,
                                        attention_weights_cat)

        monotonic_scores = self.energy_layer(energies).squeeze(-1)
        chunk_scores = self.chunk_energy_layer(energies).squeeze(-1)

        if mask is not None:
            monotonic_scores = monotonic_scores.masked_fill(mask, self.score_mask_value)
            chunk_scores = chunk_scores.masked_fill(mask, self.score_mask_value)

        if self.training and self.noise_std > 0:
            noise = torch.randn_like(monotonic_scores) * self.noise_std
            monotonic_scores = monotonic_scores + noise

        p_select = torch.sigmoid(monotonic_scores)
        prev_attention = attention_weights_cat[:, 0, :] if attention_weights_cat is not None else None
        if prev_attention is None:
            prev_attention = torch.zeros_like(p_select)
            prev_attention[:, 0] = 1.0

        alpha = _monotonic_attention(p_select, prev_attention, mask)
        beta = _compute_chunkwise_attention(alpha, chunk_scores, self.chunk_size, mask,
                                            score_mask_value=self.score_mask_value)
        attention_context = torch.bmm(beta.unsqueeze(1), memory).squeeze(1)
        return attention_context, beta

class PhaseShuffle2d(nn.Module):
    def __init__(self, n=2):
        super(PhaseShuffle2d, self).__init__()
        self.n = n
        self.random = random.Random(1)

    def forward(self, x, move=None):
        # x.size = (B, C, M, L)
        if move is None:
            move = self.random.randint(-self.n, self.n)

        if move == 0:
            return x
        else:
            left = x[:, :, :, :move]
            right = x[:, :, :, move:]
            shuffled = torch.cat([right, left], dim=3)
        return shuffled

class PhaseShuffle1d(nn.Module):
    def __init__(self, n=2):
        super(PhaseShuffle1d, self).__init__()
        self.n = n
        self.random = random.Random(1)

    def forward(self, x, move=None):
        # x.size = (B, C, M, L)
        if move is None:
            move = self.random.randint(-self.n, self.n)

        if move == 0:
            return x
        else:
            left = x[:, :,  :move]
            right = x[:, :, move:]
            shuffled = torch.cat([right, left], dim=2)

        return shuffled

class MFCC(nn.Module):
    def __init__(self, n_mfcc=40, n_mels=80):
        super(MFCC, self).__init__()
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.norm = 'ortho'
        dct_mat = audio_F.create_dct(self.n_mfcc, self.n_mels, self.norm)
        self.register_buffer('dct_mat', dct_mat)

    def forward(self, mel_specgram):
        if len(mel_specgram.shape) == 2:
            mel_specgram = mel_specgram.unsqueeze(0)
            unsqueezed = True
        else:
            unsqueezed = False
        # (channel, n_mels, time).tranpose(...) dot (n_mels, n_mfcc)
        # -> (channel, time, n_mfcc).tranpose(...)
        mfcc = torch.matmul(mel_specgram.transpose(1, 2), self.dct_mat).transpose(1, 2)

        # unpack batch
        if unsqueezed:
            mfcc = mfcc.squeeze(0)
        return mfcc
