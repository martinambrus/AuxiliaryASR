import math
import torch
from torch import nn
from typing import Optional, Any
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


class LocationSensitiveAttention(nn.Module):
    """Location-sensitive attention mechanism.

    This is the original attention used by the model and kept for backward
    compatibility. It computes attention energies using the previous alignment
    history that is processed with a small convolutional network.
    """

    attention_type = "location"

    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size):
        super().__init__()
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
        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory))

        energies = energies.squeeze(-1)
        return energies

    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat=None, mask=None, **_):
        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat)

        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights, {
            'alpha': attention_weights
        }


class MonotonicAttention(nn.Module):
    """Parallel monotonic attention as described by Raffel et al. (2017).

    The implementation follows the expectation-based formulation that can be
    evaluated efficiently with a linear pass over the encoder time steps. It is
    compatible with teacher-forced training as well as autoregressive decoding.
    """

    attention_type = "monotonic"

    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 sigmoid_noise=0.0):
        super().__init__()
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')
        self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False,
                                       w_init_gain='tanh')
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.sigmoid_noise = sigmoid_noise
        self.score_mask_value = -float("inf")
        self.eps = 1e-8

    def _apply_noise(self, energies, training):
        if self.sigmoid_noise > 0.0 and training:
            noise = torch.randn_like(energies) * self.sigmoid_noise
            energies = energies + noise
        return energies

    def _sigmoid(self, energies):
        return torch.sigmoid(energies)

    def _monotonic_attention(self, p_choose, prev_alpha):
        """Compute the expected monotonic attention distribution.

        Args:
            p_choose: Probability of choosing each encoder state. Shape (B, T).
            prev_alpha: Previous attention distribution. Shape (B, T).
        Returns:
            alpha: Updated attention distribution. Shape (B, T).
        """
        batch, time = p_choose.size()
        alpha = torch.zeros_like(p_choose)

        # First position is a special case without the recursive term.
        alpha[:, 0] = p_choose[:, 0] * prev_alpha[:, 0]

        for t in range(1, time):
            stay_prob = (1.0 - p_choose[:, t - 1]) * alpha[:, t - 1]
            stay_prob = stay_prob / torch.clamp(p_choose[:, t - 1], min=self.eps)
            alpha[:, t] = p_choose[:, t] * (prev_alpha[:, t] + stay_prob)

        # Normalise to make sure the alignment sums to one even with masking.
        alpha_sum = alpha.sum(dim=-1, keepdim=True)
        alpha = alpha / torch.clamp(alpha_sum, min=self.eps)
        return alpha

    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat=None, mask=None, prev_attention=None,
                training=True):
        if prev_attention is None:
            raise ValueError("Monotonic attention requires previous attention.")

        processed_query = self.query_layer(attention_hidden_state).unsqueeze(1)
        energies = self.v(torch.tanh(processed_query + processed_memory)).squeeze(-1)

        if mask is not None:
            energies = energies.masked_fill(mask, self.score_mask_value)

        energies = self._apply_noise(energies, training)
        p_choose = self._sigmoid(energies)

        if mask is not None:
            p_choose = p_choose.masked_fill(mask, 0.0)

        alpha = self._monotonic_attention(p_choose, prev_attention)
        attention_context = torch.bmm(alpha.unsqueeze(1), memory).squeeze(1)

        return attention_context, alpha, {
            'alpha': alpha
        }


class MoChAAttention(MonotonicAttention):
    """Monotonic Chunkwise Attention (MoChA).

    Extends monotonic attention by performing a soft attention inside a
    fixed-length chunk after the monotonic boundary is determined. The
    expectation-based formulation is used to keep the operation differentiable
    during training.
    """

    attention_type = "mocha"

    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 chunk_size=4, sigmoid_noise=0.0):
        super().__init__(attention_rnn_dim, embedding_dim, attention_dim,
                         sigmoid_noise=sigmoid_noise)
        self.chunk_size = int(max(1, chunk_size))
        self.chunk_energy = LinearNorm(attention_dim, 1, bias=False)

    def _chunkwise_attention(self, alpha, chunk_energy, mask):
        exp_energy = torch.exp(chunk_energy)
        if mask is not None:
            exp_energy = exp_energy.masked_fill(mask, 0.0)

        batch, time = alpha.size()
        beta = torch.zeros_like(alpha)

        # Pre-compute normalisation terms for each potential chunk.
        denom = torch.zeros_like(exp_energy)
        for idx in range(time):
            start = max(0, idx - self.chunk_size + 1)
            denom[:, idx] = exp_energy[:, start:idx + 1].sum(dim=-1)

        eps = self.eps
        inv_denom = torch.where(denom > 0, 1.0 / torch.clamp(denom, min=eps), torch.zeros_like(denom))

        for j in range(time):
            upper = min(time, j + self.chunk_size)
            contrib = torch.zeros(batch, device=alpha.device, dtype=alpha.dtype)
            for k in range(j, upper):
                contrib = contrib + alpha[:, k] * inv_denom[:, k]
            beta[:, j] = exp_energy[:, j] * contrib

        beta_sum = beta.sum(dim=-1, keepdim=True)
        beta = beta / torch.clamp(beta_sum, min=eps)
        return beta

    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat=None, mask=None, prev_attention=None,
                training=True):
        if prev_attention is None:
            raise ValueError("MoChA attention requires previous attention.")

        processed_query = self.query_layer(attention_hidden_state).unsqueeze(1)
        energies = self.v(torch.tanh(processed_query + processed_memory)).squeeze(-1)
        chunk_energy = self.chunk_energy(torch.tanh(processed_query + processed_memory)).squeeze(-1)

        if mask is not None:
            energies = energies.masked_fill(mask, self.score_mask_value)
            chunk_energy = chunk_energy.masked_fill(mask, self.score_mask_value)

        energies = self._apply_noise(energies, training)
        p_choose = self._sigmoid(energies)
        if mask is not None:
            p_choose = p_choose.masked_fill(mask, 0.0)

        alpha = self._monotonic_attention(p_choose, prev_attention)

        beta = self._chunkwise_attention(alpha, chunk_energy, mask)
        attention_context = torch.bmm(beta.unsqueeze(1), memory).squeeze(1)

        return attention_context, beta, {
            'alpha': alpha,
            'beta': beta
        }


# Backwards compatibility alias.
Attention = LocationSensitiveAttention

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
