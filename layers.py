import torch
from torch import nn
from typing import Optional, Any, Dict, List
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


class FeedForwardModule(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 expansion_factor: float = 4.0,
                 dropout: float = 0.1,
                 activation: str = 'swish',
                 use_layer_norm: bool = True):
        super().__init__()
        inner_dim = int(hidden_dim * expansion_factor)
        self.layer_norm = nn.LayerNorm(hidden_dim) if use_layer_norm else None
        self.linear1 = nn.Linear(hidden_dim, inner_dim)
        self.activation = _get_activation_fn(activation)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(inner_dim, hidden_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x


class MultiHeadSelfAttentionModule(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 use_layer_norm: bool = True):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_dim) if use_layer_norm else None
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                x: Tensor,
                key_padding_mask: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None) -> Tensor:
        attn_input = x
        if self.layer_norm is not None:
            attn_input = self.layer_norm(attn_input)
        attn_output, _ = self.attention(
            attn_input,
            attn_input,
            attn_input,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=False,
        )
        attn_output = self.dropout(attn_output)
        return attn_output


class ConformerConvolutionModule(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 kernel_size: int = 31,
                 dropout: float = 0.1,
                 activation: str = 'swish',
                 use_layer_norm: bool = True):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for depthwise convolution")
        self.layer_norm = nn.LayerNorm(hidden_dim) if use_layer_norm else None
        self.pointwise_conv1 = nn.Conv1d(hidden_dim, 2 * hidden_dim, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=hidden_dim,
        )
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.activation = _get_activation_fn(activation)
        self.pointwise_conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = x.transpose(1, 2)
        x = self.dropout(x)
        return x


class ConformerBlock(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 num_heads: int = 4,
                 expansion_factor: float = 4.0,
                 dropout: float = 0.1,
                 conv_kernel_size: int = 31,
                 use_macaron: bool = True,
                 use_conv_module: bool = True,
                 activation: str = 'swish'):
        super().__init__()
        self.use_macaron = use_macaron
        self.ff_scale = 0.5 if use_macaron else 1.0
        self.feed_forward1 = FeedForwardModule(
            hidden_dim,
            expansion_factor=expansion_factor,
            dropout=dropout,
            activation=activation,
        )
        self.self_attn = MultiHeadSelfAttentionModule(
            hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.conv_module = ConformerConvolutionModule(
            hidden_dim,
            kernel_size=conv_kernel_size,
            dropout=dropout,
            activation=activation,
        ) if use_conv_module else None
        self.feed_forward2 = FeedForwardModule(
            hidden_dim,
            expansion_factor=expansion_factor,
            dropout=dropout,
            activation=activation,
        )
        self.final_layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self,
                x: Tensor,
                key_padding_mask: Optional[Tensor] = None) -> Tensor:
        if self.use_macaron:
            x = x + self.ff_scale * self.feed_forward1(x)
        else:
            x = x + self.feed_forward1(x)

        attn_out = self.self_attn(x, key_padding_mask=key_padding_mask)
        x = x + attn_out

        if self.conv_module is not None:
            conv_out = self.conv_module(x)
            x = x + conv_out

        x = x + self.ff_scale * self.feed_forward2(x)
        x = self.final_layer_norm(x)
        return x


class EmformerBlock(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 num_heads: int = 4,
                 expansion_factor: float = 4.0,
                 dropout: float = 0.1,
                 conv_kernel_size: int = 31,
                 chunk_size: int = 64,
                 left_context: int = 64,
                 right_context: int = 16,
                 activation: str = 'swish',
                 use_conv_module: bool = True):
        super().__init__()
        self.feed_forward = FeedForwardModule(
            hidden_dim,
            expansion_factor=expansion_factor,
            dropout=dropout,
            activation=activation,
        )
        self.self_attn = MultiHeadSelfAttentionModule(
            hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.conv_module = ConformerConvolutionModule(
            hidden_dim,
            kernel_size=conv_kernel_size,
            dropout=dropout,
            activation=activation,
        ) if use_conv_module else None
        self.final_layer_norm = nn.LayerNorm(hidden_dim)
        self.chunk_size = max(1, chunk_size)
        self.left_context = max(0, left_context)
        self.right_context = max(0, right_context)
        self._cached_base_masks: Dict[int, Tensor] = {}

    def _build_local_mask(self, seq_len: int, device: torch.device) -> Optional[Tensor]:
        total_left = self.left_context + self.chunk_size - 1
        total_right = self.right_context
        window = total_left + total_right + 1
        if window >= seq_len:
            return None
        if seq_len not in self._cached_base_masks:
            base_mask = torch.full((seq_len, seq_len), float('-inf'))
            for index in range(seq_len):
                start = max(0, index - total_left)
                end = min(seq_len, index + total_right + 1)
                base_mask[index, start:end] = 0.0
            self._cached_base_masks[seq_len] = base_mask
        mask = self._cached_base_masks[seq_len]
        if mask.device != device:
            mask = mask.to(device)
        return mask

    def forward(self,
                x: Tensor,
                key_padding_mask: Optional[Tensor] = None) -> Tensor:
        x = x + self.feed_forward(x)
        attn_mask = self._build_local_mask(x.size(1), x.device)
        attn_out = self.self_attn(
            x,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
        )
        x = x + attn_out

        if self.conv_module is not None:
            x = x + self.conv_module(x)

        x = self.final_layer_norm(x)
        return x


class BranchformerBlock(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 num_heads: int = 4,
                 expansion_factor: float = 4.0,
                 dropout: float = 0.1,
                 conv_kernel_size: int = 31,
                 activation: str = 'swish',
                 enable_attn_branch: bool = True,
                 enable_conv_branch: bool = True,
                 enable_ffn_branch: bool = True,
                 branch_dropout: float = 0.1,
                 post_ffn: bool = True):
        super().__init__()
        self.pre_norm = nn.LayerNorm(hidden_dim)
        self.enable_attn_branch = enable_attn_branch
        self.enable_conv_branch = enable_conv_branch
        self.enable_ffn_branch = enable_ffn_branch
        self.branch_dropout = nn.Dropout(branch_dropout)
        self.attn_branch = MultiHeadSelfAttentionModule(
            hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_layer_norm=False,
        ) if enable_attn_branch else None
        self.conv_branch = ConformerConvolutionModule(
            hidden_dim,
            kernel_size=conv_kernel_size,
            dropout=dropout,
            activation=activation,
            use_layer_norm=False,
        ) if enable_conv_branch else None
        self.ffn_branch = FeedForwardModule(
            hidden_dim,
            expansion_factor=expansion_factor,
            dropout=dropout,
            activation=activation,
            use_layer_norm=False,
        ) if enable_ffn_branch else None
        self.num_branches = sum([
            1 if enable_attn_branch else 0,
            1 if enable_conv_branch else 0,
            1 if enable_ffn_branch else 0,
        ])
        self.branch_weights = nn.Parameter(torch.ones(self.num_branches)) if self.num_branches > 1 else None
        self.post_ffn = FeedForwardModule(
            hidden_dim,
            expansion_factor=expansion_factor,
            dropout=dropout,
            activation=activation,
        ) if post_ffn else None
        self.final_layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self,
                x: Tensor,
                key_padding_mask: Optional[Tensor] = None) -> Tensor:
        residual = x
        x_norm = self.pre_norm(x)
        branches: List[Tensor] = []
        if self.attn_branch is not None:
            attn_out = self.attn_branch(x_norm, key_padding_mask=key_padding_mask)
            branches.append(attn_out)
        if self.conv_branch is not None:
            conv_out = self.conv_branch(x_norm)
            branches.append(conv_out)
        if self.ffn_branch is not None:
            ffn_out = self.ffn_branch(x_norm)
            branches.append(ffn_out)

        if not branches:
            mixed = torch.zeros_like(x)
        elif len(branches) == 1:
            mixed = branches[0]
        else:
            stacked = torch.stack(branches, dim=-1)  # (B, T, H, num_branches)
            weights = torch.softmax(self.branch_weights, dim=0)
            weights = weights.view(1, 1, 1, -1)
            mixed = (stacked * weights).sum(dim=-1)

        mixed = self.branch_dropout(mixed)
        x = residual + mixed

        if self.post_ffn is not None:
            x = x + self.post_ffn(x)

        x = self.final_layer_norm(x)
        return x


class ZipformerBlock(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 num_heads: int = 4,
                 expansion_factor: float = 4.0,
                 dropout: float = 0.1,
                 conv_kernel_size: int = 31,
                 activation: str = 'swish',
                 use_attention: bool = True,
                 use_convolution: bool = True,
                 use_ffn_stream: bool = True,
                 post_ffn: bool = True):
        super().__init__()
        self.pre_norm = nn.LayerNorm(hidden_dim)
        self.use_attention = use_attention
        self.use_convolution = use_convolution
        self.use_ffn_stream = use_ffn_stream
        self.attn_stream = MultiHeadSelfAttentionModule(
            hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_layer_norm=False,
        ) if use_attention else None
        self.conv_stream = ConformerConvolutionModule(
            hidden_dim,
            kernel_size=conv_kernel_size,
            dropout=dropout,
            activation=activation,
            use_layer_norm=False,
        ) if use_convolution else None
        self.ffn_stream = FeedForwardModule(
            hidden_dim,
            expansion_factor=expansion_factor,
            dropout=dropout,
            activation=activation,
            use_layer_norm=False,
        ) if use_ffn_stream else None
        self.num_streams = sum([
            1 if use_attention else 0,
            1 if use_convolution else 0,
            1 if use_ffn_stream else 0,
        ])
        self.gate_proj = nn.Linear(hidden_dim, self.num_streams) if self.num_streams > 1 else None
        self.stream_dropout = nn.Dropout(dropout)
        self.post_ffn = FeedForwardModule(
            hidden_dim,
            expansion_factor=expansion_factor,
            dropout=dropout,
            activation=activation,
        ) if post_ffn else None
        self.final_layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self,
                x: Tensor,
                key_padding_mask: Optional[Tensor] = None) -> Tensor:
        residual = x
        x_norm = self.pre_norm(x)
        streams: List[Tensor] = []
        if self.attn_stream is not None:
            streams.append(self.attn_stream(x_norm, key_padding_mask=key_padding_mask))
        if self.conv_stream is not None:
            streams.append(self.conv_stream(x_norm))
        if self.ffn_stream is not None:
            streams.append(self.ffn_stream(x_norm))

        if not streams:
            mixed = torch.zeros_like(x)
        elif len(streams) == 1:
            mixed = streams[0]
        else:
            stacked = torch.stack(streams, dim=-1)  # (B, T, H, num_streams)
            gate_logits = self.gate_proj(x_norm)
            gate = torch.softmax(gate_logits, dim=-1).unsqueeze(-2)
            mixed = (stacked * gate).sum(dim=-1)

        mixed = self.stream_dropout(mixed)
        x = residual + mixed

        if self.post_ffn is not None:
            x = x + self.post_ffn(x)

        x = self.final_layer_norm(x)
        return x


class HybridEncoder(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 block_settings: List[Dict[str, Any]],
                 global_params: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.global_params = global_params or {}
        self.blocks = nn.ModuleList([])
        for setting in block_settings:
            block_type = setting.get('type', 'conformer').lower()
            layers = setting.get('layers', 1)
            block_kwargs = dict(self.global_params)
            block_kwargs.update(setting.get('params', {}))
            for _ in range(layers):
                block = self._create_block(block_type, block_kwargs)
                self.blocks.append(block)

    def _create_block(self, block_type: str, params: Dict[str, Any]) -> nn.Module:
        common_args = dict(
            hidden_dim=self.hidden_dim,
            num_heads=params.get('num_heads', 4),
            expansion_factor=params.get('expansion_factor', 4.0),
            dropout=params.get('dropout', 0.1),
            conv_kernel_size=params.get('conv_kernel_size', 31),
            activation=params.get('activation', 'swish'),
        )
        if block_type == 'conformer':
            return ConformerBlock(
                **common_args,
                use_macaron=params.get('use_macaron', True),
                use_conv_module=params.get('use_conv_module', True),
            )
        if block_type == 'emformer':
            return EmformerBlock(
                **common_args,
                chunk_size=params.get('chunk_size', 64),
                left_context=params.get('left_context', 64),
                right_context=params.get('right_context', 16),
                use_conv_module=params.get('use_conv_module', True),
            )
        if block_type == 'branchformer':
            return BranchformerBlock(
                **common_args,
                enable_attn_branch=params.get('enable_attn_branch', True),
                enable_conv_branch=params.get('enable_conv_branch', True),
                enable_ffn_branch=params.get('enable_ffn_branch', True),
                branch_dropout=params.get('branch_dropout', 0.1),
                post_ffn=params.get('post_ffn', True),
            )
        if block_type == 'zipformer':
            return ZipformerBlock(
                **common_args,
                use_attention=params.get('use_attention', True),
                use_convolution=params.get('use_convolution', True),
                use_ffn_stream=params.get('use_ffn_stream', True),
                post_ffn=params.get('post_ffn', True),
            )
        raise ValueError(f"Unsupported encoder block type: {block_type}")

    def forward(self,
                x: Tensor,
                key_padding_mask: Optional[Tensor] = None) -> Tensor:
        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)
        return x

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
