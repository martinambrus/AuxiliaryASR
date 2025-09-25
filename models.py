import math
from typing import Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from layers import MFCC, Attention, LinearNorm, ConvNorm, ConvBlock

def build_model(model_params: Optional[Dict] = None, model_type: str = 'asr'):
    model_params = model_params or {}
    model = ASRCNN(**model_params)
    return model


class ASRCNN(nn.Module):
    def __init__(self,
                 input_dim=80,
                 hidden_dim=256,
                 n_token=35,
                 n_layers=6,
                 token_embedding_dim=256,
                 location_kernel_size=63,
                 decoder_type='conformer',
                 decoder_config=None,
    ):
        super().__init__()
        self.n_token = n_token
        self.n_down = 1
        decoder_config = decoder_config or {}
        self.to_mfcc = MFCC()
        self.init_cnn = ConvNorm(input_dim//2, hidden_dim, kernel_size=7, padding=3, stride=2)
        self.cnns = nn.Sequential(
            *[nn.Sequential(
                ConvBlock(hidden_dim),
                nn.GroupNorm(num_groups=1, num_channels=hidden_dim)
            ) for n in range(n_layers)])
        self.projection = ConvNorm(hidden_dim, hidden_dim // 2)
        self.ctc_linear = nn.Sequential(
            LinearNorm(hidden_dim//2, hidden_dim),
            nn.ReLU(),
            LinearNorm(hidden_dim, n_token))
        decoder_type = decoder_config.get('type', decoder_type)
        if decoder_type == 'lstm':
            lstm_config = decoder_config.get('lstm', {})
            lstm_kernel = lstm_config.get('location_kernel_size', location_kernel_size)
            random_mask = lstm_config.get('random_mask_prob', lstm_config.get('random_mask', 0.1))
            self.asr_s2s = ASRS2S(
                embedding_dim=token_embedding_dim,
                hidden_dim=hidden_dim//2,
                n_token=n_token,
                location_kernel_size=lstm_kernel,
                random_mask=random_mask)
        else:
            conformer_config = decoder_config.get('conformer', {})
            self.asr_s2s = ConformerS2S(
                embedding_dim=token_embedding_dim,
                hidden_dim=hidden_dim//2,
                n_token=n_token,
                num_layers=conformer_config.get('num_layers', 4),
                nhead=conformer_config.get('nhead', 4),
                ff_multiplier=conformer_config.get('ff_multiplier', 4),
                conv_kernel_size=conformer_config.get('conv_kernel_size', 31),
                dropout=conformer_config.get('dropout', 0.1),
                use_macaron=conformer_config.get('use_macaron', True),
                use_conv_module=conformer_config.get('use_conv_module', True),
                random_mask_prob=conformer_config.get('random_mask_prob', 0.1),
                pad_token_id=conformer_config.get('pad_token_id', 0))

    def forward(self, x, src_key_padding_mask=None, text_input=None):
        x = self.to_mfcc(x)
        x = self.init_cnn(x)
        x = self.cnns(x)

        x = self.projection(x)
        x = x.transpose(1, 2)
        ctc_logit = self.ctc_linear(x)
        if text_input is not None:
            _, s2s_logit, s2s_attn = self.asr_s2s(x, src_key_padding_mask, text_input)
            return ctc_logit, s2s_logit, s2s_attn
        else:
            return ctc_logit

    def get_feature(self, x):
        x = self.to_mfcc(x)
        x = self.init_cnn(x)
        x = self.cnns(x)
        x = self.instance_norm(x)
        x = self.projection(x)
        return x

    def length_to_mask(self, lengths):
        mask = (
            torch.arange(lengths.max(), device=lengths.device)
            .unsqueeze(0)
            .expand(lengths.shape[0], -1)
            .type_as(lengths)
        )
        mask = torch.gt(mask + 1, lengths.unsqueeze(1))
        return mask

    def get_future_mask(self, out_length, unmask_future_steps=0):
        """
        Args:
            out_length (int): returned mask shape is (out_length, out_length).
            unmask_futre_steps (int): unmasking future step size.
        Return:
            mask (torch.BoolTensor): mask future timesteps mask[i, j] = True if i > j + unmask_future_steps else False
        """
        index_tensor = torch.arange(out_length).unsqueeze(0).expand(out_length, -1)
        mask = torch.gt(index_tensor, index_tensor.T + unmask_future_steps)
        return mask

class ASRS2S(nn.Module):
    def __init__(self,
                 embedding_dim=256,
                 hidden_dim=512,
                 n_location_filters=32,
                 location_kernel_size=63,
                 n_token=40,
                 random_mask=0.1):
        super(ASRS2S, self).__init__()
        self.embedding = nn.Embedding(n_token, embedding_dim)
        val_range = math.sqrt(6 / hidden_dim)
        self.embedding.weight.data.uniform_(-val_range, val_range)

        self.decoder_rnn_dim = hidden_dim
        self.project_to_n_symbols = nn.Linear(self.decoder_rnn_dim, n_token)
        self.attention_layer = Attention(
            self.decoder_rnn_dim,
            hidden_dim,
            hidden_dim,
            n_location_filters,
            location_kernel_size
        )
        self.decoder_rnn = nn.LSTMCell(self.decoder_rnn_dim + embedding_dim, self.decoder_rnn_dim)
        self.project_to_hidden = nn.Sequential(
            LinearNorm(self.decoder_rnn_dim * 2, hidden_dim),
            nn.Tanh())
        self.sos = 1
        self.eos = 2
        self.random_mask = random_mask
        self.unk_index = 3

    def initialize_decoder_states(self, memory, mask):
        """
        moemory.shape = (B, L, H) = (Batchsize, Maxtimestep, Hiddendim)
        """
        B, L, H = memory.shape
        self.decoder_hidden = torch.zeros((B, self.decoder_rnn_dim)).type_as(memory)
        self.decoder_cell = torch.zeros((B, self.decoder_rnn_dim)).type_as(memory)
        self.attention_weights = torch.zeros((B, L)).type_as(memory)
        self.attention_weights_cum = torch.zeros((B, L)).type_as(memory)
        self.attention_context = torch.zeros((B, H)).type_as(memory)
        self.memory = memory
        self.processed_memory = self.attention_layer.memory_layer(memory)
        self.mask = mask

    def forward(self, memory, memory_mask, text_input):
        """
        moemory.shape = (B, L, H) = (Batchsize, Maxtimestep, Hiddendim)
        moemory_mask.shape = (B, L, )
        texts_input.shape = (B, T)
        """
        self.initialize_decoder_states(memory, memory_mask)
        # text random mask
        if self.training and self.random_mask > 0:
            random_mask = (torch.rand(text_input.shape, device=text_input.device) < self.random_mask)
            _text_input = text_input.clone()
            _text_input.masked_fill_(random_mask, self.unk_index)
        else:
            _text_input = text_input
        decoder_inputs = self.embedding(_text_input).transpose(0, 1) # -> [T, B, channel]
        start_embedding = self.embedding(
            torch.LongTensor([self.sos]*decoder_inputs.size(1)).to(decoder_inputs.device))
        decoder_inputs = torch.cat((start_embedding.unsqueeze(0), decoder_inputs), dim=0)

        hidden_outputs, logit_outputs, alignments = [], [], []
        while len(hidden_outputs) < decoder_inputs.size(0):

            decoder_input = decoder_inputs[len(hidden_outputs)]
            hidden, logit, attention_weights = self.decode(decoder_input)
            hidden_outputs += [hidden]
            logit_outputs += [logit]
            alignments += [attention_weights]

        hidden_outputs, logit_outputs, alignments = \
            self.parse_decoder_outputs(
                hidden_outputs, logit_outputs, alignments)

        return hidden_outputs, logit_outputs, alignments


    def decode(self, decoder_input):

        cell_input = torch.cat((decoder_input, self.attention_context), -1)
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            cell_input,
            (self.decoder_hidden, self.decoder_cell))

        attention_weights_cat = torch.cat(
            (self.attention_weights.unsqueeze(1),
            self.attention_weights_cum.unsqueeze(1)),dim=1)

        self.attention_context, self.attention_weights = self.attention_layer(
            self.decoder_hidden,
            self.memory,
            self.processed_memory,
            attention_weights_cat,
            self.mask)

        self.attention_weights_cum += self.attention_weights

        hidden_and_context = torch.cat((self.decoder_hidden, self.attention_context), -1)
        hidden = self.project_to_hidden(hidden_and_context)

        # dropout to increasing g
        logit = self.project_to_n_symbols(F.dropout(hidden, 0.5, self.training))

        return hidden, logit, self.attention_weights

    def parse_decoder_outputs(self, hidden, logit, alignments):

        # -> [B, T_out + 1, max_time]
        alignments = torch.stack(alignments).transpose(0,1)
        # [T_out + 1, B, n_symbols] -> [B, T_out + 1,  n_symbols]
        logit = torch.stack(logit).transpose(0, 1).contiguous()
        hidden = torch.stack(hidden).transpose(0, 1).contiguous()

        return hidden, logit, alignments


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(0)
        return x + self.pe[:length]


class FeedForwardModule(nn.Module):
    def __init__(self, d_model: int, multiplier: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden_dim = d_model * multiplier
        self.linear1 = nn.Linear(d_model, hidden_dim)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x


class ConformerConvModule(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=padding,
            groups=d_model,
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (T, B, D)
        x = x.transpose(0, 1).transpose(1, 2)  # (B, D, T)
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        x = x.transpose(1, 2).transpose(0, 1)  # (T, B, D)
        return x


class ConformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        ff_multiplier: int = 4,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
        use_macaron: bool = True,
        use_conv_module: bool = True,
    ):
        super().__init__()
        self.use_macaron = use_macaron
        self.use_conv_module = use_conv_module
        self.nhead = nhead

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.ffn1 = FeedForwardModule(d_model, ff_multiplier, dropout)
        self.ffn2 = FeedForwardModule(d_model, ff_multiplier, dropout)
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.cross_attn_norm = nn.LayerNorm(d_model)
        self.ffn1_norm = nn.LayerNorm(d_model)
        self.ffn2_norm = nn.LayerNorm(d_model)
        self.conv_norm = nn.LayerNorm(d_model)
        self.final_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        if use_conv_module:
            self.conv_module = ConformerConvModule(d_model, conv_kernel_size, dropout)
        else:
            self.conv_module = None

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = tgt
        if self.use_macaron:
            x = x + 0.5 * self.ffn1(self.ffn1_norm(x))

        residual = x
        q = self.self_attn_norm(x)
        attn_output, _ = self.self_attn(
            q, q, q,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False,
        )
        x = residual + self.dropout(attn_output)

        residual = x
        q = self.cross_attn_norm(x)
        attn_output, attn_weights = self.cross_attn(
            q,
            memory,
            memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        x = residual + self.dropout(attn_output)

        if self.use_conv_module and self.conv_module is not None:
            residual = x
            x = residual + self.conv_module(self.conv_norm(x))

        x = x + 0.5 * self.ffn2(self.ffn2_norm(x))
        x = self.final_norm(x)

        if attn_weights is not None:
            attn_weights = attn_weights.contiguous()
            target_len = attn_weights.size(-2)
            source_len = attn_weights.size(-1)
            batch_size = memory.size(1)
            attn_weights = attn_weights.view(batch_size, self.nhead, target_len, source_len)
            attn_weights = attn_weights.mean(dim=1)

        return x, attn_weights


class ConformerS2S(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        n_token: int,
        num_layers: int = 4,
        nhead: int = 4,
        ff_multiplier: int = 4,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
        use_macaron: bool = True,
        use_conv_module: bool = True,
        random_mask_prob: float = 0.1,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(n_token, embedding_dim)
        val_range = math.sqrt(6 / hidden_dim)
        self.embedding.weight.data.uniform_(-val_range, val_range)

        self.d_model = hidden_dim
        self.project_input = nn.Linear(embedding_dim, hidden_dim)
        self.layers = nn.ModuleList([
            ConformerDecoderLayer(
                hidden_dim,
                nhead,
                ff_multiplier=ff_multiplier,
                conv_kernel_size=conv_kernel_size,
                dropout=dropout,
                use_macaron=use_macaron,
                use_conv_module=use_conv_module,
            )
            for _ in range(num_layers)
        ])
        self.positional_encoding = PositionalEncoding(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(hidden_dim, n_token)
        self.random_mask_prob = random_mask_prob
        self.pad_token_id = pad_token_id
        self.sos = 1
        self.eos = 2
        self.unk_index = 3

    def _generate_subsequent_mask(self, size: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(size, size, device=device, dtype=torch.bool), diagonal=1)
        return mask

    def forward(self, memory: torch.Tensor, memory_mask: Optional[torch.Tensor], text_input: torch.Tensor):
        B, L, H = memory.shape
        device = memory.device

        if self.training and self.random_mask_prob > 0:
            random_mask = torch.rand(text_input.shape, device=device) < self.random_mask_prob
            _text_input = text_input.clone()
            _text_input.masked_fill_(random_mask, self.unk_index)
        else:
            _text_input = text_input

        decoder_inputs = self.embedding(_text_input)
        start_tokens = torch.full((B, 1), self.sos, dtype=text_input.dtype, device=device)
        start_embedding = self.embedding(start_tokens)
        decoder_inputs = torch.cat((start_embedding, decoder_inputs), dim=1)

        decoder_inputs = self.project_input(decoder_inputs)
        decoder_inputs = decoder_inputs.transpose(0, 1)  # (T, B, D)
        decoder_inputs = self.positional_encoding(decoder_inputs)
        decoder_inputs = self.dropout(decoder_inputs)

        tgt_mask = self._generate_subsequent_mask(decoder_inputs.size(0), device)
        if memory_mask is not None:
            memory_key_padding_mask = memory_mask
        else:
            memory_key_padding_mask = None

        tgt_key_padding_mask = None
        if self.pad_token_id is not None:
            pad_mask_tokens = torch.cat((torch.zeros((B, 1), device=device, dtype=torch.bool), text_input.eq(self.pad_token_id)), dim=1)
            tgt_key_padding_mask = pad_mask_tokens

        memory_transposed = memory.transpose(0, 1)  # (L, B, H)
        x = decoder_inputs
        attn = None
        for layer in self.layers:
            x, attn_weights = layer(
                x,
                memory_transposed,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
            if attn_weights is not None:
                attn = attn_weights

        hidden = x.transpose(0, 1)
        logits = self.output_projection(hidden)

        if attn is None:
            attn = torch.zeros(
                hidden.size(0),
                hidden.size(1),
                memory.size(1),
                device=hidden.device,
            )

        return hidden, logits, attn
