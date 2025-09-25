import math
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from layers import MFCC, Attention, LinearNorm, ConvNorm, ConvBlock

def build_model(model_params={}, model_type='asr'):
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
                 decoder_type: str = "transformer",
                 decoder_params: Optional[dict] = None,
    ):
        super().__init__()
        self.n_token = n_token
        self.n_down = 1
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
        decoder_params = decoder_params or {}
        decoder_type = decoder_type.lower()
        if isinstance(decoder_params, dict) and decoder_type in decoder_params and isinstance(decoder_params[decoder_type], dict):
            selected_decoder_params = decoder_params[decoder_type]
        else:
            selected_decoder_params = decoder_params
        if decoder_type == "transformer":
            self.asr_s2s = TransformerASRS2S(
                embedding_dim=token_embedding_dim,
                hidden_dim=hidden_dim//2,
                n_token=n_token,
                decoder_params=selected_decoder_params,
            )
        elif decoder_type == "lstm":
            self.asr_s2s = ASRS2S(
                embedding_dim=token_embedding_dim,
                hidden_dim=hidden_dim//2,
                n_token=n_token,
                location_kernel_size=location_kernel_size,
                decoder_params=selected_decoder_params,
            )
        else:
            raise ValueError(f"Unsupported decoder type: {decoder_type}")
        self.decoder_type = decoder_type

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
                 decoder_params: Optional[dict] = None):
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
        decoder_params = decoder_params or {}
        self.random_mask = float(decoder_params.get("random_mask", 0.1))
        self.unk_index = int(decoder_params.get("unk_index", 3))

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


class RelativePositionBias(nn.Module):
    def __init__(self, num_heads: int, max_distance: int = 200):
        super().__init__()
        self.num_heads = num_heads
        self.max_distance = max_distance
        self.bias = nn.Embedding(2 * max_distance + 1, num_heads)

    def forward(self, q_len: int, k_len: int, device: torch.device) -> torch.Tensor:
        context_position = torch.arange(q_len, device=device)[:, None]
        memory_position = torch.arange(k_len, device=device)[None, :]
        relative_position = memory_position - context_position
        relative_position = relative_position.clamp(-self.max_distance, self.max_distance) + self.max_distance
        values = self.bias(relative_position)
        return values.permute(2, 0, 1)  # [n_heads, q_len, k_len]


class RelativeMultiHeadSelfAttention(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.1,
                 use_relative_bias: bool = True,
                 max_relative_distance: int = 200):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.use_relative_bias = use_relative_bias
        if use_relative_bias:
            self.relative_bias = RelativePositionBias(num_heads, max_distance=max_relative_distance)
        else:
            self.relative_bias = None
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self,
                x: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, tgt_len, _ = x.size()
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)

        q = q * self.scaling
        attn_weights = torch.matmul(q, k.transpose(-2, -1))  # [B, num_heads, tgt, tgt]

        if self.use_relative_bias and self.relative_bias is not None:
            rel = self.relative_bias(tgt_len, tgt_len, x.device)
            attn_weights = attn_weights + rel.unsqueeze(0)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_weights = attn_weights.masked_fill(attn_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            else:
                attn_weights = attn_weights + attn_mask.unsqueeze(0).unsqueeze(0)

        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights


class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 d_model: int,
                 nhead: int,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 activation: str = "relu",
                 use_relative_positional_bias: bool = True,
                 max_relative_distance: int = 200):
        super().__init__()
        self.self_attn = RelativeMultiHeadSelfAttention(
            d_model,
            nhead,
            dropout=dropout,
            use_relative_bias=use_relative_positional_bias,
            max_relative_distance=max_relative_distance,
        )
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self,
                tgt: torch.Tensor,
                memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x, _ = self.self_attn(tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(x)
        tgt = self.norm1(tgt)

        if memory is not None:
            attn_output, attn_weights = self.multihead_attn(
                tgt,
                memory,
                memory,
                attn_mask=None,
                key_padding_mask=memory_key_padding_mask,
                need_weights=True,
                average_attn_weights=True,
            )
            tgt = tgt + self.dropout2(attn_output)
            tgt = self.norm2(tgt)
        else:
            attn_weights = None

        x = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(x)
        tgt = self.norm3(tgt)
        return tgt, attn_weights


class TransformerDecoder(nn.Module):
    def __init__(self, layer: TransformerDecoderLayer, num_layers: int, norm: Optional[nn.Module] = None):
        super().__init__()
        self.layers = nn.ModuleList([layer if i == 0 else self._clone_layer(layer) for i in range(num_layers)])
        self.norm = norm

    @staticmethod
    def _clone_layer(layer: TransformerDecoderLayer) -> TransformerDecoderLayer:
        import copy

        return copy.deepcopy(layer)

    def forward(self,
                tgt: torch.Tensor,
                memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        output = tgt
        attn_weights = None

        for mod in self.layers:
            output, layer_attn = mod(output, memory, tgt_mask, tgt_key_padding_mask, memory_key_padding_mask)
            if layer_attn is not None:
                attn_weights = layer_attn

        if self.norm is not None:
            output = self.norm(output)

        return output, attn_weights


class TransformerASRS2S(nn.Module):
    def __init__(self,
                 embedding_dim: int = 256,
                 hidden_dim: int = 512,
                 n_token: int = 40,
                 decoder_params: Optional[dict] = None):
        super().__init__()
        decoder_params = decoder_params or {}
        self.embedding = nn.Embedding(n_token, embedding_dim)
        val_range = math.sqrt(6 / hidden_dim)
        self.embedding.weight.data.uniform_(-val_range, val_range)
        self.hidden_dim = hidden_dim
        self.embedding_scale = math.sqrt(hidden_dim)
        self.random_mask = float(decoder_params.get("random_mask", 0.0))
        self.unk_index = int(decoder_params.get("unk_index", 3))

        self.embedding_projection = nn.Linear(embedding_dim, hidden_dim) if embedding_dim != hidden_dim else nn.Identity()
        num_layers = int(decoder_params.get("num_layers", 6))
        num_heads = int(decoder_params.get("num_heads", 8))
        ffn_dim = int(decoder_params.get("ffn_dim", hidden_dim * 4))
        dropout = float(decoder_params.get("dropout", 0.1))
        activation = decoder_params.get("activation", "relu")
        use_relative_bias = bool(decoder_params.get("use_relative_positional_bias", True))
        max_relative_distance = int(decoder_params.get("max_relative_distance", 200))

        decoder_layer = TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation=activation,
            use_relative_positional_bias=use_relative_bias,
            max_relative_distance=max_relative_distance,
        )
        self.decoder = TransformerDecoder(decoder_layer, num_layers, norm=nn.LayerNorm(hidden_dim))
        self.output_projection = nn.Linear(hidden_dim, n_token)
        self.dropout = nn.Dropout(dropout)
        self.sos = 1
        self.eos = 2

    def forward(self, memory: torch.Tensor, memory_mask: Optional[torch.Tensor], text_input: torch.Tensor):
        if self.training and self.random_mask > 0:
            random_mask = (torch.rand(text_input.shape, device=text_input.device) < self.random_mask)
            decoder_tokens = text_input.clone()
            decoder_tokens.masked_fill_(random_mask, self.unk_index)
        else:
            decoder_tokens = text_input

        start_tokens = torch.full((decoder_tokens.size(0), 1), self.sos, dtype=torch.long, device=decoder_tokens.device)
        decoder_tokens = torch.cat([start_tokens, decoder_tokens], dim=1)
        padding_mask = torch.cat([
            torch.zeros((decoder_tokens.size(0), 1), dtype=torch.bool, device=decoder_tokens.device),
            decoder_tokens[:, 1:] == 0,
        ], dim=1)

        embeddings = self.embedding(decoder_tokens)
        embeddings = self.embedding_projection(embeddings)
        embeddings = embeddings * self.embedding_scale
        embeddings = self.dropout(embeddings)

        tgt_len = embeddings.size(1)
        tgt_mask = torch.triu(torch.ones((tgt_len, tgt_len), device=embeddings.device, dtype=torch.bool), diagonal=1)

        decoder_output, attn_weights = self.decoder(
            embeddings,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=padding_mask,
            memory_key_padding_mask=memory_mask,
        )

        logits = self.output_projection(decoder_output)
        return decoder_output, logits, attn_weights
