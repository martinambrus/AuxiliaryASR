import math
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _generate_square_subsequent_mask(sz: int, device: torch.device) -> Tensor:
    mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
    mask = mask.masked_fill(mask.bool(), float('-inf'))
    return mask


class RelativePositionBias(nn.Module):
    """T5-style relative position bias."""

    def __init__(self, num_heads: int, max_distance: int = 512) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Parameter(
            torch.zeros(num_heads, 2 * max_distance - 1)
        )
        nn.init.xavier_uniform_(self.relative_attention_bias)

    def forward(self, qlen: int, klen: int) -> Tensor:
        device = self.relative_attention_bias.device
        context_position = torch.arange(qlen, device=device)[:, None]
        memory_position = torch.arange(klen, device=device)[None, :]
        relative_position = memory_position - context_position
        relative_position = relative_position.clamp(
            -self.max_distance + 1, self.max_distance - 1
        )
        relative_position = relative_position + self.max_distance - 1
        values = self.relative_attention_bias[:, relative_position]
        # (num_heads, qlen, klen)
        return values


class FeedForwardModule(nn.Module):
    def __init__(self, d_model: int, dim_feedforward: int, dropout: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear1(x)
        x = F.silu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class RelativeTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        max_position: int,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.ffn = FeedForwardModule(d_model, dim_feedforward, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.self_bias = RelativePositionBias(nhead, max_distance=max_position)
        self.cross_bias = RelativePositionBias(nhead, max_distance=max_position)

    def _combine_bias(self, base_mask: Optional[Tensor], bias: Optional[Tensor]) -> Optional[Tensor]:
        if base_mask is None and bias is None:
            return base_mask
        if base_mask is None:
            mask = bias.mean(dim=0)
        elif bias is None:
            mask = base_mask
        else:
            mask = base_mask + bias.mean(dim=0)
        return mask

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor],
        tgt_key_padding_mask: Optional[Tensor],
        memory_key_padding_mask: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        qlen = tgt.size(1)
        mlen = memory.size(1)
        device = tgt.device
        rel_self = self.self_bias(qlen, qlen)
        rel_cross = self.cross_bias(qlen, mlen)

        combined_self_mask = self._combine_bias(tgt_mask, rel_self)
        combined_cross_mask = self._combine_bias(None, rel_cross)

        residual = tgt
        tgt_norm = self.norm1(tgt)
        attn_output, self_attn_weights = self.self_attn(
            tgt_norm,
            tgt_norm,
            tgt_norm,
            attn_mask=combined_self_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        tgt = residual + self.dropout(attn_output)

        residual = tgt
        tgt_norm = self.norm2(tgt)
        attn_output, cross_attn_weights = self.cross_attn(
            tgt_norm,
            memory,
            memory,
            attn_mask=combined_cross_mask,
            key_padding_mask=memory_key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        tgt = residual + self.dropout(attn_output)

        residual = tgt
        tgt = residual + self.dropout(self.ffn(self.norm3(tgt)))
        return tgt, cross_attn_weights.mean(dim=1)


class RelativeTransformerDecoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        n_token: int,
        num_layers: int = 4,
        nhead: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_position: int = 512,
        random_mask: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_token, embedding_dim)
        nn.init.uniform_(
            self.embedding.weight, -math.sqrt(6.0 / hidden_dim), math.sqrt(6.0 / hidden_dim)
        )
        self.layers = nn.ModuleList(
            [
                RelativeTransformerDecoderLayer(
                    hidden_dim,
                    nhead,
                    dim_feedforward,
                    dropout,
                    max_position,
                )
                for _ in range(num_layers)
            ]
        )
        self.input_proj = nn.Linear(embedding_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, n_token)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.random_mask = random_mask
        self.sos = 1
        self.eos = 2

    def forward(
        self,
        memory: Tensor,
        memory_mask: Optional[Tensor],
        text_input: Tensor,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        if text_input is None:
            raise ValueError("text_input must be provided for transformer decoding")
        if self.training and self.random_mask > 0:
            random_mask = (torch.rand_like(text_input.float()) < self.random_mask)
            text_input = text_input.masked_fill(random_mask, 3)

        start_symbol = torch.full(
            (text_input.size(0), 1), self.sos, dtype=text_input.dtype, device=text_input.device
        )
        decoder_inputs = torch.cat([start_symbol, text_input], dim=1)
        tgt_key_padding_mask = decoder_inputs.eq(0)
        embedded = self.embedding(decoder_inputs)
        tgt = self.input_proj(embedded)

        seq_len = tgt.size(1)
        tgt_mask = _generate_square_subsequent_mask(seq_len, tgt.device)

        attn_list: List[Tensor] = []
        for layer in self.layers:
            tgt, attn = layer(
                tgt,
                memory,
                tgt_mask,
                tgt_key_padding_mask,
                memory_mask,
            )
            attn_list.append(attn)

        logits = self.output_proj(self.output_norm(tgt))
        alignments = torch.stack(attn_list, dim=1).mean(dim=1) if attn_list else None
        return tgt, logits, alignments


class ConvolutionModule(nn.Module):
    def __init__(self, d_model: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        padding = (kernel_size - 1) // 2
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

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.layer_norm(x)
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)
        return residual + x


class ConformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        kernel_size: int,
    ) -> None:
        super().__init__()
        self.ffn1 = FeedForwardModule(d_model, dim_feedforward, dropout)
        self.ffn2 = FeedForwardModule(d_model, dim_feedforward, dropout)
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.conv_module = ConvolutionModule(d_model, kernel_size, dropout)
        self.norm_ffn1 = nn.LayerNorm(d_model)
        self.norm_ffn2 = nn.LayerNorm(d_model)
        self.norm_self_attn = nn.LayerNorm(d_model)
        self.norm_cross_attn = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor],
        tgt_key_padding_mask: Optional[Tensor],
        memory_key_padding_mask: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        residual = tgt
        tgt = residual + 0.5 * self.dropout(self.ffn1(self.norm_ffn1(tgt)))

        residual = tgt
        normed = self.norm_self_attn(tgt)
        self_attn_out, _ = self.self_attn(
            normed,
            normed,
            normed,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False,
        )
        tgt = residual + self.dropout(self_attn_out)

        residual = tgt
        normed = self.norm_cross_attn(tgt)
        cross_attn_out, cross_attn_weights = self.cross_attn(
            normed,
            memory,
            memory,
            attn_mask=None,
            key_padding_mask=memory_key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        tgt = residual + self.dropout(cross_attn_out)

        tgt = self.conv_module(tgt)

        residual = tgt
        tgt = residual + 0.5 * self.dropout(self.ffn2(self.norm_ffn2(tgt)))
        tgt = self.final_norm(tgt)
        return tgt, cross_attn_weights.mean(dim=1)


class ConformerDecoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        n_token: int,
        num_layers: int = 4,
        nhead: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        kernel_size: int = 15,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_token, embedding_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=hidden_dim ** -0.5)
        self.input_proj = nn.Linear(embedding_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [
                ConformerDecoderLayer(
                    hidden_dim,
                    nhead,
                    dim_feedforward,
                    dropout,
                    kernel_size,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, n_token)
        self.dropout = nn.Dropout(dropout)
        self.sos = 1
        self.random_mask = 0.1

    def forward(
        self,
        memory: Tensor,
        memory_mask: Optional[Tensor],
        text_input: Tensor,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        if text_input is None:
            raise ValueError("text_input must be provided for conformer decoding")
        if self.training and self.random_mask > 0:
            random_mask = (torch.rand_like(text_input.float()) < self.random_mask)
            text_input = text_input.masked_fill(random_mask, 3)
        start_symbol = torch.full(
            (text_input.size(0), 1), self.sos, dtype=text_input.dtype, device=text_input.device
        )
        decoder_inputs = torch.cat([start_symbol, text_input], dim=1)
        tgt_key_padding_mask = decoder_inputs.eq(0)
        embedded = self.dropout(self.embedding(decoder_inputs))
        tgt = self.input_proj(embedded)

        seq_len = tgt.size(1)
        tgt_mask = _generate_square_subsequent_mask(seq_len, tgt.device)

        attn_list: List[Tensor] = []
        for layer in self.layers:
            tgt, attn = layer(
                tgt,
                memory,
                tgt_mask,
                tgt_key_padding_mask,
                memory_mask,
            )
            attn_list.append(attn)

        tgt = self.output_norm(tgt)
        logits = self.output_proj(tgt)
        alignments = torch.stack(attn_list, dim=1).mean(dim=1) if attn_list else None
        return tgt, logits, alignments


class RNNTDecoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        n_token: int,
        joint_dim: int = 256,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_token, embedding_dim)
        self.prediction_network = nn.LSTM(
            embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True
        )
        self.encoder_proj = nn.Linear(hidden_dim, joint_dim)
        self.decoder_proj = nn.Linear(hidden_dim, joint_dim)
        self.output_proj = nn.Linear(joint_dim, n_token)
        self.hidden_dim = hidden_dim
        self.sos = 1

    def forward(
        self,
        memory: Tensor,
        memory_mask: Optional[Tensor],
        text_input: Tensor,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        if text_input is None:
            raise ValueError("text_input must be provided for RNN-T decoding")
        start_symbol = torch.full(
            (text_input.size(0), 1), self.sos, dtype=text_input.dtype, device=text_input.device
        )
        decoder_inputs = torch.cat([start_symbol, text_input], dim=1)
        embedded = self.embedding(decoder_inputs)
        decoder_outputs, _ = self.prediction_network(embedded)

        enc = self.encoder_proj(memory)
        dec = self.decoder_proj(decoder_outputs)
        joint = torch.tanh(enc.unsqueeze(2) + dec.unsqueeze(1))
        logits = self.output_proj(joint)
        return decoder_outputs, logits, None


class MonotonicAttention(nn.Module):
    def __init__(self, query_dim: int, memory_dim: int, chunk_size: int = 4) -> None:
        super().__init__()
        self.query_layer = nn.Linear(query_dim, memory_dim)
        self.memory_layer = nn.Linear(memory_dim, memory_dim)
        self.v = nn.Linear(memory_dim, 1)
        self.chunk_size = chunk_size

    def forward(
        self,
        query: Tensor,
        memory: Tensor,
        mask: Optional[Tensor],
        current_position: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        processed_query = self.query_layer(query).unsqueeze(1)
        processed_memory = self.memory_layer(memory)
        scores = self.v(torch.tanh(processed_query + processed_memory)).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        attn = []
        new_positions = []
        for b in range(scores.size(0)):
            start = current_position[b].item()
            end = start + self.chunk_size if self.chunk_size > 0 else scores.size(1)
            end = min(end, scores.size(1))
            allowed = scores[b, start:end]
            weight = F.softmax(allowed, dim=-1)
            full = torch.zeros_like(scores[b])
            full[start:end] = weight
            attn.append(full)
            max_pos = full.argmax(dim=-1)
            new_positions.append(max_pos)
        attn_tensor = torch.stack(attn, dim=0)
        new_pos_tensor = torch.stack(new_positions, dim=0)
        context = torch.bmm(attn_tensor.unsqueeze(1), memory).squeeze(1)
        return context, attn_tensor, new_pos_tensor


class MonotonicDecoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        n_token: int,
        chunk_size: int = 4,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_token, embedding_dim)
        self.decoder_rnn = nn.LSTMCell(embedding_dim + hidden_dim, hidden_dim)
        self.project_to_hidden = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.Tanh()
        )
        self.project_to_n_symbols = nn.Linear(hidden_dim, n_token)
        self.attention = MonotonicAttention(hidden_dim, hidden_dim, chunk_size)
        self.sos = 1
        self.random_mask = 0.1

    def initialize(self, memory: Tensor, memory_mask: Optional[Tensor]) -> None:
        B, T, H = memory.shape
        device = memory.device
        self.decoder_hidden = torch.zeros((B, H), device=device)
        self.decoder_cell = torch.zeros((B, H), device=device)
        self.attention_context = torch.zeros((B, H), device=device)
        self.memory = memory
        self.memory_mask = memory_mask
        self.current_position = torch.zeros(B, dtype=torch.long, device=device)

    def decode_step(self, decoder_input: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        cell_input = torch.cat([decoder_input, self.attention_context], dim=-1)
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            cell_input, (self.decoder_hidden, self.decoder_cell)
        )
        context, attn, new_position = self.attention(
            self.decoder_hidden, self.memory, self.memory_mask, self.current_position
        )
        self.attention_context = context
        self.current_position = torch.max(self.current_position, new_position)
        hidden_and_context = torch.cat([self.decoder_hidden, self.attention_context], dim=-1)
        hidden = self.project_to_hidden(hidden_and_context)
        logit = self.project_to_n_symbols(F.dropout(hidden, 0.5, self.training))
        return hidden, logit, attn

    def forward(
        self,
        memory: Tensor,
        memory_mask: Optional[Tensor],
        text_input: Tensor,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        if text_input is None:
            raise ValueError("text_input must be provided for monotonic decoding")
        self.initialize(memory, memory_mask)
        if self.training and self.random_mask > 0:
            random_mask = (torch.rand_like(text_input.float()) < self.random_mask)
            text_input = text_input.masked_fill(random_mask, 3)
        start_symbol = torch.full(
            (text_input.size(0), 1), self.sos, dtype=text_input.dtype, device=text_input.device
        )
        inputs = torch.cat([start_symbol, text_input], dim=1)
        embeddings = self.embedding(inputs)

        hidden_outputs: List[Tensor] = []
        logit_outputs: List[Tensor] = []
        alignments: List[Tensor] = []
        for t in range(embeddings.size(1)):
            hidden, logit, attn = self.decode_step(embeddings[:, t])
            hidden_outputs.append(hidden)
            logit_outputs.append(logit)
            alignments.append(attn)
        hidden_tensor = torch.stack(hidden_outputs, dim=1)
        logit_tensor = torch.stack(logit_outputs, dim=1)
        alignment_tensor = torch.stack(alignments, dim=1)
        return hidden_tensor, logit_tensor, alignment_tensor


class CIFDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        n_token: int,
        output_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim or hidden_dim
        self.alpha_projection = nn.Linear(hidden_dim, 1)
        self.output_projection = nn.Linear(hidden_dim, n_token)

    def forward(
        self,
        memory: Tensor,
        memory_mask: Optional[Tensor],
        text_input: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        alphas = torch.sigmoid(self.alpha_projection(memory)).squeeze(-1)
        batch_outputs: List[Tensor] = []
        lengths: List[int] = []
        device = memory.device
        for b in range(memory.size(0)):
            integrate = torch.tensor(0.0, device=device)
            residue = torch.zeros(self.hidden_dim, device=device)
            emitted: List[Tensor] = []
            for t in range(memory.size(1)):
                alpha = alphas[b, t]
                integrate = integrate + alpha
                residue = residue + memory[b, t] * alpha
                while integrate.item() >= 1.0:
                    fired = residue / integrate.clamp(min=1e-6)
                    emitted.append(fired)
                    integrate = integrate - 1.0
                    residue = residue - fired
            if text_input is not None:
                target_len = text_input.size(1)
            else:
                target_len = max(len(emitted), 1)
            output = torch.zeros(target_len, self.hidden_dim, device=device)
            for i in range(min(target_len, len(emitted))):
                output[i] = emitted[i]
            if len(emitted) < target_len and integrate.item() > 0:
                output[len(emitted)] = residue / integrate.clamp(min=1e-6)
            batch_outputs.append(output)
            lengths.append(target_len)
        max_len = max(lengths)
        padded_outputs = torch.zeros(
            memory.size(0), max_len, self.hidden_dim, device=device
        )
        for b, output in enumerate(batch_outputs):
            padded_outputs[b, : output.size(0)] = output
        logits = self.output_projection(padded_outputs)
        return padded_outputs, logits, None


def build_decoder(
    config: Optional[Dict[str, Any]],
    encoder_dim: int,
    n_token: int,
    embedding_dim: int,
) -> nn.Module:
    config = config or {}
    decoder_type = config.get("type", "transformer_relative")

    def _ensure_enabled(section: Dict[str, Any]) -> None:
        if not section.get("enabled", True):
            raise ValueError(
                f"Decoder '{decoder_type}' is disabled in the configuration."
            )

    if decoder_type == "transformer_relative":
        section = config.get("transformer_relative", {})
        _ensure_enabled(section)
        return RelativeTransformerDecoder(
            embedding_dim=embedding_dim,
            hidden_dim=section.get("hidden_dim", encoder_dim),
            n_token=n_token,
            num_layers=section.get("num_layers", 4),
            nhead=section.get("nhead", 4),
            dim_feedforward=section.get("dim_feedforward", 1024),
            dropout=section.get("dropout", 0.1),
            max_position=section.get("max_position", 512),
            random_mask=section.get("random_mask", 0.1),
        )
    if decoder_type == "conformer":
        section = config.get("conformer", {})
        _ensure_enabled(section)
        return ConformerDecoder(
            embedding_dim=embedding_dim,
            hidden_dim=section.get("hidden_dim", encoder_dim),
            n_token=n_token,
            num_layers=section.get("num_layers", 4),
            nhead=section.get("nhead", 4),
            dim_feedforward=section.get("dim_feedforward", 1024),
            dropout=section.get("dropout", 0.1),
            kernel_size=section.get("kernel_size", 15),
        )
    if decoder_type == "rnnt":
        section = config.get("rnnt", {})
        _ensure_enabled(section)
        return RNNTDecoder(
            embedding_dim=section.get("embedding_dim", embedding_dim),
            hidden_dim=section.get("hidden_dim", encoder_dim),
            n_token=n_token,
            joint_dim=section.get("joint_dim", encoder_dim),
            num_layers=section.get("num_layers", 1),
        )
    if decoder_type == "mocha":
        section = config.get("mocha", {})
        _ensure_enabled(section)
        return MonotonicDecoder(
            embedding_dim=section.get("embedding_dim", embedding_dim),
            hidden_dim=section.get("hidden_dim", encoder_dim),
            n_token=n_token,
            chunk_size=section.get("chunk_size", 4),
        )
    if decoder_type == "cif":
        section = config.get("cif", {})
        _ensure_enabled(section)
        return CIFDecoder(
            hidden_dim=section.get("hidden_dim", encoder_dim),
            n_token=n_token,
            output_dim=section.get("output_dim", encoder_dim),
        )
    raise ValueError(f"Unsupported decoder type: {decoder_type}")
