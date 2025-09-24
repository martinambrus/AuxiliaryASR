import math
from typing import Optional, Tuple

import torch
from torch import nn
import torchaudio.functional as AF

try:
    from transformers import AutoModel
except ImportError as exc:  # pragma: no cover - handled via runtime error
    AutoModel = None
    _TRANSFORMERS_IMPORT_ERROR = exc
else:  # pragma: no cover - executed when dependency is available
    _TRANSFORMERS_IMPORT_ERROR = None


class SSLFrontendError(RuntimeError):
    """Raised when the self-supervised frontend cannot be constructed."""


class SSLFrontend(nn.Module):
    """Wrapper around Hugging Face SSL acoustic models.

    Parameters
    ----------
    architecture:
        Optional label describing the model family. Used only for informative
        error messages.
    pretrained_model_name:
        Hugging Face model identifier. Required when ``architecture`` alone
        is not sufficient to locate a checkpoint.
    fine_tune:
        Whether to update the encoder parameters during training. When set to
        ``False`` gradients are not computed through the frontend to keep the
        pretrained weights frozen.
    output_layer:
        Index of the hidden layer to expose as features.
    layer_norm:
        Apply :class:`torch.nn.LayerNorm` to the extracted features.
    feature_projection_dim:
        Optional dimensionality reduction applied through a learnable linear
        layer.
    target_sample_rate:
        Target sampling rate expected by the frontend. When ``None`` the value
        is inferred from the checkpoint configuration or defaults to 16000 Hz.
    """

    def __init__(
        self,
        architecture: Optional[str] = None,
        pretrained_model_name: Optional[str] = None,
        fine_tune: bool = False,
        output_layer: int = -1,
        layer_norm: bool = True,
        feature_projection_dim: Optional[int] = None,
        target_sample_rate: Optional[int] = None,
    ) -> None:
        super().__init__()

        if AutoModel is None:
            raise SSLFrontendError(
                "transformers is required to use SSL front-ends. Install it via"
                " `pip install transformers` and retry."  # pragma: no cover - informative message
            ) from _TRANSFORMERS_IMPORT_ERROR

        if not pretrained_model_name:
            raise SSLFrontendError(
                "A pretrained model identifier must be provided to construct"
                " the SSL frontend."
            )

        self.architecture = architecture or "unknown"
        self.model = AutoModel.from_pretrained(pretrained_model_name)
        self.output_layer = int(output_layer)
        self.fine_tune = bool(fine_tune)

        hidden_size = getattr(self.model.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(self.model.config, "encoder_embed_dim", None)
        if hidden_size is None:
            raise SSLFrontendError(
                f"Could not infer hidden size for frontend `{pretrained_model_name}`."
            )

        if target_sample_rate is None:
            target_sample_rate = getattr(
                self.model.config, "sample_rate",
                getattr(self.model.config, "sampling_rate", None)
            )
        if target_sample_rate is None:
            target_sample_rate = 16000
        self.target_sample_rate = int(target_sample_rate)

        projection_dim = int(feature_projection_dim) if feature_projection_dim else None
        if projection_dim is not None and projection_dim <= 0:
            projection_dim = None

        if projection_dim is not None:
            self.projection = nn.Linear(hidden_size, projection_dim)
            output_dim = projection_dim
        else:
            self.projection = None
            output_dim = hidden_size

        if layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)
        else:
            self.layer_norm = None

        if not self.fine_tune:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

        self.output_dim = output_dim

    def forward(
        self,
        waveforms: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        sampling_rate: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if waveforms.dim() == 1:
            waveforms = waveforms.unsqueeze(0)

        if sampling_rate is not None and sampling_rate != self.target_sample_rate:
            waveforms = AF.resample(waveforms, sampling_rate, self.target_sample_rate)
            if lengths is not None:
                lengths = (
                    lengths.to(torch.float32)
                    * self.target_sample_rate
                    / sampling_rate
                ).floor().to(torch.long)
                lengths = lengths.clamp(min=1)

        if lengths is not None:
            lengths = lengths.to(torch.long)
            lengths = lengths.clamp(min=1, max=waveforms.size(1))

        device = waveforms.device
        dtype = next(self.model.parameters()).dtype
        waveforms = waveforms.to(dtype=dtype)

        # Normalize to the range expected by wav2vec-style models.
        peak = waveforms.abs().amax(dim=1, keepdim=True).clamp_min(1e-5)
        waveforms = waveforms / peak

        attention_mask = None
        if lengths is not None:
            attention_mask = torch.arange(waveforms.size(1), device=device).unsqueeze(0)
            attention_mask = attention_mask.expand(waveforms.size(0), -1)
            attention_mask = attention_mask < lengths.unsqueeze(1)

        model_kwargs = {"attention_mask": attention_mask, "output_hidden_states": True}
        if not self.fine_tune:
            with torch.no_grad():
                outputs = self.model(waveforms, **model_kwargs)
        else:
            outputs = self.model(waveforms, **model_kwargs)

        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise SSLFrontendError(
                f"Frontend `{self.architecture}` did not return hidden states."
            )

        layer_index = self._resolve_layer_index(len(hidden_states))
        features = hidden_states[layer_index]

        if self.projection is not None:
            features = self.projection(features)
        if self.layer_norm is not None:
            features = self.layer_norm(features)

        features = features.transpose(1, 2).contiguous()
        max_feature_length = features.size(-1)

        if lengths is not None and hasattr(self.model, "_get_feat_extract_output_lengths"):
            cpu_lengths = lengths.detach().to(torch.long).cpu()
            encoder_lengths = self.model._get_feat_extract_output_lengths(cpu_lengths)
            if torch.is_tensor(encoder_lengths):
                encoder_lengths = encoder_lengths.to(device=device, dtype=torch.long)
            else:
                encoder_lengths = torch.tensor(encoder_lengths, device=device, dtype=torch.long)
        else:
            encoder_lengths = None

        if encoder_lengths is not None:
            if max_feature_length > 0:
                encoder_lengths = encoder_lengths.clamp(min=1, max=max_feature_length)
            else:
                encoder_lengths = torch.zeros_like(encoder_lengths)

        return features, encoder_lengths

    def _resolve_layer_index(self, num_layers: int) -> int:
        if self.output_layer < 0:
            layer_index = num_layers + self.output_layer
        else:
            layer_index = self.output_layer
        layer_index = max(0, min(num_layers - 1, layer_index))
        return layer_index
