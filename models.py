import torch
from torch import nn

from layers import MFCC, LinearNorm, ConvNorm, ConvBlock
from decoders import build_decoder


def build_model(model_params=None, model_type='asr'):
    model_params = model_params or {}
    model = ASRCNN(**model_params)
    return model


class ASRCNN(nn.Module):
    def __init__(
        self,
        input_dim=80,
        hidden_dim=256,
        n_token=35,
        n_layers=6,
        token_embedding_dim=256,
        location_kernel_size=63,
        decoder_config=None,
    ):
        super().__init__()
        self.n_token = n_token
        self.n_down = 1
        self.to_mfcc = MFCC()
        self.init_cnn = ConvNorm(input_dim // 2, hidden_dim, kernel_size=7, padding=3, stride=2)
        self.cnns = nn.Sequential(
            *[
                nn.Sequential(
                    ConvBlock(hidden_dim),
                    nn.GroupNorm(num_groups=1, num_channels=hidden_dim)
                )
                for _ in range(n_layers)
            ]
        )
        self.projection = ConvNorm(hidden_dim, hidden_dim // 2)
        self.ctc_linear = nn.Sequential(
            LinearNorm(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            LinearNorm(hidden_dim, n_token)
        )
        self.decoder = build_decoder(
            decoder_config,
            hidden_dim // 2,
            n_token,
            token_embedding_dim,
        )

    def forward(self, x, src_key_padding_mask=None, text_input=None):
        x = self.to_mfcc(x)
        x = self.init_cnn(x)
        x = self.cnns(x)

        x = self.projection(x)
        x = x.transpose(1, 2)
        ctc_logit = self.ctc_linear(x)
        if text_input is not None:
            _, s2s_logit, s2s_attn = self.decoder(x, src_key_padding_mask, text_input)
            return ctc_logit, s2s_logit, s2s_attn
        else:
            return ctc_logit

    def get_feature(self, x):
        x = self.to_mfcc(x)
        x = self.init_cnn(x)
        x = self.cnns(x)
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
