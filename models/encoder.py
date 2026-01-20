"""
Inception-based Encoder for Time Series

Channel-wise Inception encoder with optional weight sharing.
"""

import sys
import os
sys.path.append('..')
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch
import torch.nn as nn
from layers.InceptionBlock import InceptionLayer, ResidualBlock


class ShapeletInceptionEncoder(nn.Module):
    """
    Channel-wise Inception encoder.
    
    Args:
        seq_len: Input sequence length T
        in_channels: Number of channels M
        num_layers: Number of Inception block repetitions
        kernel_size: Largest kernel size (InceptionTime style)
        nb_filters: Filters per branch
        num_kernels: Number of Conv branches
        use_bottleneck: Whether to use bottleneck
        bottleneck_size: Bottleneck output channels
        emulate_keras: Reproduce original Keras behavior
        channel_dep: Independent stack per channel
        use_channel_pos_emb: Use channel positional embedding
    """
    def __init__(self,
                 seq_len: int,
                 in_channels: int,
                 num_layers: int = 6,
                 kernel_size: int = 41,
                 nb_filters: int = 32,
                 num_kernels: int = 3,
                 use_bottleneck: bool = True,
                 bottleneck_size: int = 32,
                 emulate_keras: bool = True,
                 channel_dep: bool = True,
                 use_channel_pos_emb: bool = True):
        super().__init__()

        self.seq_len = seq_len
        self.in_channels = in_channels
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters
        self.num_kernels = num_kernels
        self.use_bottleneck = use_bottleneck
        self.bottleneck_size = bottleneck_size
        self.emulate_keras = emulate_keras
        self.channel_dep = channel_dep

        self.out_dim = (num_kernels + 1) * nb_filters

        if use_channel_pos_emb and in_channels > 1:
            self.channel_pos_embedding = nn.Parameter(torch.randn(in_channels, 1, seq_len))
        else:
            self.channel_pos_embedding = None

        if channel_dep:
            self.encoder_list = nn.ModuleList([
                self._build_stack(in_ch=1)
                for _ in range(in_channels)
            ])
        else:
            self.shared_stack = self._build_stack(in_ch=1)

        self._init_weights()

    def _build_stack(self, in_ch: int) -> nn.Sequential:
        layers = []
        ch_in = in_ch
        res_in = ch_in

        for d in range(self.num_layers):
            inc = InceptionLayer(
                in_channels=ch_in,
                nb_filters=self.nb_filters,
                kernel_size=self.kernel_size,
                num_kernels=self.num_kernels,
                use_bottleneck=self.use_bottleneck,
                bottleneck_size=self.bottleneck_size,
                emulate_keras=self.emulate_keras,
                force_same_padding=True
            )
            layers.append(inc)
            ch_out = self.out_dim

            if d % 3 == 2:
                layers.append(ResidualBlock(res_in, ch_out))
                res_in = ch_out

            ch_in = ch_out

        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)

    def _forward_stack(self, x: torch.Tensor, stack: nn.Sequential) -> torch.Tensor:
        """Forward through stack with residual connections."""
        res = x
        for layer in stack:
            if isinstance(layer, ResidualBlock):
                x = layer(res, x)
                res = x
            else:
                x = layer(x)
        return x

    @staticmethod
    def _extract_patches(seq_feat: torch.Tensor,
                         window_size: int,
                         stride: int) -> torch.Tensor:
        """Extract patches from sequence features."""
        B, M, D, T = seq_feat.shape
        feat = seq_feat.view(B * M, D, T)
        patches = feat.unfold(dimension=2, size=window_size, step=stride)
        L = patches.size(-2)
        patches = patches.permute(0, 2, 3, 1).contiguous()
        patches = patches.view(B, M, L, window_size, D)
        return patches

    def forward(self, x: torch.Tensor, window_size: int, stride: int):
        """
        Args:
            x: (B, M, T) input tensor
        
        Returns:
            patch_feat: (B, M, L, window_size, D_out)
            seq_feat: (B*M, T, D_out)
        """
        B, M, T = x.shape
        assert M == self.in_channels, f"Expected {self.in_channels} channels, got {M}"

        if self.channel_pos_embedding is not None:
            x = x + self.channel_pos_embedding.squeeze(1)

        if self.channel_dep:
            seq_list = []
            for c in range(M):
                xc = x[:, c, :].unsqueeze(1)
                fc = self._forward_stack(xc, self.encoder_list[c])
                seq_list.append(fc.unsqueeze(1))
            seq_feat = torch.cat(seq_list, dim=1)
        else:
            x_flat = x.view(B * M, 1, T)
            f_flat = self._forward_stack(x_flat, self.shared_stack)
            seq_feat = f_flat.view(B, M, self.out_dim, T)

        patch_feat = self._extract_patches(seq_feat, window_size, stride)
        return patch_feat, seq_feat.view(B*M, T, self.out_dim)


if __name__ == "__main__":
    encoder = ShapeletInceptionEncoder(seq_len=40, in_channels=1)
    x_debug = torch.randn(5, 1, 40)
    patch_feat, seq_feat = encoder(x_debug, window_size=10, stride=5)
    print("Patch shape:", patch_feat.shape)
    print("Seq shape:", seq_feat.shape)