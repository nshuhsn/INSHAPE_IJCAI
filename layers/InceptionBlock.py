"""
Inception Block Layers for Time Series

Keras InceptionTime-style 1D Inception blocks with residual connections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


def kaiming_init(m: nn.Module):
    """Initialize weights using Kaiming initialization."""
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


class InceptionLayer(nn.Module):
    """
    Keras InceptionTime-style Inception layer.
    
    Features:
        - Optional Bottleneck 1x1 conv (32 channels)
        - Multiple kernel sizes: kernel_size_s = (kernel_size-1)//(2**i)
        - 3 Conv branches + MaxPool->1x1 branch
        - Concat all branches -> BN + ReLU
        - SAME padding
    
    Args:
        in_channels: Number of input channels
        nb_filters: Filters per branch
        kernel_size: Largest kernel size
        num_kernels: Number of Conv branches
        use_bottleneck: Whether to use bottleneck
        bottleneck_size: Bottleneck output channels
        emulate_keras: Reproduce Keras behavior
        force_same_padding: Force same padding
    """
    def __init__(self,
                 in_channels: int,
                 nb_filters: int = 32,
                 kernel_size: int = 41,
                 num_kernels: int = 3,
                 use_bottleneck: bool = True,
                 bottleneck_size: int = 32,
                 emulate_keras: bool = True,
                 force_same_padding: bool = True):
        super().__init__()
        self.use_bottleneck = use_bottleneck and in_channels > 1
        self.nb_filters = nb_filters
        self.num_kernels = num_kernels
        self.emulate_keras = emulate_keras
        self.force_same_padding = force_same_padding

        # Keras uses kernel_size - 1 internally
        effective_base = kernel_size - 1 if emulate_keras else kernel_size
        kernel_sizes: List[int] = [max(1, effective_base // (2 ** i)) for i in range(num_kernels)]

        # Bottleneck
        if self.use_bottleneck:
            self.bottleneck = nn.Conv1d(in_channels, bottleneck_size, kernel_size=1, bias=False)
            conv_in = bottleneck_size
        else:
            self.bottleneck = None
            conv_in = in_channels

        self.kernel_sizes = kernel_sizes
        self.conv_layers = nn.ModuleList()
        self.need_postpad = False

        for ks in kernel_sizes:
            if force_same_padding:
                try:
                    conv = nn.Conv1d(conv_in, nb_filters, kernel_size=ks,
                                     padding='same', bias=False)
                except TypeError:
                    # Fallback for older PyTorch versions
                    pad = ks // 2 if ks % 2 == 1 else ks // 2 - 1
                    conv = nn.Conv1d(conv_in, nb_filters, kernel_size=ks,
                                     padding=pad, bias=False)
                    if ks % 2 == 0:
                        self.need_postpad = True
            else:
                pad = ks // 2 if ks % 2 == 1 else ks // 2 - 1
                conv = nn.Conv1d(conv_in, nb_filters, kernel_size=ks,
                                 padding=pad, bias=False)
                if ks % 2 == 0:
                    self.need_postpad = True
            self.conv_layers.append(conv)

        # MaxPool branch
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, nb_filters, kernel_size=1, bias=False)
        )

        total_out = (num_kernels + 1) * nb_filters
        self.bn = nn.BatchNorm1d(total_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) input tensor
        Returns:
            (B, (num_kernels+1)*nb_filters, T)
        """
        if self.bottleneck is not None:
            x_in = self.bottleneck(x)
        else:
            x_in = x

        outs = []
        for conv, ks in zip(self.conv_layers, self.kernel_sizes):
            o = conv(x_in)
            # Manual even kernel padding correction
            if self.need_postpad and ks % 2 == 0 and o.shape[-1] == x.shape[-1] - 1:
                o = F.pad(o, (0, 1))
            outs.append(o)

        pool_out = self.pool_branch(x)
        outs.append(pool_out)

        # Align lengths if mismatch
        lengths = [t.shape[-1] for t in outs]
        if len(set(lengths)) > 1:
            max_len = max(lengths)
            outs = [F.pad(t, (0, max_len - t.shape[-1])) for t in outs]

        x_cat = torch.cat(outs, dim=1)
        x_cat = self.bn(x_cat)
        return F.relu(x_cat, inplace=True)


class ResidualBlock(nn.Module):
    """
    Residual block with 1x1 Conv + BN projection.
    Always applies projection (Keras style).
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x_prev: torch.Tensor, x_cur: torch.Tensor) -> torch.Tensor:
        shortcut = self.proj(x_prev)
        return F.relu(shortcut + x_cur, inplace=True)
