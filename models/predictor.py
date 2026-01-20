"""
Inception-based Predictor for Time Series Classification

Keras InceptionTime-style predictor with GAP + Linear classifier head.
"""

import sys
import os
sys.path.append('..')
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.InceptionBlock import InceptionLayer, ResidualBlock, kaiming_init


class InceptionPredictor(nn.Module):
    """
    Keras InceptionTime-style predictor.
    Classifier head: Global Average Pooling + Linear
    
    Args:
        seq_len: Input sequence length
        num_classes: Number of output classes
        in_channels: Number of input channels
        nb_filters: Filters per branch
        depth: Number of Inception layers
        kernel_size: Largest kernel size
        num_kernels: Number of Conv branches
        use_residual: Whether to use residual connections
        use_bottleneck: Whether to use bottleneck
        bottleneck_size: Bottleneck output channels
        emulate_keras: Reproduce original Keras behavior
    """
    def __init__(self,
                 seq_len: int,
                 num_classes: int,
                 in_channels: int,
                 nb_filters: int = 32,
                 depth: int = 6,
                 kernel_size: int = 41,
                 num_kernels: int = 3,
                 use_residual: bool = True,
                 use_bottleneck: bool = True,
                 bottleneck_size: int = 32,
                 emulate_keras: bool = True):
        super().__init__()
        print(f"[InceptionPredictor] in={in_channels}, nb_filters={nb_filters}, depth={depth}")

        self.use_residual = use_residual
        layers = []
        res_root_channels = in_channels
        x_ch = in_channels
        last_out_ch = None

        for d in range(depth):
            inc = InceptionLayer(
                in_channels=x_ch,
                nb_filters=nb_filters,
                kernel_size=kernel_size,
                num_kernels=num_kernels,
                use_bottleneck=use_bottleneck,
                bottleneck_size=bottleneck_size,
                emulate_keras=emulate_keras,
                force_same_padding=True
            )
            layers.append(inc)
            out_ch = (num_kernels + 1) * nb_filters
            last_out_ch = out_ch

            if use_residual and d % 3 == 2:
                layers.append(ResidualBlock(res_root_channels, out_ch))
                res_root_channels = out_ch

            x_ch = out_ch

        self.feature_extractor = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(last_out_ch, num_classes)
        self.num_classes = num_classes 
        self.apply(kaiming_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) input tensor
        
        Returns:
            logits: (B, num_classes)
        """
        res = x
        for layer in self.feature_extractor:
            if isinstance(layer, ResidualBlock):
                x = layer(res, x)
                res = x
            else:
                x = layer(x)

        x = self.gap(x).squeeze(-1)
        return self.classifier(x)


if __name__ == "__main__":
    pass