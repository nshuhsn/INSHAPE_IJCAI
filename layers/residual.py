import torch
import torch.nn as nn

class ResidualLayer(nn.Module):
    """
    Residual layer for 1D data with Conv1d, BatchNorm and optional dilation
    """
    def __init__(self, in_dim, h_dim, res_h_dim, dilation=1):
        super(ResidualLayer, self).__init__()
        kernel_size=3

        padding = ((kernel_size - 1) * dilation) // 2

        self.res_block = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.GELU(),
            nn.Conv1d(in_dim, res_h_dim, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(res_h_dim),
            nn.GELU(),
            nn.Conv1d(res_h_dim, h_dim, kernel_size=1, stride=1, bias=False)
        )

        self.residual_proj = nn.Identity()
        if in_dim != h_dim:
            self.residual_proj = nn.Conv1d(in_dim, h_dim, kernel_size=1, bias=False)

    def forward(self, x):
        res = self.residual_proj(x)
        x = res + self.res_block(x)
        return x
    
class ResidualStack(nn.Module):
    """
    A stack of residual layers inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack
    """
    def __init__(self, in_dim, h_dim, res_h_dim, n_res_layers):
        super(ResidualStack, self).__init__()
        self.n_res_layers = n_res_layers
        self.stack = nn.ModuleList(
            [ResidualLayer(in_dim, h_dim, res_h_dim)]*n_res_layers)

    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
        x = nn.ReLU()(x)
        return x
