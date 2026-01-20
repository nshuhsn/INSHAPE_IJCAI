import torch
import torch.nn as nn
import torch.nn.functional as F


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TCNBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(TCNBlock1d, self).__init__()
        padding = (kernel_size-1) * dilation

        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size, padding=padding,
                              dilation=dilation)
        
        self.chomp = Chomp1d(padding)
        
        self.norm = nn.BatchNorm1d(out_channels)

        self.elu = nn.ELU()


    def forward(self, x):
        out = self.conv(x)

        out = self.chomp(out)

        out = self.norm(out)

        out = self.elu(out)

        return out
    
class TCNBlock1dwithDropout(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(TCNBlock1dwithDropout, self).__init__()
        padding = (kernel_size-1) * dilation

        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size, padding=padding,
                              dilation=dilation)
        
        self.chomp = Chomp1d(padding)
        
        self.norm = nn.BatchNorm1d(out_channels)

        self.elu = nn.ELU()

        self.dropout = nn.Dropout(p=0.1)


    def forward(self, x):
        out = self.conv(x)

        out = self.chomp(out)

        out = self.norm(out)

        out = self.elu(out)

        out = self.dropout(out)

        return out

