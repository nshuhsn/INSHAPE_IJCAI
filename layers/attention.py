import torch
import torch.nn as nn
import torch.nn.functional as F

class SelectorAttnLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()

        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads,
                                               dropout=0, batch_first=True) 
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*4),
            nn.ELU(),
            nn.Linear(hidden_dim*4, hidden_dim),
        )

        self.norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, h):

        h_attn, _ = self.attention(h,h,h)

        h = self.norm1(h_attn+h)

        h_ffn = self.ffn(h)

        h = self.norm2(h_ffn+h)

        return h

