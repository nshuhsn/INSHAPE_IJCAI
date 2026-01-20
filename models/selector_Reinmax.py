import math
import torch
import torch.nn as nn
from models.Reinmax import reinmax  

class ROITransformerSelector(nn.Module):
    def __init__(self, d_model=64, n_head=4, n_layer=3):
        super().__init__()
        self.proj = nn.Linear(1, d_model)
        self.cls  = nn.Parameter(torch.randn(1,1,d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model, n_head, 4*d_model, batch_first=True)
        self.enc  = nn.TransformerEncoder(enc_layer, n_layer)
        self.head = nn.Linear(d_model, 2)          # select / not

    def init_gate_bias(self, target_p=0.995, zero_weight=True):
        delta = math.log(target_p / (1 - target_p))  # 로짓 차이
        with torch.no_grad():
            if zero_weight: self.head.weight.zero_()
            bias = self.head.bias.view(-1, 2)
            bias[:, 0] = -delta / 2
            bias[:, 1] =  delta / 2

    def forward(self, seg, pad_mask, tau=1.3, training=True):
        """
        seg       : (N, T, 1)   raw+pad=0
        pad_mask  : (N, T)      True pad
        """
        x = self.proj(seg)                          # (N,T,D)
        cls = self.cls.expand(x.size(0), -1, -1)    # (N,1,D)
        x = torch.cat([cls, x], dim=1)              # prepend CLS
        pad_mask = torch.cat(
            [torch.zeros_like(pad_mask[:,:1]), pad_mask], dim=1).bool()

        h = self.enc(x, src_key_padding_mask=pad_mask)   # (N,T+1,D)
        h_cls = h[:,0]
        logits = self.head(h_cls)                        # (N,2)

        m, probs = reinmax(logits, tau=tau, training=training)
        return m[..., 1:2], logits[:,1:2], probs                  # (N,1)

if __name__=="__main__":
    pass