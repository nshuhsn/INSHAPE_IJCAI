import numpy as np
import torch 
import torch.nn.functional as F
import copy
import os
import random 
import pdb
import math
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_curve, auc, precision_recall_curve, matthews_corrcoef  
from torch.distributions import Bernoulli

import torch
import torch.nn as nn

class GSATLoss(nn.Module):

    def __init__(self, r):
        super(GSATLoss, self).__init__()
        self.r = r

    def forward(self, att):
        if torch.any(torch.isnan(att)):
            print('ALERT - att has nans')
            exit()
        if torch.any(att < 0):
            print('ALERT - att less than 0')
            exit()
        assert (att < 0).sum() == 0
        info_loss = (att * torch.log(att/self.r + 1e-6) + (1-att) * torch.log((1-att)/(1-self.r + 1e-6) + 1e-6)).mean()
        ##print(info_loss)
        if torch.any(torch.isnan(info_loss)):
            print('INFO LOSS NAN')
            exit()
        return info_loss

def setup_optimizers(model, train_loader, args):
    """
    Setup optimizers for predictor and selector/encoder.
    Returns: optimizers dict, schedulers dict
    """
    def trainable(module):
        return [p for p in module.parameters() if p.requires_grad]

    has_selector = hasattr(model, 'selector') and model.selector is not None
    has_encoder = hasattr(model, 'encoder') and model.encoder is not None

    pred_params = trainable(model.predictor)
    optimizer_pred = torch.optim.Adam(pred_params, lr=args.pred_lr)

    if has_selector:
        sel_params = trainable(model.selector)
        if has_encoder:
            sel_params += trainable(model.encoder)

        if sel_params:
            optimizer_sel = torch.optim.Adam(sel_params, lr=args.sel_lr)
        else:
            optimizer_sel = None
            print("[setup_optimizers] All selector/encoder params are frozen.")
    else:
        optimizer_sel = None
        print("[setup_optimizers] No selector module, skipping sel optimizer.")

    optimizers = {'pred': optimizer_pred, 'sel': optimizer_sel}
    schedulers = {'pred': None, 'sel': None}
    return optimizers, schedulers

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    print(f"[Info] Setting seed to {seed} for all relevant libraries.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    os.environ["PYTHONHASHSEED"] = str(seed)


def diversity_reg(m, z, lambda_div=0.01):
    '''
    m : [B, M, L, 1]
    z : [B, M, L, l]
    '''
    channels = m.size(1)
    div_list = []

    for channel in range(channels):
        z_ch = z[:,channel,:,:]
        m_ch = m[:,channel,:,:]
        distance_matrix = torch.exp(-1*torch.cdist(z_ch, z_ch, p=2))  # [B, L, L]
        masking_matrix = m_ch @ m_ch.transpose(2, 1) # [B, L, L]
        masked_distance = distance_matrix * masking_matrix

        div_reg = torch.norm(masked_distance, p='fro',dim=(1,2)) # [B]

        norm_term = masking_matrix.sum(dim=(1, 2))
        norm_term = torch.where(norm_term > 0, norm_term, torch.ones_like(norm_term))  # avoid division by zero
        if div_reg.dim() != 1:
            raise AssertionError

        div_reg = div_reg/norm_term # [B]
        div_list.append(div_reg.unsqueeze(1))
    
    div_reg = torch.cat(div_list,dim=1) # [B,M]
    div_reg = div_reg.mean()
    
    return lambda_div * div_reg

def standardization(x):
    '''
    x : [B, M, seq_len]
    '''
    mean = x.mean(dim=2, keepdim=True) # [B, M, 1]
    std = x.std(dim=2, keepdim=True) + 1e-8 # [B, M, 1]
    x = (x - mean) / std 
    return x, mean, std

def extract_segments_batch(z_tilde, pad_value=0.0):
    """
    Differentiable batch version of extract_segments.
    Args:
        z_tilde: [batch, seq_len]
    Returns:
        padded: [batch, num_segments, max_len] tensor
    """
    batch_size, seq_len = z_tilde.shape
    all_segments = []
    num_segments_list = []
    seg_lens = []

    # 1. 각 배치별로 segment 추출
    for b in range(batch_size):
        seq = z_tilde[b]
        is_zero = (seq == 0).to(torch.long)
        changes = torch.cat([torch.tensor([1], device=seq.device), is_zero[1:] != is_zero[:-1]])
        segment_starts = torch.where(changes)[0]
        segment_ends = torch.cat([segment_starts[1:], torch.tensor([seq.size(0)], device=seq.device)])
        segments = [seq[start:end] for start, end in zip(segment_starts, segment_ends)]
        all_segments.append(segments)
        num_segments_list.append(len(segments))
        seg_lens.extend([seg.size(0) for seg in segments])

    max_num_segments = max(num_segments_list)
    max_len = max(seg_lens)

    # 2. batch tensor로 합치기 (패딩)
    padded = z_tilde.new_full((batch_size, max_num_segments, max_len), pad_value)

    for b, segments in enumerate(all_segments):
        for i, seg in enumerate(segments):
            seg_len = seg.size(0)
            pad_left = (max_len - seg_len) // 2
            padded[b, i, pad_left:pad_left+seg_len] = seg

    return padded
    
def sliding_window(input: torch.Tensor, window: int, stride: int = 1) -> torch.Tensor:
    '''
    Change the input tensor size for making custom dataset
    Args:
        input: (Batch Size, Channels, TS_length)
        window: Window size for sliding
    Returns:
        Tensor of shape (Batch Size, Channels, N_seg, window)
    '''
    return input.unfold(dimension=2, size=window, step=stride)

def union_segments_scatter(
        m: torch.Tensor,     # [B, M, L, 1]
        z: torch.Tensor,     # [B, M, L, l]
        stride: int,
        seq_len: int,
        hard: bool
    ) -> torch.Tensor:       # [B, M, seq_len]

    B, M, L, l = z.shape
    device, dtype = z.device, z.dtype

    max_end = (L - 1) * stride + l
    if max_end > seq_len:
        raise ValueError(f"seq_len({seq_len}) is insufficient. Minimum required: {max_end}.")

    base = torch.arange(L, device=device) * stride
    idx = base.unsqueeze(1) + torch.arange(l, device=device)
    idx_flat = idx.reshape(-1)
    idx_expand = idx_flat.view(1, 1, -1).expand(B, M, -1)

    weight_per_token = m.expand(-1, -1, -1, l)
    z_flat = (z * weight_per_token).reshape(B, M, -1)
    cnt_flat = (weight_per_token if hard
                else torch.ones_like(weight_per_token)).reshape(B, M, -1)

    z_out = torch.zeros(B, M, seq_len, device=device, dtype=dtype)
    cnt_out = torch.zeros_like(z_out)

    z_out.scatter_add_(2, idx_expand, z_flat)
    cnt_out.scatter_add_(2, idx_expand, cnt_flat)

    cnt_out.clamp_min_(1.)
    return z_out / cnt_out

def union_mask_scatter(
        m: torch.Tensor,
        stride: int,
        seq_len: int,
        window_size: int,
        hard: bool
    ) -> torch.Tensor:
    """
    Scatter masks back to sequence length.
    m: [B, M, L, 1], returns: [B, M, seq_len]
    """
    B, M, L, _ = m.shape
    device, dtype = m.device, m.dtype

    max_end = (L - 1) * stride + window_size
    if max_end > seq_len:
        raise ValueError(
            f"seq_len({seq_len}) is insufficient. Minimum required: {max_end}."
        )

    base = torch.arange(L, device=device) * stride
    idx = base.unsqueeze(1) + torch.arange(window_size, device=device)
    idx_flat = idx.reshape(-1)
    idx_expand = idx_flat.view(1, 1, -1).expand(B, M, -1)

    m_expand = m.expand(-1, -1, -1, window_size)
    mask_flat = m_expand.reshape(B, M, -1)

    cnt_flat = mask_flat if hard else torch.ones_like(mask_flat)
    out = torch.zeros(B, M, seq_len, device=device, dtype=dtype)
    count = torch.zeros_like(out)

    out.scatter_add_(2, idx_expand, mask_flat)
    count.scatter_add_(2, idx_expand, cnt_flat)

    count.clamp_min_(1.)
    return out / count

def predictor_loss(y_true, pred_logits):
    '''
    pred_logits : [B, n_classes]
    y_true : [B,1]
    '''
    y = y_true.squeeze(-1).long()
    y = y.view(-1)

    # Ensure pred_logits and y have matching shapes for cross_entropy
    assert pred_logits.size(0) == y.size(0), f"pred_logits batch: {pred_logits.size(0)}, y batch: {y.size(0)}"
    
    pred_loss = F.cross_entropy(pred_logits, y, reduction='mean')

    return pred_loss

def get_numlayers(seq_len, kernel_size):
    '''
    Calculate the number of layers needed for a TCN to cover the input sequence length
    '''
    return int(np.ceil(np.log2((seq_len-1)/(kernel_size-1) + 1)))

def get_all_metrics(y_true, y_pred, y_pred_proba, n_classes, prefix=""):
    """Compute and return all classification metrics."""
    metrics_dict = {}
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    mcc = matthews_corrcoef(y_true, y_pred)
    
    metrics_dict.update({
        f'{prefix}_accuracy': accuracy,
        f'{prefix}_precision': precision,
        f'{prefix}_recall': recall,
        f'{prefix}_f1': f1,
        f'{prefix}_mcc': mcc,
    })
    
    roc_aucs = []
    pr_aucs = []
    
    for i in range(n_classes):
        y_true_binary = (y_true == i)
        
        fpr, tpr, _ = roc_curve(y_true_binary, y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        roc_aucs.append(roc_auc)
        metrics_dict[f'{prefix}_roc_auc_class_{i}'] = roc_auc

        precision_curve, recall_curve, _ = precision_recall_curve(y_true_binary, y_pred_proba[:, i])
        pr_auc = auc(recall_curve, precision_curve)
        pr_aucs.append(pr_auc)
        metrics_dict[f'{prefix}_pr_auc_class_{i}'] = pr_auc

    metrics_dict[f'{prefix}_mean_roc_auc'] = np.mean(roc_aucs)
    metrics_dict[f'{prefix}_mean_pr_auc'] = np.mean(pr_aucs)
    
    # 결과 출력
    print(f"\n{prefix} Metrics:")
    print(f"  Basic Classification Metrics:")
    print(f"    Accuracy: {accuracy:.4f}")
    print(f"    Precision: {precision:.4f}")
    print(f"    Recall: {recall:.4f}")
    print(f"    F1-score: {f1:.4f}")
    print(f"    F1-score: {mcc:.4f}")
    
    print(f"  ROC-AUC scores per class:")
    for i, roc_auc in enumerate(roc_aucs):
        print(f"    Class {i}: {roc_auc:.4f}")
    print(f"    Mean ROC-AUC: {np.mean(roc_aucs):.4f}")
    
    print(f"  PR-AUC scores per class:")
    for i, pr_auc in enumerate(pr_aucs):
        print(f"    Class {i}: {pr_auc:.4f}")
    print(f"    Mean PR-AUC: {np.mean(pr_aucs):.4f}")
    
    return metrics_dict

if __name__=="__main__":
    # test new union_segments
    import torch

    B, M, L, l = 2, 1, 3, 3
    stride = 1
    seq_len = 5

    # 예시 segment 값 (B=2, L=3, l=3)
    z = torch.tensor([
        [[1,2,3], [2,3,4], [3,4,5]],
        [[10,11,12], [11,12,13], [12,13,14]]
    ], dtype=torch.float)

    z = z.unsqueeze(1) 

    m = torch.ones(B, 1, L, 1)
    m[:, : , 2,:] = 0

    z_tilde = union_segments_scatter(m, z, stride=stride, seq_len=seq_len, hard = True)

    print("m: ", m)
    print("z_tilde: ", z_tilde)
    print("m and z_tilde shape: ", m.shape, z_tilde.shape)