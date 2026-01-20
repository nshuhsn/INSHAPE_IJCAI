"""
UEA Dataset Utilities

Data loading and preprocessing utilities for UEA multivariate time series datasets.
"""

import os
import numpy as np
import pandas as pd
import torch


def collate_fn(batch, max_len=None):
    """
    Collate function for UEA dataloader.
    
    Args:
        batch: list of tuples, each can be:
            - (x, y): basic format
            - (x, y, mask): with mask
            - (x, y, sid, mask): with sample id and mask
        x: (T, D) tensor
        y: (1,) or scalar
        mask: (T,) bool (True=valid step)
        sid: int sample id
    
    Returns:
        X: (B, T, D) padded input
        targets: (B, 1) labels
        sample_ids: (B,) sample identifiers
        padding_masks: (B, T) validity mask
    """
    xs, ys, sids, masks = [], [], [], []

    for item in batch:
        if len(item) == 4:
            x, y, sid, mask = item
            sids.append(int(sid))
            masks.append(mask.bool())
        elif len(item) == 3:
            x, y, mask = item
            sids.append(-1)
            masks.append(mask.bool())
        elif len(item) == 2:
            x, y = item
            sids.append(-1)
            masks.append(None)
        else:
            raise ValueError(f"Unexpected batch item length: {len(item)}")
        xs.append(x)
        ys.append(y)

    # Calculate lengths (prefer mask-based if available)
    lengths = []
    for i, x in enumerate(xs):
        if masks[i] is not None:
            lengths.append(int(masks[i].sum().item()))
        else:
            lengths.append(x.shape[0])

    if max_len is None:
        max_len = max(lengths)

    B = len(xs)
    T = max_len
    D = xs[0].shape[1]

    X = torch.zeros(B, T, D, dtype=xs[0].dtype)
    padding_masks = torch.zeros(B, T, dtype=torch.bool)

    for i, x in enumerate(xs):
        Li = min(x.shape[0], T)
        X[i, :Li, :] = x[:Li, :]

        if masks[i] is not None:
            mi = masks[i]
            if mi.numel() >= T:
                padding_masks[i] = mi[:T]
            else:
                padding_masks[i, :mi.numel()] = mi
        else:
            padding_masks[i, :Li] = True

    targets = torch.stack(ys, dim=0).long().view(B, -1)
    sample_ids = torch.tensor(sids, dtype=torch.long)

    return X, targets, sample_ids, padding_masks


def padding_mask(lengths, max_len=None):
    """
    Create boolean mask from sequence lengths.
    True means keep element at this position.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max_val()
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


class Normalizer(object):
    """
    Normalizes dataframe across ALL rows (time steps).
    Different from per-sample normalization.
    """

    def __init__(self, norm_type='standardization', mean=None, std=None, min_val=None, max_val=None):
        """
        Args:
            norm_type: normalization type
                - "standardization": z-score across all rows
                - "minmax": min-max across all rows
                - "per_sample_std": z-score per sample
                - "per_sample_minmax": min-max per sample
            mean, std, min_val, max_val: optional pre-computed values
        """
        self.norm_type = norm_type
        self.mean = mean
        self.std = std
        self.min_val = min_val
        self.max_val = max_val

    def normalize(self, df):
        """Normalize input dataframe."""
        if self.norm_type == "standardization":
            if self.mean is None:
                self.mean = df.mean()
                self.std = df.std()
            return (df - self.mean) / (self.std + np.finfo(float).eps)

        elif self.norm_type == "minmax":
            if self.max_val is None:
                self.max_val = df.max()
                self.min_val = df.min()
            return (df - self.min_val) / (self.max_val - self.min_val + np.finfo(float).eps)

        elif self.norm_type == "per_sample_std":
            grouped = df.groupby(by=df.index)
            return (df - grouped.transform('mean')) / grouped.transform('std')

        elif self.norm_type == "per_sample_minmax":
            grouped = df.groupby(by=df.index)
            min_vals = grouped.transform('min')
            return (df - min_vals) / (grouped.transform('max') - min_vals + np.finfo(float).eps)

        else:
            raise NameError(f'Normalize method "{self.norm_type}" not implemented')


def interpolate_missing(y):
    """Replace NaN values using linear interpolation."""
    if y.isna().any():
        y = y.interpolate(method='linear', limit_direction='both')
    return y


def subsample(y, limit=256, factor=2):
    """Subsample series if longer than limit."""
    if len(y) > limit:
        return y[::factor].reset_index(drop=True)
    return y
