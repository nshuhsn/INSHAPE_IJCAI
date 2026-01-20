import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .encoder import ShapeletInceptionEncoder
from .predictor import InceptionPredictor
from .selector_Reinmax import ROITransformerSelector
from utils.util import sliding_window, standardization, extract_segments_batch, union_segments_scatter
from .ROI_search import segment, pack_valid_roi_fast

class MainFlow(nn.Module):
    def __init__(self, seq_len, num_channels,
                 num_layers=6, num_classes=2, device="cuda:0", window_size=None, args=None):
        super().__init__()
        self.device = torch.device(device)
        self.window_size = window_size if window_size is not None else max(1, seq_len // 60)
        self.seq_len = seq_len
        self.num_classes = num_classes

        nb_filters = 32
        self.selector = ROITransformerSelector(d_model=32, n_head=4, n_layer=1)
        self.predictor = InceptionPredictor(
            seq_len=self.seq_len,
            num_classes=num_classes,
            in_channels=num_channels,
            nb_filters=nb_filters,
            depth=num_layers,
            kernel_size=41,
            num_kernels=3,
            use_residual=True,
            use_bottleneck=True,
            bottleneck_size=32,
            emulate_keras=True
        )

        def kaiming_init_weights(m):
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.apply(kaiming_init_weights)
        self.selector.init_gate_bias(target_p=0.80, zero_weight=False)

        # Batch-level segmentation cache (key=(tuple(sample_ids), T))
        self._seg_cache = {}

    def forward(self, x: torch.Tensor, padding_mask=None, training: bool = True, tau: float = 1.3,
            sample_ids=None):
        """
        Args:
            x: [B, M, T] input tensor
            sample_ids: [B] sample identifiers from DataLoader; used as cache key
        """
        dbg = bool(getattr(self, "debug_cache", False))

        x = x.to(self.device)
        if x.dim() != 3:
            x = x.unsqueeze(1)
        if padding_mask is not None:
            x = x * padding_mask.to(self.device).unsqueeze(1)
        x = x + 1e-6

        B, M, T = x.shape
        pen0 = 0.1 * np.log(T)
        min_len = max(1, T // 30)

        # 1) sample_ids required: for batch-independent per-sample caching
        if sample_ids is None:
            if dbg:
                print("[Cache] sample_ids=None -> per-sample cache unavailable. Computing directly.")
            roi_time_mask, roi_valid, L_max = segment(x, min_len=min_len, pen=pen0)
        else:
            # Normalize sample_ids
            if torch.is_tensor(sample_ids):
                sids = [int(s) for s in sample_ids.tolist()]
            else:
                sids = [int(s) for s in sample_ids]

            if not hasattr(self, "_seg_cache"):
                self._seg_cache = {}

            # 2) Collect only cache-miss samples and run segment() once
            miss_mask = [sid not in self._seg_cache for sid in sids]
            if any(miss_mask):
                miss_idx = [i for i, m in enumerate(miss_mask) if m]
                x_miss = x[miss_idx]  # (Bm, M, T)
                if dbg:
                    print(f"[Cache Miss] {len(miss_idx)}/{B} samples -> running segment()")
                roi_tm_m, roi_v_m, Lm = segment(x_miss, min_len=min_len, pen=pen0)  # (Bm,M,Rm,T),(Bm,M,Rm),int
                # Store per-sample in cache
                for j, i in enumerate(miss_idx):
                    sid = sids[i]
                    roi_tm_b = roi_tm_m[j]  # (M,Rb,T)
                    roi_v_b = roi_v_m[j]    # (M,Rb)
                    Lb = Lm
                    self._seg_cache[sid] = (roi_tm_b, roi_v_b, Lb)
                if dbg:
                    print(f"    -> Cache stored: now cache_size={len(self._seg_cache)}")
            else:
                if dbg:
                    print("All samples in batch are cached")

            # 3) Combine into batch (pad R axis)
            R_list = []
            for sid in sids:
                roi_tm_b, _, _ = self._seg_cache[sid]
                R_list.append(roi_tm_b.shape[1])
            R_max = max(R_list)
            L_max = max(self._seg_cache[sid][2] for sid in sids)

            roi_time_mask = torch.zeros(B, M, R_max, T, device=self.device, dtype=torch.float32)
            roi_valid = torch.zeros(B, M, R_max, device=self.device, dtype=torch.bool)

            for b, sid in enumerate(sids):
                roi_tm_b, roi_v_b, _ = self._seg_cache[sid]  # (M,Rb,T), (M,Rb)
                Rb = roi_tm_b.shape[1]
                roi_time_mask[b, :, :Rb, :] = roi_tm_b.to(self.device)
                roi_valid[b, :, :Rb] = roi_v_b.to(self.device)

        seg, pad_mask, idx_map = pack_valid_roi_fast(x, roi_time_mask, roi_valid, L_max)
        m_flat, logit, probs = self.selector(seg, pad_mask, tau=tau, training=training)

        roi_mask_valid = roi_time_mask[roi_valid]  # (N,T)
        selected_ROI = roi_mask_valid * m_flat     # (N,T)

        rows = list(idx_map.keys())
        if len(rows) == 0:
            time_mask = torch.zeros(B, M, T, device=self.device, dtype=selected_ROI.dtype)
        else:
            b_idx, m_idx, _ = map(torch.tensor, zip(*rows))
            b_idx = b_idx.to(self.device)
            m_idx = m_idx.to(self.device)
            flat_id = b_idx * M + m_idx
            time_mask_flat = torch.zeros(B * M, T, device=self.device, dtype=selected_ROI.dtype)
            time_mask_flat.index_add_(0, flat_id, selected_ROI)
            time_mask = time_mask_flat.view(B, M, T)

        masked_X = x * time_mask
        pred_logits = self.predictor(masked_X)

        if dbg:
            with torch.no_grad():
                sel_rate = (time_mask > 0).float().mean().item()
                print(f"   [dbg] T={T}, L_max={L_max}, ROI(valid)={int(roi_valid.sum().item())}, "
                      f"selection_rate={sel_rate:.4f}")

        return pred_logits, time_mask, masked_X, x, logit, x, probs

    def reset_seg_cache(self):
        """Clear the segmentation cache."""
        self._seg_cache.clear()