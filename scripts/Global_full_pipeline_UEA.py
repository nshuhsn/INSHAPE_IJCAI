"""
Unified Pipeline for Multivariate (UEA) Shapelet Extraction and Dataset Replacement

This script orchestrates the full pipeline for UEA multivariate datasets:
1. Check if global shapelets exist
2. If not: Extract candidates (Step 1) + Extract global shapelets (Step 2)
3. Apply global similarity filtering (optional)
4. Create replaced datasets (Step 3)

Key differences from run_full_pipeline.py (UCR univariate):
- Handles multivariate data: [B, T, D] → transposed to [B, D, T]
- Each channel is processed independently but stored in a single file
- Supports channel-aware shapelet matching for replacement
- Includes global similarity filtering option

Supports multi-GPU parallel processing for Step 3.
"""

import sys
import os
import gc
import glob
import argparse
import datetime
from types import SimpleNamespace
from pathlib import Path
from multiprocessing import Pool
from typing import List, Optional, Tuple, Dict, Any
from collections import defaultdict

import torch
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Add paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_provider.data_factory import data_provider, UEAloader
from data_provider.uea import collate_fn
from models.INSHAPE import MainFlow
from models.ROI_search import segment, pack_valid_roi_fast


# ============================================================================
# Helper Functions
# ============================================================================

def get_latest_file(pattern):
    """Get most recent file matching pattern"""
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def to_2d_np(x) -> np.ndarray:
    """Convert Tensor/ndarray [L] or [L,F] -> np.ndarray [L,F]"""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x)
    return arr[:, None] if arr.ndim == 1 else arr


def z_norm_2d(arr: np.ndarray) -> np.ndarray:
    """Z-normalize 2D array along time axis"""
    mu = arr.mean(axis=0, keepdims=True)
    sd = arr.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    return (arr - mu) / sd


def dtw_distance(a: np.ndarray, b: np.ndarray, radius: Optional[int] = None) -> float:
    """Compute DTW distance between two time series"""
    if radius is None:
        d, _ = fastdtw(a, b, dist=euclidean)
    else:
        d, _ = fastdtw(a, b, dist=euclidean, radius=radius)
    return d


# ============================================================================
# Step 1: Extract Shapelet Candidates (Multivariate)
# ============================================================================

@torch.inference_mode()
def run_selector_only_and_save_multivariate(
    selector,
    loader,
    dataset_name: str,
    out_root: str = "shapelet_output_UEA",
    buffer_limit: int = 4000,
    tau: float = 1.3,
    min_len_frac: float = 1/30,
    pen_scale: float = 0.1,
    save_fp16: bool = False,
    algorithm: str = 'pelt',
) -> str:
    """
    Run selector model to extract shapelet candidates from multivariate data.
    Saves channel information with each candidate.
    
    Args:
        selector: Trained selector model
        loader: DataLoader for input data (UEA format)
        dataset_name: Name of dataset
        out_root: Root directory for output
        buffer_limit: Number of candidates per shard file
        tau: Temperature for Gumbel-Softmax
        min_len_frac: Minimum segment length as fraction of sequence length
        pen_scale: Penalty scale for segmentation
        save_fp16: Whether to save in fp16
        algorithm: Segmentation algorithm
    
    Returns:
        str: Output directory path
    """
    device = next(selector.parameters()).device
    out_dir = os.path.join(out_root, dataset_name)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"[Candidate Extraction - Multivariate] Dataset: {dataset_name}")
    print(f"  Output: {out_dir}")
    print(f"  Algorithm: {algorithm}, tau: {tau}, min_len_frac: {min_len_frac}")

    def flush(buf, shard_id):
        """Save buffer to shard file"""
        if not buf:
            return []
        path = os.path.join(out_dir, f"selector_only_shard_{shard_id:05d}.pt")
        packed = []
        for s in buf:
            vals = s['values']
            if save_fp16:
                vals = vals.to(torch.float16)
            packed.append({
                'b': int(s['b']),
                'm': int(s['m']),          # Channel index
                'channel': int(s['m']),    # Explicit channel field
                'label': int(s['label']),
                'start': int(s['start']),
                'end': int(s['end']),
                'score': float(s['score']),
                'values': vals
            })
        torch.save({'dataset': dataset_name, 'candidates': packed}, path)
        print(f"  [Shard {shard_id:05d}] Saved {len(buf)} candidates -> {path}")
        return []

    buf = []
    shard_id = 0
    total_saved = 0

    for batch_idx, batch in enumerate(loader):
        # Parse batch - UEA format: (x, y, sid, mask)
        if isinstance(batch, (list, tuple)):
            if len(batch) >= 4:
                data, target, sid, padding_mask = batch[:4]
            elif len(batch) == 3:
                data, target, sid = batch[:3]
                padding_mask = None
            else:
                data, target = batch[:2]
                padding_mask = None
        else:
            data = batch
            target = torch.zeros(data.shape[0], dtype=torch.long)
            padding_mask = None
        
        # UEA data comes as [B, T, D], transpose to [B, D, T] for model
        x = data.detach().to('cpu').contiguous()
        if x.dim() == 3 and x.shape[1] != x.shape[2]:
            # Assume [B, T, D] format, transpose to [B, D, T]
            if x.shape[1] > x.shape[2]:  # T > D typically
                x = x.transpose(1, 2)  # Now [B, D, T]
        elif x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, T]
        
        B, M, T = x.shape

        # Add epsilon, apply padding
        eps = 1e-6
        x = x + eps
        if padding_mask is not None:
            pm = padding_mask.to('cpu')
            if pm.dim() == 1:
                pm = pm.unsqueeze(1)
            x = x * pm.unsqueeze(1) if pm.dim() == 2 else x

        # Segmentation (CPU)
        min_len = max(1, int(T * min_len_frac))
        pen0 = float(pen_scale * np.log(T))
        roi_time_mask, roi_valid, L_max = segment(
            x, min_len=min_len, pen=pen0, algorithm=algorithm
        )

        if roi_time_mask.dtype != torch.bool:
            roi_time_mask = roi_time_mask.bool()

        # Packing (CPU)
        seg, pad_mask, idx_map = pack_valid_roi_fast(x, roi_time_mask, roi_valid, L_max)

        # Selector call (CUDA)
        seg_d = seg.to(device, non_blocking=True)
        pad_d = pad_mask.to(device, non_blocking=True)
        m_flat, logit, probs = selector(seg_d, pad_d, tau=tau, training=False)
        m_flat = m_flat.detach().to('cpu').view(-1, 1)

        # Extract selected ROI regions
        roi_mask_valid = roi_time_mask[roi_valid]
        selected_ROI = roi_mask_valid.to(torch.float32) * m_flat

        # Build (n -> (b,m)) reverse mapping
        rows = list(idx_map.keys())
        b_idx, m_idx, _ = map(torch.tensor, zip(*rows))
        b_idx = b_idx.to(x.device)
        m_idx = m_idx.to(x.device)

        BM = B * M
        flat_id = b_idx * M + m_idx

        # time-mask accumulation
        time_mask_flat = torch.zeros(BM, T, dtype=selected_ROI.dtype)
        time_mask_flat.index_add_(0, flat_id, selected_ROI)

        time_mask = time_mask_flat.view(B, M, T)
        masked_X = x * time_mask
        time_mask_bool = (time_mask > 0)

        # Extract continuous segments as candidates (with channel info)
        cands = []
        for b in range(B):
            for m in range(M):
                mask = time_mask_bool[b, m]
                if not mask.any():
                    continue
                mask_i = mask.to(torch.int32)
                pad = torch.zeros(T + 2, dtype=torch.int32)
                pad[1:-1] = mask_i
                diff = pad[1:] - pad[:-1]
                starts_u = torch.nonzero(diff == 1, as_tuple=True)[0]
                ends_u = torch.nonzero(diff == -1, as_tuple=True)[0]

                for s_i, e_i in zip(starts_u.tolist(), ends_u.tolist()):
                    if e_i - s_i < 2:
                        continue
                    vals = masked_X[b, m, s_i:e_i].clone()
                    if save_fp16:
                        vals = vals.to(torch.float16)
                    cands.append({
                        'b': int(b), 
                        'm': int(m),          # Channel index
                        'channel': int(m),    # Explicit channel field
                        'label': int(target[b]),
                        'start': int(s_i), 
                        'end': int(e_i),
                        'score': 1.0,
                        'values': vals
                    })

        buf.extend(cands)
        total_saved += len(cands)
        if len(buf) >= buffer_limit:
            buf = flush(buf, shard_id)
            shard_id += 1

        # Cleanup
        del seg_d, pad_d, m_flat, logit, probs
        del roi_time_mask, roi_valid, roi_mask_valid, seg, pad_mask, idx_map, x
        del time_mask_flat, time_mask, time_mask_bool, masked_X
        torch.cuda.empty_cache()
        gc.collect()

    if buf:
        buf = flush(buf, shard_id)
        shard_id += 1

    print(f"[Candidate Extraction] Complete!")
    print(f"  Total candidates: {total_saved}")
    print(f"  Total shards: {shard_id}")
    
    return out_dir


# ============================================================================
# Step 2: Extract Global Shapelets (Multivariate with Channel Info)
# ============================================================================

def load_candidates_from_shards_multivariate(
    root_dir: str,
    dataset_name: str,
    pattern: str = "selector_only_shard_*.pt",
    max_candidates: Optional[int] = None
) -> Tuple[List[torch.Tensor], List[int], List[int]]:
    """
    Load shapelet candidates from shard files with channel information.
    
    Returns:
        Tuple of (candidate tensors, labels, channels)
    """
    ds_dir = os.path.join(root_dir, dataset_name)
    shard_paths = sorted(glob.glob(os.path.join(ds_dir, pattern)))
    if not shard_paths:
        raise FileNotFoundError(
            f"No shards matched in {ds_dir} with pattern '{pattern}'"
        )

    values_list = []
    labels_list = []
    channels_list = []
    
    for sp in shard_paths:
        obj = torch.load(sp, map_location="cpu", weights_only=False)
        cands = obj.get("candidates", [])
        for rec in cands:
            values_list.append(rec["values"])
            labels_list.append(rec["label"])
            channels_list.append(rec.get("channel", rec.get("m", 0)))
            if max_candidates is not None and len(values_list) >= max_candidates:
                break
        if max_candidates is not None and len(values_list) >= max_candidates:
            break
    
    print(f"  Loaded {len(values_list)} candidates from {len(shard_paths)} shards")
    print(f"  Unique channels: {sorted(set(channels_list))}")
    return values_list, labels_list, channels_list


def prepare_segments(values_list: List[torch.Tensor], z_norm: bool = True) -> List[np.ndarray]:
    """Convert candidates to 2D numpy arrays and optionally z-normalize"""
    arrs = []
    for v in values_list:
        a = to_2d_np(v).astype(np.float32, copy=False)
        if z_norm:
            a = z_norm_2d(a)
        arrs.append(a)
    return arrs


def build_distance_matrix(arrs: List[np.ndarray], radius: Optional[int] = None) -> np.ndarray:
    """Build pairwise DTW distance matrix"""
    N = len(arrs)
    D = np.zeros((N, N), dtype=np.float32)
    
    print(f"  Computing DTW distance matrix ({N} x {N})...")
    for i in range(N):
        if i % 50 == 0:
            print(f"    Progress: {i}/{N}")
        for j in range(i+1, N):
            D[i, j] = D[j, i] = dtw_distance(arrs[i], arrs[j], radius=radius)
    
    print(f"  Distance matrix computed!")
    return D


def agglomerative_precomputed(D: np.ndarray, n_clusters: int = 20) -> np.ndarray:
    """Perform hierarchical clustering on precomputed distance matrix"""
    try:
        clust = AgglomerativeClustering(
            n_clusters=n_clusters, 
            metric='precomputed', 
            linkage='average'
        )
    except TypeError:
        clust = AgglomerativeClustering(
            n_clusters=n_clusters, 
            affinity='precomputed', 
            linkage='average'
        )
    labels = clust.fit_predict(D)
    return labels


def pick_medoids(D: np.ndarray, labels: np.ndarray, n_clusters: int) -> List[int]:
    """Select medoid for each cluster"""
    medoids = []
    for k in range(n_clusters):
        idx = np.where(labels == k)[0]
        if idx.size == 0:
            continue
        sub = D[np.ix_(idx, idx)]
        sums = sub.sum(axis=1)
        center = idx[int(np.argmin(sums))]
        medoids.append(center)
    return medoids


def choose_k_by_silhouette(
    D: np.ndarray, 
    k_min: int, 
    k_max: int, 
    k_step: int = 1
) -> Tuple[int, List[Tuple[int, float]]]:
    """Choose optimal K using silhouette score"""
    N = D.shape[0]
    ub = max(k_min, min(k_max, N - 1))
    ks = [k for k in range(max(2, k_min), ub + 1, max(1, k_step))]
    if not ks:
        raise ValueError(f"No valid K candidates in range [{k_min}, {k_max}] for N={N}")

    best_k, best_s = None, -np.inf
    scores = []
    
    print(f"  Testing K values: {ks}")
    for k in ks:
        labels = agglomerative_precomputed(D, n_clusters=k)
        if len(np.unique(labels)) < 2:
            continue
        try:
            s = silhouette_score(D, labels, metric='precomputed')
        except Exception as e:
            print(f"    [Warning] K={k} failed: {e}")
            continue
        scores.append((k, s))
        print(f"    K={k}: silhouette={s:.4f}")
        if s > best_s:
            best_s, best_k = s, k

    if best_k is None:
        raise RuntimeError("Failed to compute silhouette for any K.")

    print(f"  Best K={best_k} (silhouette={best_s:.4f})")
    return best_k, scores


def compute_pairwise_similarity(
    shapelets: List[torch.Tensor],
    z_norm: bool = True,
    dtw_radius: Optional[int] = None
) -> np.ndarray:
    """Compute pairwise normalized DTW similarity matrix"""
    arrs = []
    for s in shapelets:
        a = to_2d_np(s).astype(np.float32, copy=False)
        if z_norm:
            a = z_norm_2d(a)
        arrs.append(a)
    
    N = len(arrs)
    sim_matrix = np.zeros((N, N), dtype=np.float32)
    
    print(f"  Computing pairwise similarity ({N} shapelets)...")
    
    for i in range(N):
        if i % 10 == 0:
            print(f"    Progress: {i}/{N}")
        for j in range(i, N):
            if i == j:
                sim_matrix[i, j] = 1.0
            else:
                d = dtw_distance(arrs[i], arrs[j], radius=dtw_radius)
                max_len = max(len(arrs[i]), len(arrs[j]))
                norm_d = d / max_len if max_len > 0 else d
                sim = 1.0 / (1.0 + norm_d)
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim
    
    print(f"  Similarity matrix computed!")
    return sim_matrix


def filter_by_similarity_per_channel(
    cluster_centers: List[torch.Tensor],
    class_labels: List[int],
    channel_labels: List[int],
    similarity_threshold: float = 0.8,
    z_norm: bool = True,
    dtw_radius: Optional[int] = None
) -> Tuple[List[torch.Tensor], List[int], List[int], List[int]]:
    """
    Filter similar shapelets independently within each channel.
    
    For each channel, uses a greedy approach to remove highly similar shapelets.
    Shapelets from different channels are NOT compared with each other.
    
    Args:
        cluster_centers: List of shapelet tensors
        class_labels: List of class labels for each shapelet
        channel_labels: List of channel labels for each shapelet
        similarity_threshold: Threshold above which shapelets are considered similar
        z_norm: Whether to z-normalize for similarity computation
        dtw_radius: DTW radius constraint
    
    Returns:
        Tuple of (filtered_centers, filtered_class_labels, filtered_channel_labels, kept_indices)
    """
    N = len(cluster_centers)
    print(f"\n  [Per-Channel Similarity Filtering] Processing {N} shapelets...")
    
    if N <= 1:
        return cluster_centers, class_labels, channel_labels, list(range(N))
    
    # Group shapelets by channel
    channel_to_indices = defaultdict(list)
    for idx, ch in enumerate(channel_labels):
        channel_to_indices[ch].append(idx)
    
    unique_channels = sorted(channel_to_indices.keys())
    print(f"  Unique channels: {unique_channels}")
    
    kept_indices = []
    
    for channel_id in unique_channels:
        ch_indices = channel_to_indices[channel_id]
        n_ch = len(ch_indices)
        
        print(f"\n    [Channel {channel_id}] Processing {n_ch} shapelets...")
        
        if n_ch <= 1:
            kept_indices.extend(ch_indices)
            print(f"      → Keeping all (only {n_ch} shapelet)")
            continue
        
        # Get shapelets for this channel
        ch_shapelets = [cluster_centers[i] for i in ch_indices]
        
        # Compute similarity matrix for this channel only
        sim_matrix = compute_pairwise_similarity(
            ch_shapelets, z_norm=z_norm, dtw_radius=dtw_radius
        )
        
        # Greedy filtering within this channel
        ch_kept = []
        remaining = set(range(n_ch))
        
        while remaining:
            # Pick the first remaining shapelet
            current = min(remaining)
            ch_kept.append(current)
            remaining.remove(current)
            
            # Remove all shapelets too similar to current (within same channel)
            to_remove = set()
            for other in remaining:
                if sim_matrix[current, other] >= similarity_threshold:
                    to_remove.add(other)
            
            remaining -= to_remove
            if to_remove:
                print(f"      Shapelet {ch_indices[current]} (class {class_labels[ch_indices[current]]}): "
                      f"removed {len(to_remove)} similar shapelets")
        
        # Map local indices back to global indices
        for local_idx in ch_kept:
            kept_indices.append(ch_indices[local_idx])
        
        print(f"      → Kept {len(ch_kept)}/{n_ch} shapelets for Channel {channel_id}")
    
    print(f"\n  → Total kept: {len(kept_indices)}/{N} shapelets")
    
    # Create filtered results (preserve original order by sorting kept_indices)
    kept_indices = sorted(kept_indices)
    filtered_centers = [cluster_centers[i] for i in kept_indices]
    filtered_class_labels = [class_labels[i] for i in kept_indices]
    filtered_channel_labels = [channel_labels[i] for i in kept_indices]
    
    return filtered_centers, filtered_class_labels, filtered_channel_labels, kept_indices


def extract_global_shapelets_multivariate(
    dataset_name: str,
    out_root: str = "shapelet_output_UEA",
    shard_pattern: str = "selector_only_shard_*.pt",
    max_candidates: int = 500,
    auto_select_k: bool = True,
    k_min: int = 5,
    k_max: int = 15,
    k_step: int = 5,
    z_norm_segment: bool = False,
    dtw_radius: Optional[int] = None,
    per_class_clustering: bool = True,
    apply_similarity_filter: bool = True,
    similarity_threshold: float = 0.8,
) -> str:
    """
    Extract global shapelets from multivariate candidates with channel info.
    
    The result is stored in a single file with channel_labels field.
    When per_class_clustering=True, applies global similarity filtering.
    """
    run_dir = Path(out_root) / dataset_name
    
    print(f"[Global Shapelet Extraction - Multivariate] Dataset: {dataset_name}")
    
    # Load candidates with channel information
    values_list, labels_list, channels_list = load_candidates_from_shards_multivariate(
        out_root, dataset_name, shard_pattern, max_candidates
    )
    
    if len(values_list) < 2:
        raise ValueError(f"Not enough candidates: {len(values_list)}")
    
    num_channels = len(set(channels_list))
    print(f"  Total candidates: {len(values_list)}, Channels: {num_channels}")
    
    # Group by class (or treat all as one group)
    if per_class_clustering:
        candidates_by_class = defaultdict(list)
        for idx, (val, lbl, ch) in enumerate(zip(values_list, labels_list, channels_list)):
            candidates_by_class[lbl].append((idx, val, ch))
        print(f"  Mode: Per-Class Clustering")
    else:
        candidates_by_class = {0: [(idx, val, ch) for idx, (val, ch) in enumerate(zip(values_list, channels_list))]}
        print(f"  Mode: Global Clustering")
    
    num_classes = len(candidates_by_class)
    print(f"  Found {num_classes} classes")
    
    # Process each class
    all_cluster_centers = []
    all_class_labels = []
    all_channel_labels = []
    
    for class_id in sorted(candidates_by_class.keys()):
        class_data = candidates_by_class[class_id]
        class_values = [v for _, v, _ in class_data]
        class_channels = [c for _, _, c in class_data]
        N = len(class_values)
        
        print(f"\n  [Class {class_id}] Processing {N} candidates...")
        
        if N < 2:
            print(f"    Skipping (not enough candidates)")
            continue
        
        arrs = prepare_segments(class_values, z_norm=z_norm_segment)
        D = build_distance_matrix(arrs, radius=dtw_radius)
        
        if N <= k_min:
            print(f"    Warning: Only {N} candidates (< k_min={k_min}), using all as shapelets")
            class_centers = [torch.from_numpy(arrs[i]) for i in range(N)]
            medoid_indices = list(range(N))
        else:
            effective_k_max = min(k_max, N - 1)
            if auto_select_k:
                best_k, _ = choose_k_by_silhouette(D, k_min=k_min, k_max=effective_k_max, k_step=k_step)
                n_use = best_k
            else:
                n_use = min(k_min, N - 1)
            
            print(f"    Performing hierarchical clustering (K={n_use})...")
            labels = agglomerative_precomputed(D, n_clusters=n_use)
            
            print(f"    Selecting medoids...")
            medoid_indices = pick_medoids(D, labels, n_use)
            class_centers = [torch.from_numpy(arrs[i]) for i in medoid_indices]
        
        print(f"    Selected {len(class_centers)} shapelets for Class {class_id}")
        
        # Accumulate results with channel info
        all_cluster_centers.extend(class_centers)
        all_class_labels.extend([class_id] * len(class_centers))
        all_channel_labels.extend([class_channels[i] for i in medoid_indices])
    
    # Apply per-channel similarity filtering if enabled
    if apply_similarity_filter and per_class_clustering and len(all_cluster_centers) > 1:
        print(f"\nApplying Per-Channel Similarity Filtering (threshold={similarity_threshold})...")
        all_cluster_centers, all_class_labels, all_channel_labels, kept_indices = filter_by_similarity_per_channel(
            all_cluster_centers, all_class_labels, all_channel_labels,
            similarity_threshold=similarity_threshold,
            z_norm=True,
            dtw_radius=dtw_radius
        )
    else:
        kept_indices = list(range(len(all_cluster_centers)))
    
    total_shapelets = len(all_cluster_centers)
    
    # Generate shapelet IDs (global indexing across all channels)
    shapelet_ids = list(range(total_shapelets))
    
    print(f"\n  Summary:")
    print(f"    Total candidates: {len(values_list)}")
    print(f"    Total classes: {num_classes}")
    print(f"    Total channels: {num_channels}")
    print(f"    Total shapelets: {total_shapelets}")
    for ch in sorted(set(all_channel_labels)):
        count = all_channel_labels.count(ch)
        print(f"      Channel {ch}: {count} shapelets")
    
    # Save with channel info and shapelet IDs
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = run_dir / f"global_shapelets_{total_shapelets}c_{ts}.pt"
    torch.save({
        "dataset": dataset_name,
        "n_candidates": len(values_list),
        "n_clusters": total_shapelets,
        "n_classes": num_classes,
        "n_channels": num_channels,
        "class_labels": all_class_labels,
        "channel_labels": all_channel_labels,     # NEW: channel info
        "shapelet_ids": shapelet_ids,             # NEW: global IDs (0, 1, 2, ...)
        "cluster_centers": all_cluster_centers,
        "params": {
            "per_class_clustering": per_class_clustering,
            "auto_select_k": auto_select_k,
            "k_min": k_min,
            "k_max": k_max,
            "k_step": k_step,
            "z_norm_segment": z_norm_segment,
            "dtw_radius": dtw_radius,
            "apply_similarity_filter": apply_similarity_filter,
            "similarity_threshold": similarity_threshold,
        }
    }, out_path)
    
    print(f"\n[Global Shapelet Extraction] Complete!")
    print(f"  Saved: {out_path}")
    
    return str(out_path)


def check_global_shapelet_exists(
    dataset_name: str,
    base_dir: str = "shapelet_output_UEA"
) -> Optional[str]:
    """Check if global shapelet file exists"""
    pattern = os.path.join(base_dir, dataset_name, "global_shapelets_*.pt")
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def ensure_global_shapelets_multivariate(
    selector,
    loader,
    dataset_name: str,
    out_root: str = "shapelet_output_UEA",
    force_regenerate: bool = False,
    # Step 1 params
    buffer_limit: int = 4000,
    tau: float = 1.3,
    min_len_frac: float = 1/30,
    pen_scale: float = 0.1,
    save_fp16: bool = False,
    algorithm: str = 'pelt',
    # Step 2 params
    max_candidates: int = 500,
    auto_select_k: bool = True,
    k_min: int = 5,
    k_max: int = 15,
    k_step: int = 5,
    z_norm_segment: bool = False,
    dtw_radius: Optional[int] = None,
    per_class_clustering: bool = True,
    apply_similarity_filter: bool = True,
    similarity_threshold: float = 0.8,
) -> str:
    """Ensure global shapelets exist for multivariate dataset"""
    existing = check_global_shapelet_exists(dataset_name, out_root)
    
    if existing and not force_regenerate:
        print(f"Using existing global shapelets: {existing}")
        return existing
    
    if existing and force_regenerate:
        print(f"Force regenerating (existing: {existing})")
    else:
        print(f"No existing global shapelets, extracting...")
    
    # Step 1: Extract candidates
    print(f"\n{'='*60}")
    print(f"Step 1: Extracting Candidates (Multivariate)")
    print(f"{'='*60}")
    run_selector_only_and_save_multivariate(
        selector=selector,
        loader=loader,
        dataset_name=dataset_name,
        out_root=out_root,
        buffer_limit=buffer_limit,
        tau=tau,
        min_len_frac=min_len_frac,
        pen_scale=pen_scale,
        save_fp16=save_fp16,
        algorithm=algorithm,
    )
    
    # Step 2: Extract global shapelets
    print(f"\n{'='*60}")
    print(f"Step 2: Extracting Global Shapelets (Multivariate)")
    print(f"{'='*60}")
    global_path = extract_global_shapelets_multivariate(
        dataset_name=dataset_name,
        out_root=out_root,
        max_candidates=max_candidates,
        auto_select_k=auto_select_k,
        k_min=k_min,
        k_max=k_max,
        k_step=k_step,
        z_norm_segment=z_norm_segment,
        dtw_radius=dtw_radius,
        per_class_clustering=per_class_clustering,
        apply_similarity_filter=apply_similarity_filter,
        similarity_threshold=similarity_threshold,
    )
    
    print(f"\n{'='*60}")
    print(f"Global shapelets ready: {global_path}")
    print(f"{'='*60}\n")
    
    return global_path


# ============================================================================
# Step 3: Dataset Replacement (Multivariate)
# ============================================================================

def load_cluster_centers_multivariate(path: str) -> Dict[str, Any]:
    """Load global shapelets with channel info"""
    obj = torch.load(path, map_location='cpu', weights_only=False)
    return obj


def extract_shapelet_candidates_from_union(masked_X, eps=1e-8):
    """Extract shapelet candidates from masked data (same as original)"""
    B, M, T = masked_X.shape
    candidates, locations = [], []
    mx = masked_X.detach().cpu()
    for b in range(B):
        for m in range(M):
            series = mx[b, m]
            mask_series = (series.abs() > eps)
            in_seg, seg_start = False, 0
            for t in range(T):
                if mask_series[t] and not in_seg:
                    in_seg, seg_start = True, int(t)
                elif (not mask_series[t]) and in_seg:
                    in_seg = False
                    seg_end = int(t)
                    seg = series[seg_start:seg_end].unsqueeze(1)
                    if seg.numel() > 1:
                        candidates.append(seg)
                        locations.append((b, m, seg_start, seg_end))
            if in_seg:
                seg = series[seg_start:].unsqueeze(1)
                if seg.numel() > 1:
                    candidates.append(seg)
                    locations.append((b, m, seg_start, T))
    return candidates, locations


def match_shapelet_by_channel(
    shapelet_query: torch.Tensor,
    cluster_centers: List[torch.Tensor],
    channel_labels: List[int],
    target_channel: int
) -> Tuple[int, float]:
    """
    Match shapelet to the best global shapelet of the same channel.
    """
    # Filter centers by channel
    channel_indices = [i for i, ch in enumerate(channel_labels) if ch == target_channel]
    
    if not channel_indices:
        # Fallback: use all centers if no match for this channel
        channel_indices = list(range(len(cluster_centers)))
    
    query_np = to_2d_np(shapelet_query)
    
    best_idx = -1
    best_dist = float('inf')
    
    for idx in channel_indices:
        center_np = to_2d_np(cluster_centers[idx])
        d, _ = fastdtw(query_np, center_np, dist=euclidean)
        if d < best_dist:
            best_dist = d
            best_idx = idx
    
    return best_idx, best_dist


def apply_dtw_warping(shapelet: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    """Apply DTW warping to match shapelet length"""
    shapelet_np = to_2d_np(shapelet).astype(np.float64)
    center_np = to_2d_np(center).astype(np.float64)
    
    _, path = fastdtw(shapelet_np, center_np, dist=euclidean)
    
    target_len = len(shapelet_np)
    warped = np.zeros(target_len, dtype=np.float64)
    counts = np.zeros(target_len, dtype=np.int32)
    
    for (i_shp, i_ctr) in path:
        warped[i_shp] += center_np[i_ctr, 0]
        counts[i_shp] += 1
    
    counts[counts == 0] = 1
    warped = warped / counts
    
    return torch.from_numpy(warped).float()

def replace_dataset_with_shapelets_multivariate(
    model, 
    data_loader, 
    shapelet_data: Dict[str, Any], 
    device: str
):
    """
    Replace ROI regions with warped global shapelets (multivariate version).
    Each channel is matched to shapelets of the same channel.
    """
    model.eval()
    replaced_data = []
    labels = []
    
    cluster_centers = shapelet_data['cluster_centers']
    channel_labels = shapelet_data.get('channel_labels', [0] * len(cluster_centers))
    
    with torch.no_grad():
        for batch in data_loader:
            # Parse batch
            if isinstance(batch, (list, tuple)):
                if len(batch) >= 4:
                    data, label, sid, mask = batch[:4]
                elif len(batch) >= 2:
                    data, label = batch[:2]
                else:
                    data = batch[0]
                    label = None
            else:
                data = batch
                label = None
            
            # UEA: [B, T, D] -> [B, D, T]
            if data.dim() == 3 and data.shape[1] != data.shape[2]:
                if data.shape[1] > data.shape[2]:
                    data = data.transpose(1, 2)
            
            B, M, T = data.shape
            data_dev = data.to(device)
            
            # Forward pass
            outputs = model(data_dev, training=False, tau=1.3)
            z_tilde = outputs[2]  # ROI
            
            # Extract candidates with locations
            candidates, locations = extract_shapelet_candidates_from_union(z_tilde)
            
            # Initialize with zeros
            data_replaced = torch.zeros_like(data)
            
            # Replace each ROI with warped global shapelet (channel-aware)
            for i, (b, m, start, end) in enumerate(locations):
                candidate = candidates[i]
                
                # Match to global shapelet of same channel
                center_idx, _ = match_shapelet_by_channel(
                    candidate, cluster_centers, channel_labels, target_channel=m
                )
                global_shapelet = cluster_centers[center_idx].squeeze()
                
                # Warp to fit candidate length
                warped_global = apply_dtw_warping(candidate, global_shapelet)
                
                # Replace ROI
                data_replaced[b, m, start:end] = warped_global
            
            # Store (transpose back for saving if needed)
            replaced_data.extend(data_replaced.cpu().numpy())
            if label is not None:
                labels.extend(label.cpu().numpy())
    
    return np.array(replaced_data), np.array(labels) if labels else None


def save_replaced_datasets_multivariate(
    model,
    train_loader,
    val_loader,
    test_loader,
    shapelet_data: Dict[str, Any],
    dataset_name: str,
    save_dir: str = "replaced_datasets_UEA",
    device: str = "cuda:0"
) -> str:
    """Process and save all splits"""
    os.makedirs(save_dir, exist_ok=True)
    dataset_save_dir = os.path.join(save_dir, dataset_name)
    os.makedirs(dataset_save_dir, exist_ok=True)
    
    print(f"\n[Processing] {dataset_name}...")
    
    # Train
    print("   Replacing Train set...")
    train_data, train_labels = replace_dataset_with_shapelets_multivariate(
        model, train_loader, shapelet_data, device
    )
    train_path = os.path.join(dataset_save_dir, "train_replaced.npz")
    np.savez(train_path, data=train_data, labels=train_labels)
    print(f"   Saved train: {train_data.shape} -> {train_path}")
    
    # Val
    print("   Replacing Val set...")
    val_data, val_labels = replace_dataset_with_shapelets_multivariate(
        model, val_loader, shapelet_data, device
    )
    val_path = os.path.join(dataset_save_dir, "val_replaced.npz")
    np.savez(val_path, data=val_data, labels=val_labels)
    print(f"   Saved val: {val_data.shape} -> {val_path}")
    
    # Test
    print("   Replacing Test set...")
    test_data, test_labels = replace_dataset_with_shapelets_multivariate(
        model, test_loader, shapelet_data, device
    )
    test_path = os.path.join(dataset_save_dir, "test_replaced.npz")
    np.savez(test_path, data=test_data, labels=test_labels)
    print(f"   Saved test: {test_data.shape} -> {test_path}")
    
    return dataset_save_dir


# ============================================================================
# Model and Data Loading (UEA)
# ============================================================================

def load_model_and_data_uea(dataset_name: str, args_cli, device: str = "cuda:0"):
    """Load model and data loaders for UEA dataset"""
    uea_root = args_cli.data_root
    dataset_path = os.path.join(uea_root, dataset_name)
    
    # Create temp dataset to get seq_len
    temp_dataset = UEAloader(dataset_path, flag='TRAIN')
    seq_len = temp_dataset.max_seq_len
    num_channels = len(temp_dataset.feature_names)
    num_classes = len(set(temp_dataset.labels_df.values.flatten()))
    
    print(f"  Dataset: {dataset_name}")
    print(f"  seq_len: {seq_len}, channels: {num_channels}, classes: {num_classes}")

    args = SimpleNamespace(
            seed=42,
            data='UEA',                    
            task_name='classification',   
            num_layers=6,
            root_path=dataset_path,      
            seq_len=seq_len,        
            batch_size=32,                  
            num_workers=4,               
            device=device,
            dataset=dataset_name,            
            PRETRAIN_ROOT="./saved_models/pretrain",
            embed='fixed',
            freq='d',
        )
    
    # Create DataLoaders
    train_set, train_loader = data_provider(args, flag='TRAIN')
    val_set, val_loader = data_provider(args, flag='TEST')
    test_set, test_loader = data_provider(args, flag='TEST')
    
    param_dict = {
        "seq_len": seq_len,
        "enc_in": num_channels,
        "num_classes": num_classes
    }
    
    # Init model
    model = MainFlow(
        seq_len=args.seq_len,
        num_channels=num_channels,
        num_layers=args.num_layers,
        num_classes=num_classes,
        device=device,
    ).to(device)
    
    # Load weights
    model_dir = os.path.join(args_cli.model_dir, dataset_name)
    model_path = get_latest_file(os.path.join(model_dir, "UEA_Input_ROI_result_*.pt"))
    
    if not model_path:
        raise FileNotFoundError(f"No UEA model found in {model_dir}")
    
    state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(state_dict)
    model.to(device)
    
    # Freeze selector
    for param in model.selector.parameters():
        param.requires_grad = False
    
    print(f"   Loaded model: {os.path.basename(model_path)}")
    
    return model, train_loader, val_loader, test_loader, param_dict


# ============================================================================
# Single Dataset Processing
# ============================================================================

def process_single_dataset_uea(dataset_name: str, args_cli, gpu_id: int):
    """Process a single UEA dataset through the full pipeline"""
    try:
        device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        
        print(f"\n{'='*70}")
        print(f"[GPU {gpu_id}] Processing: {dataset_name}")
        print(f"{'='*70}")
        
        # Load model and data
        model, train_loader, val_loader, test_loader, param_dict = load_model_and_data_uea(
            dataset_name, args_cli, device=device
        )
        
        # Step 1+2: Ensure global shapelets exist
        print(f"\nChecking global shapelets...")
        global_shapelet_path = ensure_global_shapelets_multivariate(
            selector=model.selector,
            loader=train_loader,
            dataset_name=dataset_name,
            out_root=args_cli.shapelet_output_dir,
            force_regenerate=args_cli.force_regenerate,
            # Step 1 params
            buffer_limit=4000,
            tau=1.3,
            min_len_frac=1/30,
            pen_scale=0.1,
            save_fp16=False,
            algorithm='pelt',
            # Step 2 params
            max_candidates=500,
            auto_select_k=True,
            k_min=5,
            k_max=15,
            k_step=5,
            z_norm_segment=False,
            dtw_radius=None,
            per_class_clustering=args_cli.per_class_clustering,
            apply_similarity_filter=args_cli.apply_similarity_filter,
            similarity_threshold=args_cli.similarity_threshold,
        )
        
        if args_cli.skip_replacement:
            print(f"[{dataset_name}] Global shapelets ready (skipping replacement)")
            return dataset_name
        
        # Step 3: Create replaced dataset
        print(f"\nStep 3: Creating Replaced Dataset")
        print(f"{'='*60}")
        
        shapelet_data = load_cluster_centers_multivariate(global_shapelet_path)
        
        result_dir = save_replaced_datasets_multivariate(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            shapelet_data=shapelet_data,
            dataset_name=f"{dataset_name}_fold{args_cli.fold}",
            save_dir=args_cli.replaced_dataset_dir,
            device=device
        )
        
        print(f"\n[{dataset_name}] Complete!")
        print(f"   Global shapelets: {global_shapelet_path}")
        print(f"   Replaced data: {result_dir}")
        
        return dataset_name
        
    except Exception as e:
        print(f"\n[{dataset_name}] Failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def _worker_wrapper(args_tuple):
    """Wrapper for multiprocessing"""
    dataset_name, args_cli, gpu_id = args_tuple
    return process_single_dataset_uea(dataset_name, args_cli, gpu_id)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified pipeline for multivariate (UEA) shapelet extraction"
    )
    
    # Dataset selection
    parser.add_argument(
        "--dataset", type=str, default="ArticularyWordRecognition",
        help="Dataset name or 'all', 'sub_datasets', comma-separated list"
    )
    
    # Paths
    parser.add_argument(
        "--model_dir", type=str, default="saved_models",
        help="Directory containing trained UEA models"
    )
    parser.add_argument(
        "--data_root", type=str,
        default="./data/Multivariate_ts",
        help="UEA dataset root directory"
    )
    parser.add_argument(
        "--shapelet_output_dir", type=str,
        default="shapelet_output_UEA",
        help="Directory for shapelet outputs"
    )
    parser.add_argument(
        "--replaced_dataset_dir", type=str,
        default="replaced_datasets_UEA",
        help="Directory for replaced datasets"
    )
    
    # Execution options
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--force_regenerate", action="store_true")
    parser.add_argument("--skip_replacement", action="store_true")
    parser.add_argument("--num_workers", type=int, default=None)
    
    # Clustering options
    parser.add_argument("--per_class_clustering", type=bool, default=True)
    parser.add_argument("--apply_similarity_filter", type=bool, default=True)
    parser.add_argument("--similarity_threshold", type=float, default=0.8)
    
    args_cli = parser.parse_args()
    
    # Parse GPU devices
    if ',' in args_cli.device:
        gpu_ids = [int(d.split(':')[1]) for d in args_cli.device.split(',')]
    else:
        gpu_ids = [int(args_cli.device.split(':')[1]) if 'cuda' in args_cli.device else 0]
    
    print(f"\n{'='*70}")
    print(f"Unified Shapelet Pipeline (UEA Multivariate)")
    print(f"{'='*70}")
    print(f"GPUs: {gpu_ids}")
    print(f"Model dir: {args_cli.model_dir}")
    print(f"Data root: {args_cli.data_root}")
    print(f"Shapelet output: {args_cli.shapelet_output_dir}")
    print(f"Per-class clustering: {args_cli.per_class_clustering}")
    print(f"Similarity filter: {args_cli.apply_similarity_filter} (threshold={args_cli.similarity_threshold})")
    print(f"{'='*70}\n")
    
    # Get target datasets
    if args_cli.dataset == "all":
        target_datasets = sorted(os.listdir(args_cli.data_root))
        print(f"Mode: ALL datasets ({len(target_datasets)} total)")
    elif args_cli.dataset == "sub_datasets":
        # Predefined subset for testing
        target_datasets = ["ArticularyWordRecognition", "BasicMotions", "Cricket", "Epilepsy"]
        print(f"Mode: PREDEFINED subset ({len(target_datasets)} datasets)")
    elif ',' in args_cli.dataset:
        target_datasets = [ds.strip() for ds in args_cli.dataset.split(',')]
        print(f"Mode: MULTIPLE datasets ({len(target_datasets)} specified)")
    else:
        target_datasets = [args_cli.dataset]
        print(f"Mode: SINGLE dataset ({args_cli.dataset})")
    
    # Filter datasets with models
    valid_datasets = []
    for dataset_name in target_datasets:
        model_dir = os.path.join(args_cli.model_dir, dataset_name)
        if os.path.exists(model_dir):
            valid_datasets.append(dataset_name)
        else:
            print(f"   Warning: Skipping {dataset_name}: no model directory")
    
    print(f"\nValid datasets with models: {len(valid_datasets)}/{len(target_datasets)}")
    
    if not valid_datasets:
        print("No valid datasets found!")
        return
    
    # Process datasets
    if len(valid_datasets) == 1 or len(gpu_ids) == 1:
        print(f"\nUsing SINGLE-PROCESS mode\n")
        for dataset_name in valid_datasets:
            process_single_dataset_uea(dataset_name, args_cli, gpu_ids[0])
    else:
        print(f"\nUsing MULTI-PROCESS mode with {len(gpu_ids)} GPUs\n")
        worker_args = [
            (ds, args_cli, gpu_ids[i % len(gpu_ids)])
            for i, ds in enumerate(valid_datasets)
        ]
        
        batch_size = len(gpu_ids)
        for i in range(0, len(worker_args), batch_size):
            batch = worker_args[i:i+batch_size]
            print(f"\nProcessing batch {i//batch_size + 1}/{(len(worker_args)-1)//batch_size + 1}")
            
            with Pool(processes=len(batch)) as pool:
                results = pool.map(_worker_wrapper, batch)
            
            successful = [r for r in results if r is not None]
            print(f"   Batch complete: {len(successful)}/{len(batch)} successful")
    
    print(f"\n{'='*70}")
    print(f"Pipeline Complete!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
