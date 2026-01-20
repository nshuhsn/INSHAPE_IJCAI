"""
Shapelet Extraction Module

Provides functions for:
1. Extracting shapelet candidates from ROI masks
2. Clustering candidates via DTW to extract global shapelets
3. Automatic detection and generation of global shapelets
"""

import os
import gc
import glob
import datetime
import sys
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from models.ROI_search import segment, pack_valid_roi_fast

# Default output directory (relative to project root)
DEFAULT_OUTPUT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs', 'shapelets'))

# ============================================================================
# Step 1: Extract Shapelet Candidates from Selector
# ============================================================================
@torch.inference_mode()
def run_selector_only_and_save(
    selector,
    loader,
    dataset_name: str,
    out_root: str = None,
    buffer_limit: int = 4000,
    tau: float = 1.3,
    min_len_frac: float = 1/30,
    pen_scale: float = 0.1,
    save_fp16: bool = False,
    algorithm: str = 'pelt',
    n_jobs_pelt: int = -1,
) -> str:
    """
    Run selector model to extract shapelet candidates and save to shard files.
    
    Args:
        selector: Trained selector model
        loader: DataLoader for input data
        dataset_name: Name of dataset
        out_root: Root directory for output
        buffer_limit: Number of candidates per shard file
        tau: Temperature for Gumbel-Softmax
        min_len_frac: Minimum segment length as fraction of sequence length
        pen_scale: Penalty scale for segmentation
        save_fp16: Whether to save in fp16 to reduce file size
        algorithm: Segmentation algorithm ('pelt' or 'spline')
        n_jobs_pelt: Number of parallel jobs for PELT
    
    Returns:
        Output directory path
    """
    if out_root is None:
        out_root = DEFAULT_OUTPUT_ROOT
    
    device = next(selector.parameters()).device
    out_dir = os.path.join(out_root, dataset_name)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"[Candidate Extraction] Dataset: {dataset_name}")
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
                'm': int(s['m']),
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
        if isinstance(batch, (list, tuple)):
            if len(batch) == 3:
                data, target, padding_mask = batch
            else:
                data, target = batch[:2]
                padding_mask = None
        else:
            data = batch
            padding_mask = None
        
        padding_mask = None

        x = data.detach().to('cpu').contiguous()
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B,1,T]
        B, M, T = x.shape

        eps = 1e-6
        x = x + eps
        if padding_mask is not None:
            pm = padding_mask.to('cpu')
            x = x * pm.unsqueeze(1)

        min_len = max(1, int(T * min_len_frac))
        pen0 = float(pen_scale * np.log(T))
        roi_time_mask, roi_valid, L_max = segment(
            x, min_len=min_len, pen=pen0, algorithm=algorithm
        )

        if roi_time_mask.dtype != torch.bool:
            roi_time_mask = roi_time_mask.bool()

        seg, pad_mask, idx_map = pack_valid_roi_fast(x, roi_time_mask, roi_valid, L_max)

        seg_d = seg.to(device, non_blocking=True)
        pad_d = pad_mask.to(device, non_blocking=True)
        m_flat, logit, probs = selector(seg_d, pad_d, tau=tau, training=False)
        m_flat = m_flat.detach().to('cpu').view(-1, 1)

        roi_mask_valid = roi_time_mask[roi_valid]
        selected_ROI = roi_mask_valid.to(torch.float32) * m_flat

        rows = list(idx_map.keys())
        b_idx, m_idx, _ = map(torch.tensor, zip(*rows))
        b_idx = b_idx.to(x.device)
        m_idx = m_idx.to(x.device)

        BM = B * M
        flat_id = b_idx * M + m_idx

        time_mask_flat = torch.zeros(BM, T, dtype=selected_ROI.dtype)
        time_mask_flat.index_add_(0, flat_id, selected_ROI)

        time_mask = time_mask_flat.view(B, M, T)
        masked_X = x * time_mask
        time_mask_bool = (time_mask > 0)

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
                        'b': int(b), 'm': int(m),
                        'label': int(target[b]),
                        'start': int(s_i), 'end': int(e_i),
                        'score': 1.0,
                        'values': vals
                    })

        buf.extend(cands)
        total_saved += len(cands)
        if len(buf) >= buffer_limit:
            buf = flush(buf, shard_id)
            shard_id += 1

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
# Step 2: Extract Global Shapelets via DTW Clustering
# ============================================================================

def to_2d_np(x):
    """Convert Tensor/ndarray [L] or [L,F] -> np.ndarray [L,F]"""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x)
    return arr[:, None] if arr.ndim == 1 else arr


def z_norm_2d(arr: np.ndarray):
    """Z-normalize 2D array along time axis"""
    mu = arr.mean(axis=0, keepdims=True)
    sd = arr.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    return (arr - mu) / sd


def load_candidates_from_shards(
    root_dir: str,
    dataset_name: str,
    pattern: str = "selector_only_shard_*.pt",
    max_candidates: Optional[int] = None
) -> Tuple[List[torch.Tensor], List[int]]:
    """
    Load shapelet candidates from shard files.
    
    Args:
        root_dir: Root directory containing dataset folders
        dataset_name: Name of dataset
        pattern: Glob pattern for shard files
        max_candidates: Maximum number of candidates to load
    
    Returns:
        Tuple of (candidate tensors, labels)
    """
    ds_dir = os.path.join(root_dir, dataset_name)
    shard_paths = sorted(glob.glob(os.path.join(ds_dir, pattern)))
    if not shard_paths:
        raise FileNotFoundError(
            f"No shards matched in {ds_dir} with pattern '{pattern}'"
        )

    values_list = []
    labels_list = []
    for sp in shard_paths:
        obj = torch.load(sp, map_location="cpu", weights_only=False)
        cands = obj.get("candidates", [])
        for rec in cands:
            values_list.append(rec["values"])
            labels_list.append(rec["label"])
            if max_candidates is not None and len(values_list) >= max_candidates:
                break
        if max_candidates is not None and len(values_list) >= max_candidates:
            break
    
    print(f"  Loaded {len(values_list)} candidates from {len(shard_paths)} shards")
    return values_list, labels_list


def prepare_segments(values_list: List[torch.Tensor], z_norm: bool = True) -> List[np.ndarray]:
    """Convert candidates to 2D numpy arrays and optionally z-normalize"""
    arrs = []
    for v in values_list:
        a = to_2d_np(v).astype(np.float32, copy=False)
        if z_norm:
            a = z_norm_2d(a)
        arrs.append(a)
    return arrs


def dtw_distance(a: np.ndarray, b: np.ndarray, radius: Optional[int] = None) -> float:
    """Compute DTW distance between two time series"""
    if radius is None:
        d, _ = fastdtw(a, b, dist=euclidean)
    else:
        d, _ = fastdtw(a, b, dist=euclidean, radius=radius)
    return d


def build_distance_matrix(arrs: List[np.ndarray], radius: Optional[int] = None) -> np.ndarray:
    """
    Build pairwise DTW distance matrix.
    
    Args:
        arrs: List of 2D numpy arrays
        radius: DTW radius constraint
    
    Returns:
        N x N distance matrix
    """
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
    """
    Select medoid (most central point) for each cluster.
    
    Args:
        D: Distance matrix
        labels: Cluster labels
        n_clusters: Number of clusters
    
    Returns:
        List of medoid indices
    """
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
    """
    Choose optimal K using silhouette score.
    
    Args:
        D: Precomputed distance matrix
        k_min: Minimum K to test
        k_max: Maximum K to test
        k_step: Step size for K
    
    Returns:
        Tuple of (best_k, list of (k, score) pairs)
    """
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


def extract_global_shapelets_from_shards(
    dataset_name: str,
    out_root: str = None,
    shard_pattern: str = "selector_only_shard_*.pt",
    max_candidates: int = 500,
    auto_select_k: bool = True,
    k_min: int = 10,
    k_max: int = 60,
    k_step: int = 10,
    z_norm_segment: bool = False,
    dtw_radius: Optional[int] = None,
    per_class_clustering: bool = True,
) -> str:
    """
    Extract global shapelets from candidate shards via class-specific DTW clustering.
    
    Args:
        dataset_name: Name of dataset
        out_root: Root directory for shapelets
        shard_pattern: Pattern for shard files
        max_candidates: Max candidates to load
        auto_select_k: Whether to auto-select K via silhouette
        k_min: Minimum K for silhouette search (per class)
        k_max: Maximum K for silhouette search (per class)
        k_step: Step size for K search
        z_norm_segment: Whether to z-normalize segments
        dtw_radius: DTW radius constraint
        per_class_clustering: Whether to cluster per class
    
    Returns:
        Path to saved global_shapelets_*.pt file
    """
    from collections import defaultdict
    
    if out_root is None:
        out_root = DEFAULT_OUTPUT_ROOT
    
    run_dir = Path(out_root) / dataset_name
    
    print(f"[Global Shapelet Extraction] Dataset: {dataset_name}")
    
    values_list, labels_list = load_candidates_from_shards(
        out_root, dataset_name, shard_pattern, max_candidates
    )
    
    if len(values_list) < 2:
        raise ValueError(f"Not enough candidates: {len(values_list)}")
    
    if per_class_clustering:
        candidates_by_class = defaultdict(list)
        for val, lbl in zip(values_list, labels_list):
            candidates_by_class[lbl].append(val)
        print(f"  Mode: Per-Class Clustering")
    else:
        candidates_by_class = {0: values_list}
        print(f"  Mode: Global Clustering")
    
    num_classes = len(candidates_by_class)
    print(f"  Found {num_classes} {'classes' if per_class_clustering else 'group'}")
    for class_id in sorted(candidates_by_class.keys()):
        print(f"    {'Class' if per_class_clustering else 'Group'} {class_id}: {len(candidates_by_class[class_id])} candidates")
    
    all_cluster_centers = []
    all_class_labels = []
    all_medoid_info = []
    
    for class_id in sorted(candidates_by_class.keys()):
        class_values = candidates_by_class[class_id]
        
        print(f"\n  [Class {class_id}] Processing {len(class_values)} candidates...")
        
        arrs = prepare_segments(class_values, z_norm=z_norm_segment)
        N = len(arrs)
        
        if N < 2:
            print(f"    Warning: Skipping (not enough candidates)")
            continue
        
        D = build_distance_matrix(arrs, radius=dtw_radius)
        
        if N <= k_min:
            print(f"    Warning: Only {N} candidates (< k_min={k_min}), using all as shapelets")
            class_centers = [torch.from_numpy(arrs[i]) for i in range(N)]
            n_use = N
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
        
        all_cluster_centers.extend(class_centers)
        all_class_labels.extend([class_id] * len(class_centers))
        all_medoid_info.append({
            'class_id': class_id,
            'n_candidates': N,
            'n_clusters': n_use,
            'medoid_indices': list(range(N)) if N <= k_min else medoid_indices
        })
    
    cluster_centers = all_cluster_centers
    total_shapelets = len(cluster_centers)
    
    print(f"\n  Summary:")
    print(f"    Total candidates: {len(values_list)}")
    print(f"    Total classes: {num_classes}")
    print(f"    Total shapelets: {total_shapelets}")
    for class_id in sorted(set(all_class_labels)):
        count = all_class_labels.count(class_id)
        print(f"      Class {class_id}: {count} shapelets")
    
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = run_dir / f"global_shapelets_{total_shapelets}c_{ts}.pt"
    torch.save({
        "dataset": dataset_name,
        "n_candidates": len(values_list),
        "n_clusters": total_shapelets,
        "n_classes": num_classes,
        "class_labels": all_class_labels,
        "cluster_centers": cluster_centers,
        "medoid_info": all_medoid_info,
        "params": {
            "per_class_clustering": per_class_clustering,
            "auto_select_k": auto_select_k,
            "k_min": k_min,
            "k_max": k_max,
            "k_step": k_step,
            "z_norm_segment": z_norm_segment,
            "dtw_radius": dtw_radius,
        }
    }, out_path)
    
    print(f"\n[Global Shapelet Extraction] Complete!")
    print(f"  Saved: {out_path}")
    
    return str(out_path)


# ============================================================================
# Orchestration: Ensure Global Shapelets Exist
# ============================================================================

def check_global_shapelet_exists(
    dataset_name: str,
    base_dir: str = None
) -> Optional[str]:
    """
    Check if global shapelet file exists for dataset.
    
    Args:
        dataset_name: Name of dataset
        base_dir: Base directory for shapelets
    
    Returns:
        Path to latest global_shapelets_*.pt if exists, None otherwise
    """
    if base_dir is None:
        base_dir = DEFAULT_OUTPUT_ROOT
    
    pattern = os.path.join(base_dir, dataset_name, "global_shapelets_*.pt")
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def ensure_global_shapelets(
    selector,
    loader,
    dataset_name: str,
    out_root: str = None,
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
    k_min: int = 10,
    k_max: int = 60,
    k_step: int = 10,
    z_norm_segment: bool = False,
    dtw_radius: Optional[int] = None,
    per_class_clustering: bool = True,
) -> str:
    """
    Ensure global shapelets exist. If not, extract them.
    
    Orchestrates Step 1 (extract candidates) and Step 2 (clustering).
    
    Args:
        selector: Trained selector model
        loader: DataLoader
        dataset_name: Name of dataset
        out_root: Output root directory
        force_regenerate: Force regeneration even if exists
        buffer_limit: Candidates per shard
        tau: Temperature for Gumbel-Softmax
        min_len_frac: Min segment length fraction
        pen_scale: Penalty scale
        save_fp16: Save in fp16
        algorithm: 'pelt' or 'spline'
        max_candidates: Max candidates to load
        auto_select_k: Auto-select K via silhouette
        k_min, k_max, k_step: K search range
        z_norm_segment: Z-normalize segments
        dtw_radius: DTW radius constraint
        per_class_clustering: Whether to cluster per class
    
    Returns:
        Path to global_shapelets_*.pt file
    """
    if out_root is None:
        out_root = DEFAULT_OUTPUT_ROOT
    
    existing = check_global_shapelet_exists(dataset_name, out_root)
    
    if existing and not force_regenerate:
        print(f"Using existing global shapelets: {existing}")
        return existing
    
    if existing and force_regenerate:
        print(f"Force regenerating (existing: {existing})")
    else:
        print(f"No existing global shapelets, extracting...")
    
    print(f"\n{'='*60}")
    print(f"Step 1: Extracting Candidates")
    print(f"{'='*60}")
    run_selector_only_and_save(
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
    
    print(f"\n{'='*60}")
    print(f"Step 2: Extracting Global Shapelets")
    print(f"{'='*60}")
    global_path = extract_global_shapelets_from_shards(
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
    )
    
    print(f"\n{'='*60}")
    print(f"Global shapelets ready: {global_path}")
    print(f"{'='*60}\n")
    
    return global_path