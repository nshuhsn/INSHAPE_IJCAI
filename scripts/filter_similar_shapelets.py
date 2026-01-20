"""
Filter Similar Shapelets

Loads global shapelets and filters out similar shapelets based on DTW distance.

Usage:
    python filter_similar_shapelets.py --dataset ECG5000 --threshold 0.8
    python filter_similar_shapelets.py --dataset all --threshold 0.8
"""

import os
import sys
import glob
import argparse
import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


# ============================================================================
# Helper Functions
# ============================================================================

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


def get_latest_file(pattern: str) -> Optional[str]:
    """Get most recent file matching pattern"""
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


# ============================================================================
# Similarity-based Filtering
# ============================================================================

def compute_pairwise_similarity(
    shapelets: List[torch.Tensor],
    z_norm: bool = True,
    dtw_radius: Optional[int] = None
) -> np.ndarray:
    """
    Compute pairwise normalized DTW similarity matrix.
    
    Similarity is computed as: sim = 1 / (1 + normalized_distance)
    where normalized_distance = dtw_distance / max_length
    
    Args:
        shapelets: List of shapelet tensors
        z_norm: Whether to z-normalize before computing distance
        dtw_radius: DTW radius constraint
    
    Returns:
        N x N similarity matrix
    """
    arrs = []
    for s in shapelets:
        a = to_2d_np(s).astype(np.float32, copy=False)
        if z_norm:
            a = z_norm_2d(a)
        arrs.append(a)
    
    N = len(arrs)
    sim_matrix = np.zeros((N, N), dtype=np.float32)
    
    print(f"  Computing pairwise DTW distances ({N} shapelets)...")
    
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


def filter_by_similarity_per_class(
    cluster_centers: List[torch.Tensor],
    class_labels: List[int],
    similarity_threshold: float = 0.8,
    z_norm: bool = True,
    dtw_radius: Optional[int] = None
) -> Tuple[List[torch.Tensor], List[int], List[int]]:
    """
    Filter similar shapelets within each class.
    
    Uses a greedy approach to remove highly similar shapelets.
    
    Args:
        cluster_centers: List of shapelet tensors
        class_labels: List of class labels for each shapelet
        similarity_threshold: Threshold above which shapelets are considered similar
        z_norm: Whether to z-normalize for similarity computation
        dtw_radius: DTW radius constraint
    
    Returns:
        Tuple of (filtered_centers, filtered_labels, kept_indices)
    """
    from collections import defaultdict
    
    class_to_indices = defaultdict(list)
    for idx, lbl in enumerate(class_labels):
        class_to_indices[lbl].append(idx)
    
    kept_indices = []
    
    for class_id in sorted(class_to_indices.keys()):
        indices = class_to_indices[class_id]
        n_class = len(indices)
        
        print(f"\n  [Class {class_id}] Processing {n_class} shapelets...")
        
        if n_class <= 1:
            kept_indices.extend(indices)
            print(f"    -> Keeping all (only {n_class} shapelet)")
            continue
        
        class_shapelets = [cluster_centers[i] for i in indices]
        
        sim_matrix = compute_pairwise_similarity(
            class_shapelets, z_norm=z_norm, dtw_radius=dtw_radius
        )
        
        class_kept = []
        remaining = set(range(n_class))
        
        while remaining:
            current = min(remaining)
            class_kept.append(current)
            remaining.remove(current)
            
            to_remove = set()
            for other in remaining:
                if sim_matrix[current, other] >= similarity_threshold:
                    to_remove.add(other)
            
            remaining -= to_remove
            if to_remove:
                print(f"    Shapelet {current}: removed {len(to_remove)} similar shapelets")
        
        for local_idx in class_kept:
            kept_indices.append(indices[local_idx])
        
        print(f"    -> Kept {len(class_kept)}/{n_class} shapelets")
    
    filtered_centers = [cluster_centers[i] for i in kept_indices]
    filtered_labels = [class_labels[i] for i in kept_indices]
    
    return filtered_centers, filtered_labels, kept_indices


def filter_by_similarity_global(
    cluster_centers: List[torch.Tensor],
    class_labels: List[int],
    similarity_threshold: float = 0.8,
    z_norm: bool = True,
    dtw_radius: Optional[int] = None
) -> Tuple[List[torch.Tensor], List[int], List[int]]:
    """
    Filter similar shapelets globally (across all classes).
    
    Uses a greedy approach to remove highly similar shapelets regardless of class.
    
    Args:
        cluster_centers: List of shapelet tensors
        class_labels: List of class labels for each shapelet
        similarity_threshold: Threshold above which shapelets are considered similar
        z_norm: Whether to z-normalize for similarity computation
        dtw_radius: DTW radius constraint
    
    Returns:
        Tuple of (filtered_centers, filtered_labels, kept_indices)
    """
    N = len(cluster_centers)
    print(f"\n  Processing {N} shapelets globally...")
    
    if N <= 1:
        return cluster_centers, class_labels, list(range(N))
    
    sim_matrix = compute_pairwise_similarity(
        cluster_centers, z_norm=z_norm, dtw_radius=dtw_radius
    )
    
    kept_indices = []
    remaining = set(range(N))
    
    while remaining:
        current = min(remaining)
        kept_indices.append(current)
        remaining.remove(current)
        
        to_remove = set()
        for other in remaining:
            if sim_matrix[current, other] >= similarity_threshold:
                to_remove.add(other)
        
        remaining -= to_remove
        if to_remove:
            print(f"    Shapelet {current} (class {class_labels[current]}): "
                  f"removed {len(to_remove)} similar shapelets")
    
    print(f"  -> Kept {len(kept_indices)}/{N} shapelets")
    
    filtered_centers = [cluster_centers[i] for i in kept_indices]
    filtered_labels = [class_labels[i] for i in kept_indices]
    
    return filtered_centers, filtered_labels, kept_indices


# ============================================================================
# Main Processing
# ============================================================================

def load_global_shapelets(shapelet_path: str) -> dict:
    """Load global shapelets from file"""
    obj = torch.load(shapelet_path, map_location="cpu", weights_only=False)
    return obj


def save_filtered_shapelets(
    original_data: dict,
    filtered_centers: List[torch.Tensor],
    filtered_labels: List[int],
    kept_indices: List[int],
    output_path: str,
    filter_mode: str,
    similarity_threshold: float
):
    """
    Save filtered shapelets in the same format as original.
    
    Args:
        original_data: Original loaded data dict
        filtered_centers: Filtered shapelet tensors
        filtered_labels: Class labels for filtered shapelets
        kept_indices: Indices of kept shapelets (relative to original)
        output_path: Path to save
        filter_mode: 'per_class' or 'global'
        similarity_threshold: Threshold used for filtering
    """
    from collections import Counter
    class_counts = Counter(filtered_labels)
    
    save_data = {
        "dataset": original_data.get("dataset", "unknown"),
        "n_candidates": original_data.get("n_candidates", 0),
        "n_clusters": len(filtered_centers),
        "n_classes": len(set(filtered_labels)),
        "class_labels": filtered_labels,
        "cluster_centers": filtered_centers,
        "medoid_info": [],
        "params": original_data.get("params", {}),
        "filter_info": {
            "filter_mode": filter_mode,
            "similarity_threshold": similarity_threshold,
            "original_n_clusters": original_data.get("n_clusters", 0),
            "kept_indices": kept_indices,
            "shapelets_per_class": dict(class_counts),
        }
    }
    
    torch.save(save_data, output_path)
    print(f"  Saved: {output_path}")


def process_dataset(
    dataset_name: str,
    input_dir: str,
    output_dir: str,
    similarity_threshold: float = 0.8,
    filter_mode: str = "per_class",
    z_norm: bool = True,
    dtw_radius: Optional[int] = None
) -> Optional[str]:
    """
    Process a single dataset: load, filter, and save.
    
    Args:
        dataset_name: Name of dataset
        input_dir: Input directory containing shapelets
        output_dir: Output directory for filtered shapelets
        similarity_threshold: Similarity threshold for filtering
        filter_mode: 'per_class' or 'global'
        z_norm: Whether to z-normalize for similarity computation
        dtw_radius: DTW radius constraint
    
    Returns:
        Output path if successful, None otherwise
    """
    try:
        print(f"\n{'='*70}")
        print(f"Processing: {dataset_name}")
        print(f"{'='*70}")
        
        shapelet_pattern = os.path.join(input_dir, dataset_name, "global_shapelets_*.pt")
        shapelet_path = get_latest_file(shapelet_pattern)
        
        if not shapelet_path:
            print(f"  Warning: No global shapelet file found in {input_dir}/{dataset_name}")
            return None
        
        print(f"  Loading: {os.path.basename(shapelet_path)}")
        
        data = load_global_shapelets(shapelet_path)
        cluster_centers = data.get("cluster_centers", [])
        class_labels = data.get("class_labels", [])
        
        if not cluster_centers:
            print(f"  Warning: No cluster centers found")
            return None
        
        print(f"  Original: {len(cluster_centers)} shapelets, "
              f"{len(set(class_labels))} classes")
        
        print(f"  Filtering (mode={filter_mode}, threshold={similarity_threshold})...")
        
        if filter_mode == "per_class":
            filtered_centers, filtered_labels, kept_indices = filter_by_similarity_per_class(
                cluster_centers, class_labels, similarity_threshold, z_norm, dtw_radius
            )
        else:
            filtered_centers, filtered_labels, kept_indices = filter_by_similarity_global(
                cluster_centers, class_labels, similarity_threshold, z_norm, dtw_radius
            )
        
        print(f"\n  Filtered: {len(filtered_centers)} shapelets "
              f"(removed {len(cluster_centers) - len(filtered_centers)})")
        
        for class_id in sorted(set(filtered_labels)):
            orig_count = class_labels.count(class_id)
            filt_count = filtered_labels.count(class_id)
            print(f"      Class {class_id}: {orig_count} -> {filt_count}")
        
        os.makedirs(os.path.join(output_dir, dataset_name), exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(
            output_dir, dataset_name, 
            f"global_shapelets_filtered_{len(filtered_centers)}c_{ts}.pt"
        )
        
        save_filtered_shapelets(
            data, filtered_centers, filtered_labels, kept_indices,
            out_path, filter_mode, similarity_threshold
        )
        
        return out_path
        
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Filter similar shapelets based on DTW similarity"
    )
    
    parser.add_argument(
        "--dataset", type=str, default="ECG5000",
        help="Dataset name. Options: single name, 'all', 'sub_datasets', or comma-separated list"
    )
    
    parser.add_argument(
        "--input_dir", type=str,
        default="./outputs/shapelets",
        help="Input directory containing global shapelets"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="./outputs/shapelets_filtered",
        help="Output directory for filtered shapelets"
    )
    
    parser.add_argument(
        "--threshold", type=float, default=0.8,
        help="Similarity threshold. Shapelets with similarity >= threshold are considered similar (default: 0.8)"
    )
    parser.add_argument(
        "--mode", type=str, default="global",
        choices=["per_class", "global"],
        help="Filtering mode: 'per_class' filters within each class, 'global' filters across all classes"
    )
    parser.add_argument(
        "--no_znorm", action="store_true",
        help="Disable z-normalization when computing similarity"
    )
    parser.add_argument(
        "--dtw_radius", type=int, default=None,
        help="DTW radius constraint (default: None = no constraint)"
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"Similar Shapelet Filter")
    print(f"{'='*70}")
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Similarity threshold: {args.threshold}")
    print(f"Mode: {args.mode}")
    print(f"Z-normalize: {not args.no_znorm}")
    print(f"DTW radius: {args.dtw_radius}")
    print(f"{'='*70}\n")
    
    if args.dataset == "all":
        target_datasets = sorted([
            d for d in os.listdir(args.input_dir)
            if os.path.isdir(os.path.join(args.input_dir, d))
        ])
        print(f"Mode: ALL datasets ({len(target_datasets)} total)")
    elif args.dataset == "sub_datasets":
        target_datasets = ["WordSynonyms", "CricketX", "EOGVerticalSignal", "InsectWingbeatSound", 
                          "MelbournePedestrian", "Fish", "DistalPhalanxTW", "ProximalPhalanxTW", 
                          "MiddlePhalanxTW", "SyntheticControl", "OSULeaf", "GesturePebbleZ2", 
                          "GesturePebbleZ1", "Symbols", "MixedShapesRegularTrain", "Worms", "Beef"]
        print(f"Mode: PREDEFINED subset ({len(target_datasets)} datasets)")
        print(f"   Datasets: {', '.join(target_datasets)}")
    elif ',' in args.dataset:
        target_datasets = [ds.strip() for ds in args.dataset.split(',')]
        print(f"Mode: MULTIPLE datasets ({len(target_datasets)} specified)")
    else:
        target_datasets = [args.dataset]
        print(f"Mode: SINGLE dataset ({args.dataset})")
    
    results = {"success": [], "failed": []}
    
    for dataset_name in target_datasets:
        result = process_dataset(
            dataset_name=dataset_name,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            similarity_threshold=args.threshold,
            filter_mode=args.mode,
            z_norm=not args.no_znorm,
            dtw_radius=args.dtw_radius
        )
        
        if result:
            results["success"].append(dataset_name)
        else:
            results["failed"].append(dataset_name)
    
    print(f"\n{'='*70}")
    print(f"Summary")
    print(f"{'='*70}")
    print(f"Success: {len(results['success'])}/{len(target_datasets)}")
    if results["failed"]:
        print(f"Failed: {', '.join(results['failed'])}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
