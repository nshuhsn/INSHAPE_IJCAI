"""
DTW-based Shapelet Replacement Module

Replace ROI regions with DTW-warped global shapelets.
"""

import os
import glob
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


def extract_shapelet_candidates_from_union(masked_X, eps=1e-8):
    """Extract shapelet candidates from masked data."""
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


def load_cluster_centers(path):
    """Load cluster centers from file."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    centers = None
    if isinstance(obj, list):
        centers = obj
    elif isinstance(obj, dict):
        if "cluster_centers" in obj:
            centers = obj["cluster_centers"]
    if centers is None:
        raise ValueError(f"Unrecognized format for cluster centers")
    centers = [c.detach().cpu() if isinstance(c, torch.Tensor) else torch.as_tensor(c) for c in centers]
    centers = [c if c.ndim == 2 else c.unsqueeze(1) for c in centers]
    return centers


def match_shapelet(shapelet_query, cluster_centers):
    """Match shapelet to closest cluster center using DTW."""
    best_idx, best_dist = -1, float("inf")
    q = shapelet_query.numpy()
    for idx, center in enumerate(cluster_centers):
        d, _ = fastdtw(q, center.numpy(), dist=euclidean)
        if d < best_dist:
            best_dist, best_idx = d, idx
    return best_idx, best_dist


def apply_dtw_warping(shapelet, center):
    """
    Warp cluster center to match shapelet length using DTW path.
    
    - Compression: Multiple center points mapped to one shapelet point -> Average
    - Stretching: One center point mapped to multiple shapelet points -> Repeat
    """
    if isinstance(shapelet, torch.Tensor):
        shapelet = shapelet.detach().cpu().numpy()
    if isinstance(center, torch.Tensor):
        center = center.detach().cpu().numpy()
        
    shapelet = shapelet.squeeze()
    center = center.squeeze()
    
    dist, path = fastdtw(shapelet.reshape(-1, 1), center.reshape(-1, 1), dist=euclidean)
    
    len_s = len(shapelet)
    warped_center = np.zeros(len_s)
    
    map_dict = {i: [] for i in range(len_s)}
    for s_idx, c_idx in path:
        if s_idx < len_s:
            map_dict[s_idx].append(center[c_idx])
            
    for i in range(len_s):
        vals = map_dict[i]
        if vals:
            warped_center[i] = np.mean(vals)
        else:
            if i > 0:
                warped_center[i] = warped_center[i-1]
            else:
                warped_center[i] = center[0]
                
    return torch.tensor(warped_center, dtype=torch.float32)


def save_selected_samples_with_replacement(
    model, 
    test_loader, 
    target_indices=None, 
    centers_path="./outputs/shapelets/global_shapelets.pt",
    save_dir="saved_results",
    save_filename="selected_samples_with_replacement.pt"
):
    """
    Save selected samples with ROI regions replaced by warped global shapelets.
    
    Args:
        model: Trained model
        test_loader: DataLoader for test data
        target_indices: Indices of samples to process (default: first 4)
        centers_path: Path to global shapelets file
        save_dir: Directory to save results
        save_filename: Output filename
    
    Returns:
        Path to saved file
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, save_filename)

    if not os.path.exists(centers_path):
        dirname = os.path.dirname(centers_path)
        if os.path.exists(dirname):
            candidates = [f for f in os.listdir(dirname) if f.startswith("global_shapelets") and f.endswith(".pt")]
            if candidates:
                centers_path = os.path.join(dirname, candidates[0])
                print(f"Warning: Default centers file not found. Using: {centers_path}")
            else:
                print(f"Error: No centers file found in {dirname}")
                return
        else:
            print(f"Error: Directory not found {dirname}")
            return

    cluster_centers = load_cluster_centers(centers_path)
    
    if target_indices is None:
        target_set = set(range(4))
    else:
        target_set = set(target_indices)

    saved_data_list = []
    global_idx = 0
    collected_count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if isinstance(batch, (list, tuple)):
                data = batch[0]
                target = batch[1] if len(batch) > 1 else None
            else:
                data = batch
                target = None

            B = data.shape[0]
            batch_start_idx = global_idx
            
            batch_indices_to_process = []
            for b in range(B):
                if (batch_start_idx + b) in target_set:
                    batch_indices_to_process.append(b)
            
            if batch_indices_to_process:
                data_dev = data.to(next(model.parameters()).device)
                
                outputs = model(data_dev, training=False, tau=1.3)
                
                if len(outputs) >= 7:
                    pred, m, z_tilde, x_hat, logit, x, probs = outputs[:7]
                    masked_X = z_tilde 
                else:
                    z_tilde = outputs[2]
                    masked_X = z_tilde

                candidates, locations = extract_shapelet_candidates_from_union(masked_X)

                for b in batch_indices_to_process:
                    real_idx = batch_start_idx + b
                    
                    orig_signal = data[b].detach().cpu()
                    if orig_signal.ndim > 1: orig_signal = orig_signal.squeeze()

                    z_tilde_sample = z_tilde[b].squeeze().detach().cpu()
                    z_tilde_nan = z_tilde_sample.masked_fill(z_tilde_sample == 0, float('nan'))

                    replaced_partial = torch.full_like(orig_signal, float('nan'))

                    for i, (bb, mm, start, end) in enumerate(locations):
                        if bb != b: continue
                        
                        shapelet = candidates[i]
                        center_idx, _ = match_shapelet(shapelet, cluster_centers)
                        rep = cluster_centers[center_idx].squeeze() 

                        warped_rep = apply_dtw_warping(shapelet, rep)
                        
                        L = len(warped_rep)
                        if L <= 1: continue

                        valid_len = min(L, len(replaced_partial) - start)
                        
                        if valid_len > 0:
                             replaced_partial[start : start+valid_len] = warped_rep[:valid_len]

                    current_label = target[b].item() if target is not None else None
                    current_pred = pred[b].argmax().item() if pred is not None else None

                    sample_dict = {
                        "global_index": real_idx,
                        "original": orig_signal,
                        "reconstructed": z_tilde_nan,
                        "replaced": replaced_partial, 
                        "label": current_label,
                        "prediction": current_pred
                    }
                    saved_data_list.append(sample_dict)
                    collected_count += 1

            global_idx += B
            if collected_count >= len(target_set):
                break
    
    final_save_content = {
        "samples": saved_data_list,
        "target_indices": target_indices
    }
    torch.save(final_save_content, save_path)
    print(f"Saved: {save_path} ({len(saved_data_list)} samples, replaced regions only)")
    
    model.train()
    return save_path