"""
Unified Pipeline for Shapelet Extraction and Dataset Replacement

This script orchestrates the full pipeline:
1. Check if global shapelets exist
2. If not: Extract candidates (Step 1) + Extract global shapelets (Step 2)
3. Create replaced datasets (Step 3)

Supports multi-GPU parallel processing for Step 3.
"""

import sys
import os
import glob
import argparse
from types import SimpleNamespace
from pathlib import Path
from multiprocessing import Pool
from functools import partial

import torch
import numpy as np

# Add paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_provider.UCR_data_factory import ucr_data_provider
from models.INSHAPE import MainFlow
from shapelet_extraction import ensure_global_shapelets
from Input_ROI_DTW_Replacement import (
    load_cluster_centers,
    match_shapelet,
    apply_dtw_warping,
    extract_shapelet_candidates_from_union
)

# ============================================================================
# Dataset Replacement (Step 3)
# ============================================================================

def replace_dataset_with_shapelets(model, data_loader, cluster_centers, device):
    """
    Replace ROI regions in entire dataset with warped global shapelets.
    
    Args:
        model: Trained model with selector
        data_loader: DataLoader
        cluster_centers: List of global shapelet tensors
        device: Device to run on
    
    Returns:
        Tuple of (replaced_data, labels) as numpy arrays
    """
    model.eval()
    replaced_data = []
    labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            # Extract data and labels
            if isinstance(batch, (list, tuple)):
                data, label = batch[0], batch[1]
            else:
                data = batch
                label = None
            
            B = data.shape[0]
            data_dev = data.to(device)
            
            # Forward pass through frozen selector
            outputs = model(data_dev, training=False, tau=1.3)
            z_tilde = outputs[2]  # ROI
            
            # Extract shapelet candidates
            candidates, locations = extract_shapelet_candidates_from_union(z_tilde)
            
            # Initialize with zeros - only replaced ROI regions will have non-zero values
            data_replaced = torch.zeros_like(data)
            
            # Replace each ROI with warped global shapelet
            for i, (b, m, start, end) in enumerate(locations):
                candidate = candidates[i]
                
                # Match to Global Shapelet
                center_idx, _ = match_shapelet(candidate, cluster_centers)
                global_shapelet = cluster_centers[center_idx].squeeze()
                
                # Warp Global Shapelet to fit Candidate
                warped_global = apply_dtw_warping(candidate, global_shapelet)
                
                # Replace ROI in original data
                data_replaced[b, :, start:end] = warped_global.unsqueeze(0)
            
            # Store replaced data
            replaced_data.extend(data_replaced.cpu().numpy())
            if label is not None:
                labels.extend(label.cpu().numpy())
    
    return np.array(replaced_data), np.array(labels) if labels else None


def save_replaced_datasets(
    model,
    train_loader,
    val_loader,
    test_loader,
    cluster_centers,
    dataset_name,
    save_dir="replaced_datasets",
    device="cuda:0"
):
    """
    Process and save all three splits (train/val/test).
    
    Args:
        model: Trained model
        train_loader, val_loader, test_loader: DataLoaders
        cluster_centers: Global shapelets
        dataset_name: Name of dataset
        save_dir: Output directory
        device: Device to use
    
    Returns:
        Path to dataset save directory
    """
    os.makedirs(save_dir, exist_ok=True)
    dataset_save_dir = os.path.join(save_dir, dataset_name)
    os.makedirs(dataset_save_dir, exist_ok=True)
    
    print(f"\n[Processing] {dataset_name}...")
    
    # Process Train
    print("   Replacing Train set...")
    train_data, train_labels = replace_dataset_with_shapelets(
        model, train_loader, cluster_centers, device
    )
    train_path = os.path.join(dataset_save_dir, "train_replaced.npz")
    np.savez(train_path, data=train_data, labels=train_labels)
    print(f"   Saved train: {train_data.shape[0]} samples -> {train_path}")
    
    # Process Val
    print("   Replacing Val set...")
    val_data, val_labels = replace_dataset_with_shapelets(
        model, val_loader, cluster_centers, device
    )
    val_path = os.path.join(dataset_save_dir, "val_replaced.npz")
    np.savez(val_path, data=val_data, labels=val_labels)
    print(f"   Saved val: {val_data.shape[0]} samples -> {val_path}")
    
    # Process Test
    print("   Replacing Test set...")
    test_data, test_labels = replace_dataset_with_shapelets(
        model, test_loader, cluster_centers, device
    )
    test_path = os.path.join(dataset_save_dir, "test_replaced.npz")
    np.savez(test_path, data=test_data, labels=test_labels)
    print(f"   Saved test: {test_data.shape[0]} samples -> {test_path}")
    
    return dataset_save_dir

# ============================================================================
# Model and Data Loading
# ============================================================================

def get_latest_file(pattern):
    """Get most recent file matching pattern"""
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)

def load_model_and_data(dataset_name, args_cli, fold_idx=0, device="cuda:0"):
    """
    Load model and data loaders for a dataset.
    
    Args:
        dataset_name: Name of dataset
        args_cli: Command line arguments
        fold_idx: Which fold to use
        device: Device to load model on
    
    Returns:
        Tuple of (model, train_loader, val_loader, test_loader, param_dict)
    """

    # Setup args for data provider
    args = SimpleNamespace(
        dataset=dataset_name,
        data_root=args_cli.data_root,
        seq_len=None,
        batch_size=128,
        num_workers=4,
        seed=42,
        device=device,
    )

    # Load data
    train_loaders, val_loaders, test_loaders, param_dict = ucr_data_provider(args)
    
    # Use specified fold
    if fold_idx >= len(train_loaders):
        print(f"   Warning: Fold {fold_idx} not available, using fold 0")
        fold_idx = 0
    
    train_loader = train_loaders[fold_idx]
    val_loader = val_loaders[fold_idx]
    test_loader = test_loaders[fold_idx]

    args.seq_len = param_dict['seq_len']
    num_classes = param_dict['num_classes']
    enc_in = param_dict['enc_in']
    
    # Init model
    model = MainFlow(
        seq_len=args.seq_len,
        num_channels=enc_in,
        num_classes=num_classes,
        device=device,
    ).to(device)

    # Load weights
    model_dir = os.path.join(args_cli.model_dir, dataset_name)
    model_path = get_latest_file(os.path.join(model_dir, "best_Input_ROI_result_*.pt"))
    
    if not model_path:
        raise FileNotFoundError(f"No model found in {model_dir}")
    
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

def process_single_dataset(dataset_name, args_cli, gpu_id):
    """
    Process a single dataset through the full pipeline.
    
    Args:
        dataset_name: Name of dataset
        args_cli: Command line arguments
        gpu_id: GPU ID to use
    
    Returns:
        dataset_name if successful, None otherwise
    """
    try:
        device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        
        print(f"\n{'='*70}")
        print(f"[GPU {gpu_id}] Processing: {dataset_name}")
        print(f"{'='*70}")
        
        # Load model and data
        model, train_loader, val_loader, test_loader, param_dict = load_model_and_data(
            dataset_name, args_cli, fold_idx=args_cli.fold, device=device
        )
        
        # Step 1+2: Ensure global shapelets exist
        print(f"\nChecking global shapelets...")
        global_shapelet_path = ensure_global_shapelets(
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
            k_min=5,   # Global: 10-60, Per-class: 10-20
            k_max=15,
            k_step=5,
            z_norm_segment=False,
            dtw_radius=None,
            per_class_clustering=True,  # False=Global, True=Per-class
        )
        
        # If skip_replacement, stop here
        if args_cli.skip_replacement:
            print(f"[{dataset_name}] Global shapelets ready (skipping replacement)")
            return dataset_name
        
        # Step 3: Create replaced dataset
        print(f"\nStep 3: Creating Replaced Dataset")
        print(f"{'='*60}")
        
        cluster_centers = load_cluster_centers(global_shapelet_path)
        
        result_dir = save_replaced_datasets(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            cluster_centers=cluster_centers,
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
    """
    Wrapper function for multiprocessing.
    Required because lambda functions cannot be pickled.
    
    Args:
        args_tuple: Tuple of (dataset_name, args_cli, gpu_id)
    
    Returns:
        Result from process_single_dataset
    """
    dataset_name, args_cli, gpu_id = args_tuple
    return process_single_dataset(dataset_name, args_cli, gpu_id)

# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified pipeline for shapelet extraction and dataset replacement"
    )
    
    # Dataset selection
    parser.add_argument(
        "--dataset", type=str, default="ECG5000",
        help="Dataset name. Options: single name, 'all', 'sub_datasets', or comma-separated list"
    )
    
    # Paths
    parser.add_argument(
        "--model_dir", type=str, default="./saved_models",
        help="Directory containing trained models"
    )
    parser.add_argument(
        "--data_root", type=str,
        default="./data/UCRConverted",
        help="UCR dataset root directory"
    )
    parser.add_argument(
        "--shapelet_output_dir", type=str,
        default="shapelet_output",
        help="Directory for shapelet outputs"
    )
    parser.add_argument(
        "--replaced_dataset_dir", type=str,
        default="replaced_datasets",
        help="Directory for replaced datasets"
    )
    
    # Execution options
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="GPU device(s) to use. Single: 'cuda:0', Multi: 'cuda:0,1,2'"
    )
    parser.add_argument(
        "--fold", type=int, default=0,
        help="Which fold to process (default: 0)"
    )
    parser.add_argument(
        "--force_regenerate", action="store_true",
        help="Force regenerate global shapelets even if they exist"
    )
    parser.add_argument(
        "--skip_replacement", action="store_true",
        help="Only extract shapelets, skip dataset replacement (Step 3)"
    )
    parser.add_argument(
        "--num_workers", type=int, default=None,
        help="Number of parallel workers (default: number of GPUs)"
    )
    
    args_cli = parser.parse_args()
    
    # Parse GPU devices
    if ',' in args_cli.device:
        # Multiple GPUs specified
        gpu_ids = [int(d.split(':')[1]) for d in args_cli.device.split(',')]
    else:
        # Single GPU
        gpu_ids = [int(args_cli.device.split(':')[1]) if 'cuda' in args_cli.device else 0]
    
    print(f"\n{'='*70}")
    print(f"Unified Shapelet Pipeline")
    print(f"{'='*70}")
    print(f"GPUs: {gpu_ids}")
    print(f"Model dir: {args_cli.model_dir}")
    print(f"Data root: {args_cli.data_root}")
    print(f"Shapelet output: {args_cli.shapelet_output_dir}")
    print(f"Replaced dataset output: {args_cli.replaced_dataset_dir}")
    print(f"Force regenerate: {args_cli.force_regenerate}")
    print(f"Skip replacement: {args_cli.skip_replacement}")
    print(f"{'='*70}\n")
    
    # Get target datasets
    if args_cli.dataset == "all":
        # Process all datasets in data_root
        target_datasets = sorted(os.listdir(args_cli.data_root))
        print(f"Mode: ALL datasets ({len(target_datasets)} total)")
    elif args_cli.dataset == "sub_datasets":
        # Predefined subset
        target_datasets = ["WordSynonyms", "CricketX", "EOGVerticalSignal", "InsectWingbeatSound", "MelbournePedestrian", "Fish", "DistalPhalanxTW", "ProximalPhalanxTW", "MiddlePhalanxTW", "SyntheticControl", "OSULeaf", "GesturePebbleZ2", "GesturePebbleZ1", "Symbols", "MixedShapesRegularTrain", "Worms", "Beef"]    

        print(f"Mode: PREDEFINED subset ({len(target_datasets)} datasets)")
        print(f"   Datasets: {', '.join(target_datasets)}")
    elif ',' in args_cli.dataset:
        # Multiple datasets specified (comma-separated)
        target_datasets = [ds.strip() for ds in args_cli.dataset.split(',')]
        print(f"Mode: MULTIPLE datasets ({len(target_datasets)} specified)")
        print(f"   Datasets: {', '.join(target_datasets)}")
    else:
        # Single dataset
        target_datasets = [args_cli.dataset]
        print(f"Mode: SINGLE dataset")
        print(f"   Dataset: {args_cli.dataset}")
    
    # Filter datasets that have models
    valid_datasets = []
    for dataset_name in target_datasets:
        model_dir = os.path.join(args_cli.model_dir, dataset_name)
        print(model_dir)
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
        # Single-process mode
        print(f"\nUsing SINGLE-PROCESS mode\n")
        for dataset_name in valid_datasets:
            process_single_dataset(dataset_name, args_cli, gpu_ids[0])
    else:
        # Multi-process mode
        print(f"\nUsing MULTI-PROCESS mode with {len(gpu_ids)} GPUs\n")
        
        # Assign datasets to GPUs in round-robin fashion
        # Create tuples of (dataset_name, args_cli, gpu_id) for worker
        worker_args = [
            (ds, args_cli, gpu_ids[i % len(gpu_ids)])
            for i, ds in enumerate(valid_datasets)
        ]
        
        # Process in batches equal to number of GPUs to avoid GPU memory issues
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
