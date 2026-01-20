"""
UCR Training Script

Train INSHAPE model on UCR univariate time series datasets.
"""

import argparse
import csv
from datetime import datetime
import json
import glob, os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import torch
from tqdm import tqdm
import wandb

from data_provider.UCR_data_factory import ucr_data_provider, ucr_data_provider_originalSplit
from models.INSHAPE import MainFlow
import psutil
from trainer.train_loop import test, train, valid
from utils.util import set_seed, setup_optimizers

def get_args():
    parser = argparse.ArgumentParser(description="Training configuration")
    parser.add_argument('--lambda_1', type=float, default=3, help='lambda_1 value (default: 3)')
    parser.add_argument('--num_epochs', type=float, default=1000, help='Number of epochs (default: 1000)')
    parser.add_argument('--use_default_param', action='store_true', help='Use default parameters from config file')
    parser.add_argument('--ablation', action='store_true', help='Run only ablation datasets')
    parser.add_argument('--use_warmup', action='store_true', help='Use warmup scheduler')
    parser.add_argument('--train_subdata', action='store_true', help='Run only sub datasets')
    parser.add_argument('--cpu_start', type=float, default=39, help='CPU affinity start')
    parser.add_argument('--use_cpu_num', type=float, default=40, help='Number of CPUs to use')
    parser.add_argument('--use_original_split', action='store_true', help='Use original train/test split')
    return parser.parse_args()

def load_dataset_config(filepath):
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Config file not found: {filepath} (using default values)")
        return {}
    except json.JSONDecodeError:
        print(f"Config file JSON error: {filepath} (using default values)")
        return {}


def run_single_fold(args, train_loader, val_loader, test_loader, param_dict, use_warmup, save_dir=None, fold_idx=None):
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    args.seq_len = param_dict['seq_len']
    num_classes = param_dict['num_classes']
    enc_in = param_dict['enc_in']
    
    model = MainFlow(
        seq_len=args.seq_len,
        num_channels=enc_in,
        num_layers=args.num_layers,
        num_classes=num_classes,
        device=device,
    ).to(device)

    # Load pretrained weights if available
    pt_pattern = os.path.join(args.PRETRAIN_ROOT, args.dataset,
                            f"{fold_idx}_best_result_*.pt")
    matches = glob.glob(pt_pattern)

    if matches:
        pt_path = sorted(matches)[-1]
        print(f"Loading pre-trained predictor: {pt_path}")
        orig_sd = torch.load(pt_path, map_location=device)

        incompatible = model.load_state_dict(orig_sd, strict=False)
        missing = set(incompatible.missing_keys)
        unexpected = set(incompatible.unexpected_keys)

        print(f"  Predictor weight tensors: {len(orig_sd)}")

        if missing:
            print(f"  {len(missing)} keys skipped (shape mismatch or absent)")
    else:
        print("  No pre-train weights - training from scratch")

    optimizers, schedulers = setup_optimizers(model, train_loader, args)

    def num_params(generator):
        return sum(p.numel() for p in generator if p.requires_grad == True)
    
    print(f"lambda_1: {wandb.config.lambda_1}")

    print(f"Dataset: {args.dataset}, Sequence Length: {args.seq_len}, Channels: {enc_in}, Classes: {num_classes}")
    print(f"Train: {len(train_loader.dataset)}, Valid: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
    
    print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"Selector parameters: {num_params(model.selector.parameters())}")
    print(f"Predictor parameters: {num_params(model.predictor.parameters())}")

    best_model, _, _, _ = train(model, optimizers, schedulers, train_loader, val_loader, args, use_warmup=use_warmup, use_Lc=False, is_uea=False)

    val_acc, val_sel, val_Lc = valid(best_model, val_loader, is_uea=False, tau=1.3)
    test_acc, test_sel, test_Lc, avg_segment_count = test(best_model, test_loader, is_uea=False, tau=1.3) 

    # Save model
    if save_dir is not None and fold_idx == 1:
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now(ZoneInfo("Asia/Seoul")).strftime('%Y%m%d_%H%M%S')
        model_path = os.path.join(save_dir, f"best_Input_ROI_result_{timestamp}.pt")
        best_model.cpu()
        torch.save(best_model.state_dict(), model_path)
        best_model.to(device) 
        print(f"Saved best model for fold {fold_idx} -> {model_path}")

    return val_acc, val_sel, test_acc, test_sel, val_Lc, test_Lc, avg_segment_count

def main():
    args_cli = get_args()
    
    ucr_root = './data/UCRConverted'

    p = psutil.Process()
    p.cpu_affinity(range(int(args_cli.cpu_start), int(args_cli.cpu_start + args_cli.use_cpu_num)))

    if args_cli.ablation:
        print('Running Ablation datasets in UCR')
        dataset_names = ['ArrowHead', 'CBF', 'ECG5000', 'DistalPhalanxOutlineAgeGroup','DistalPhalanxOutlineCorrect', 
                        'EOGVerticalSignal', 'EthanolLevel', 'Fish', 'GunPoint', 'InsectWingbeatSound', 
                        'ItalyPowerDemand', 'MelbournePedestrian', 'MiddlePhalanxTW', 'MixedShapesRegularTrain', 
                        'OSULeaf', 'Trace', 'WordSynonyms']
    elif args_cli.train_subdata:
        print('Running subdatasets in UCR')
        dataset_names = ['CBF', 'ECG5000']
    else:
        # Exclude ablation datasets
        ablation_exclude_list = ['ArrowHead', 'CBF', 'CricketX', 'DistalPhalanxOutlineAgeGroup',
                                'DistalPhalanxOutlineCorrect', 'ECG5000', 'EOGVerticalSignal', 'EthanolLevel', 
                                'Fish', 'GunPoint', 'InsectWingbeatSound', 'ItalyPowerDemand', 'MelbournePedestrian', 
                                'MiddlePhalanxTW', 'MixedShapesRegularTrain', 'OSULeaf', 'Trace', 'WordSynonyms']
        dataset_names = sorted(os.listdir(ucr_root))
        dataset_names = [d for d in dataset_names if d not in ablation_exclude_list]
        dataset_names = ["ECG5000"]

    os.makedirs("logs", exist_ok=True)
    log_file = f"logs/I_ROI/Input_ROI_result_ucr_log_{datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%d_%H%M%S')}.csv"

    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Start inference'])
    
    summary_results = {}
    first_dataset = True

    if args_cli.use_default_param:
        dataset_config_map = load_dataset_config('./configs/dataset_config.json')
    else:
        dataset_config_map = {}

    for dataset_name in tqdm(dataset_names):
        config = dataset_config_map.get(dataset_name, {})
        lambda_1_value = config.get("lambda_1", args_cli.lambda_1)
        num_epochs = config.get("num_epochs", args_cli.num_epochs)
        use_warmup = config.get("use_warmup", args_cli.use_warmup)

        wandb.init(project="shapelet_discovery", name=dataset_name, config={
            "lambda_1": lambda_1_value
        }, reinit=True)

        dataset_save_dir = os.path.join("saved_models", dataset_name)
        dataset_path = os.path.join(ucr_root, dataset_name)
        train_file = os.path.join(dataset_path, f"{dataset_name}_TRAIN.ts")
        test_file = os.path.join(dataset_path, f"{dataset_name}_TEST.ts")

        if not (os.path.exists(train_file) and os.path.exists(test_file)):
            print(f"Skipping {dataset_name} - missing files")
            continue

        print(f"\nProcessing {dataset_name}")
        args = SimpleNamespace(
            seed=42,
            device='cuda:0',
            dataset=dataset_name,
            data_root=ucr_root,
            pred_lr=1e-3,
            sel_lr=5e-4,
            num_epochs=num_epochs,
            patience=300,
            PRETRAIN_ROOT="./saved_models/pretrain",
            num_layers=6,
        )

        if first_dataset:
            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Hyperparameters'])
                writer.writerow([
                    f"pred_lr: {args.pred_lr}, sel_lr: {args.sel_lr}, num_layers: {args.num_layers}, "
                    f"lambda_1: {wandb.config.lambda_1}, num_epochs: {args.num_epochs}"
                ])
                writer.writerow([])
                writer.writerow(['Dataset', 'Fold', 'Val Accuracy', 'Val Selection Rate', 'Test Accuracy', 
                               'Test Selection Rate', 'Valid Lc', 'Test Lc', 'Test shapelet num'])

            first_dataset = False

        if args_cli.use_original_split:
            train_loaders, val_loaders, test_loaders, param_dict = ucr_data_provider_originalSplit(args)
        else: 
            train_loaders, val_loaders, test_loaders, param_dict = ucr_data_provider(args)

        fold_val_accuracies = []
        fold_test_accuracies = []
        fold_val_selections = []
        fold_test_selections = []
        fold_val_Lc = []
        fold_test_Lc = []
        fold_test_avg_segment_count = []
        
        for fold_idx in range(len(train_loaders)):
            set_seed(args.seed)
            val_acc, val_sel, test_acc, test_sel, val_Lc, test_Lc, avg_segment_count = run_single_fold(
                args, train_loaders[fold_idx], val_loaders[fold_idx], test_loaders[fold_idx], 
                param_dict, use_warmup, save_dir=dataset_save_dir, fold_idx=fold_idx + 1
            )
            fold_val_accuracies.append(val_acc)
            fold_val_selections.append(val_sel)
            fold_test_accuracies.append(test_acc)
            fold_test_selections.append(test_sel)
            fold_val_Lc.append(val_Lc)
            fold_test_Lc.append(test_Lc)
            fold_test_avg_segment_count.append(avg_segment_count)

            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    dataset_name, fold_idx + 1,
                    f"{val_acc:.4f}", f"{val_sel:.4f}",
                    f"{test_acc:.4f}", f"{test_sel:.4f}",
                    f"{val_Lc:.4f}", f"{test_Lc:.4f}", f"{avg_segment_count:.4f}",
                ])

        mean_val = np.mean(fold_val_accuracies)
        mean_val_sel = np.mean(fold_val_selections)
        mean_val_lc = np.mean(fold_val_Lc)

        mean_test = np.mean(fold_test_accuracies)
        mean_test_sel = np.mean(fold_test_selections)
        mean_test_lc = np.mean(fold_test_Lc)
        mean_test_sc = np.mean(fold_test_avg_segment_count)

        summary_results[dataset_name] = {
            "val_mean": mean_val,
            "val_sel_mean": mean_val_sel,
            "val_lc_mean": mean_val_lc,
            "test_mean": mean_test,
            "test_sel_mean": mean_test_sel,
            "test_ls_mean": mean_test_lc,
            "test_sc_mean": mean_test_sc,
        }

        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                dataset_name, 'AVG',
                f"{mean_val:.4f}", f"{mean_val_sel:.4f}", 
                f"{mean_test:.4f}", f"{mean_test_sel:.4f}", 
                f"{mean_val_lc:.4f}", f"{mean_test_lc:.4f}", f"{mean_test_sc:.4f}",
            ])

        print(f"{dataset_name} done: Mean Val Acc = {mean_val:.4f}, Mean Test Acc = {mean_test:.4f}")

    print("\nFinal Summary:")
    for ds, res in summary_results.items():
        print(f"{ds}: Val={res['val_mean']:.4f} (Sel: {res['val_sel_mean']:.4f}) (Lc: {res['val_lc_mean']:.4f}) | "
              f"Test={res['test_mean']:.4f} (Sel: {res['test_sel_mean']:.4f}) (Lc: {res['test_ls_mean']:.4f})")


if __name__ == '__main__':
    main()