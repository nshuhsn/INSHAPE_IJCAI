from sktime.datasets import load_UCR_UEA_dataset
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter

class CustomUCRDataset(Dataset):
    def __init__(self, data, labels, ids=None, return_mask=False):
        self.data = torch.tensor(data, dtype=torch.float32).unsqueeze(1)
        self.labels = torch.tensor(labels, dtype=torch.long)
        if ids is None:
            self.ids = torch.arange(len(self.data), dtype=torch.long)
        else:
            self.ids = torch.tensor(ids, dtype=torch.long)
        self.return_mask = return_mask

    def __getitem__(self, idx):
        x = self.data[idx]      # [1, L]
        y = self.labels[idx]    # []
        sid = self.ids[idx]     # []
        if self.return_mask:
            T = x.shape[-1]
            mask = torch.ones(T, dtype=torch.bool)  
            return x, y, sid, mask                
        return x, y, sid                          

    def __len__(self):
        return len(self.data)

import os

def debug_dataset_paths(dataset_name, data_root):
    dataset_dir = os.path.join(data_root, dataset_name)
    train_file = os.path.join(dataset_dir, f"{dataset_name}_TRAIN.ts")
    test_file = os.path.join(dataset_dir, f"{dataset_name}_TEST.ts")

    print("\n [DEBUG] Dataset Path Check")
    print(f" Dataset Dir: {dataset_dir} - Exists: {os.path.exists(dataset_dir)}")
    print(f" Train File : {train_file} - Exists: {os.path.exists(train_file)}")
    print(f" Test File  : {test_file} - Exists: {os.path.exists(test_file)}")

    if os.path.exists(dataset_dir):
        print("\n [DEBUG] Contents of Dataset Dir:")
        for fname in os.listdir(dataset_dir):
            print(" -", fname)
    else:
        print(" Dataset directory does not exist.")

def transfer_labels(labels):
    unique = np.unique(labels)
    label_map = {v: i for i, v in enumerate(unique)}
    return np.array([label_map[y] for y in labels])

def fill_nan_value(train_set, val_set, test_set):
    ind = np.where(np.isnan(train_set))
    col_mean = np.nanmean(train_set, axis=0)
    col_mean[np.isnan(col_mean)] = 1e-6

    train_set[ind] = np.take(col_mean, ind[1])

    ind_val = np.where(np.isnan(val_set))
    val_set[ind_val] = np.take(col_mean, ind_val[1])

    ind_test = np.where(np.isnan(test_set))
    test_set[ind_test] = np.take(col_mean, ind_test[1])
    
    return train_set, val_set, test_set

def normalize_per_series(data):
    std_ = data.std(axis=1, keepdims=True)
    std_[std_ == 0] = 1.0
    return (data - data.mean(axis=1, keepdims=True)) / std_

from sklearn.model_selection import StratifiedKFold, train_test_split

from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
import numpy as np
from collections import Counter

def ucr_data_provider(args, n_splits=5):
    print(args.dataset, args.data_root)
    debug_dataset_paths(args.dataset, args.data_root)

    X_train, y_train = load_UCR_UEA_dataset(
        name=args.dataset, split="train", return_type="numpy2d", extract_path=args.data_root)
    X_test, y_test = load_UCR_UEA_dataset(
        name=args.dataset, split="test", return_type="numpy2d", extract_path=args.data_root)
    
    X_all = np.concatenate([X_train, X_test], axis=0)
    y_all = np.concatenate([y_train, y_test], axis=0).astype(int)
    y_all = transfer_labels(y_all)

    all_ids = np.arange(len(X_all), dtype=np.int64)

    skf_outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)

    train_loaders, val_loaders, test_loaders = [], [], []

    for fold_idx, (raw_idx, test_idx) in enumerate(skf_outer.split(X_all, y_all)):
        X_raw, y_raw = X_all[raw_idx], y_all[raw_idx]
        X_test, y_test = X_all[test_idx], y_all[test_idx]
        ids_raw = all_ids[raw_idx]   
        ids_tst = all_ids[test_idx]  

        skf_inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=args.seed)
        inner_train_idx, val_idx = next(skf_inner.split(X_raw, y_raw))
        X_train, y_train = X_raw[inner_train_idx], y_raw[inner_train_idx]
        X_val, y_val = X_raw[val_idx], y_raw[val_idx]
        ids_trn = ids_raw[inner_train_idx]  
        ids_val = ids_raw[val_idx]        

        X_train, X_val, X_test = fill_nan_value(X_train, X_val, X_test)

        X_train = normalize_per_series(X_train)
        X_val = normalize_per_series(X_val)
        X_test = normalize_per_series(X_test)

        args.batch_size = min(len(X_train), 512)
        
        train_set = CustomUCRDataset(X_train, y_train, ids=ids_trn, return_mask=False)
        val_set   = CustomUCRDataset(X_val, y_val, ids=ids_val, return_mask=False)
        test_set  = CustomUCRDataset(X_test, y_test, ids=ids_tst, return_mask=False)

        # DataLoaders
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_set, batch_size=args.batch_size)
        test_loader = DataLoader(test_set, batch_size=args.batch_size)
        # test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)

        train_loaders.append(train_loader)
        val_loaders.append(val_loader)
        test_loaders.append(test_loader)

    # Shape info
    X_all = np.expand_dims(X_all, axis=1) if X_all.ndim == 2 else X_all
    enc_in = X_all.shape[1]
    seq_len = X_all.shape[2]

    param_dict = {
        "seq_len": seq_len,
        "enc_in": enc_in,
        "num_classes": len(np.unique(y_all))
    }

    return train_loaders, val_loaders, test_loaders, param_dict


def ucr_data_provider_originalSplit(args, n_splits=5):
    print(args.dataset, args.data_root)
    debug_dataset_paths(args.dataset, args.data_root)

    X_train, y_train = load_UCR_UEA_dataset(
        name=args.dataset, split="train", return_type="numpy2d", extract_path=args.data_root)
    X_test, y_test = load_UCR_UEA_dataset(
        name=args.dataset, split="test", return_type="numpy2d", extract_path=args.data_root)

    y_all = np.concatenate([y_train, y_test], axis=0).astype(int)
    y_all = transfer_labels(y_all)
    orig_all = np.concatenate([y_train, y_test], axis=0).astype(int)
    uniq_orig = np.unique(orig_all)
    uniq_mapped = np.unique(y_all)
    label_map = {o: m for o, m in zip(uniq_orig, uniq_mapped)}
    y_train = np.vectorize(label_map.get)(y_train.astype(int))
    y_test  = np.vectorize(label_map.get)(y_test.astype(int))

    train_loaders, val_loaders, test_loaders = [], [], []

    train_ids_global = np.arange(len(X_train), dtype=np.int64)
    test_ids_global  = np.arange(len(X_test),  dtype=np.int64)

    X_trn_raw, X_val_raw, X_tst_raw = fill_nan_value(
        X_train.copy(), X_test.copy(), X_test.copy()
    )
    X_trn = normalize_per_series(X_trn_raw)
    X_val = normalize_per_series(X_val_raw)
    X_tst = normalize_per_series(X_tst_raw)

    train_set_full = CustomUCRDataset(X_trn, y_train, ids=train_ids_global, return_mask=False)
    val_set_full   = CustomUCRDataset(X_val, y_test,  ids=test_ids_global,  return_mask=False)  # = test
    test_set_full  = CustomUCRDataset(X_tst, y_test,  ids=test_ids_global,  return_mask=False)

    args.batch_size = min(len(X_trn), 512)

    for _ in range(n_splits):
        train_loader = DataLoader(train_set_full, batch_size=args.batch_size, shuffle=True,  drop_last=False)
        val_loader   = DataLoader(val_set_full,   batch_size=args.batch_size)
        test_loader  = DataLoader(test_set_full,  batch_size=args.batch_size)

        train_loaders.append(train_loader)
        val_loaders.append(val_loader)
        test_loaders.append(test_loader)

    # Shape info
    base = X_train
    base = np.expand_dims(base, axis=1) if base.ndim == 2 else base
    enc_in = base.shape[1]
    seq_len = base.shape[2]

    param_dict = {
        "seq_len": seq_len,
        "enc_in": enc_in,
        "num_classes": len(np.unique(y_all))
    }

    return train_loaders, val_loaders, test_loaders, param_dict