"""
Global Shapelet Classification Module

Provides utilities for shapelet-based time series classification using global shapelets.
"""

import os
from pathlib import Path
import numpy as np
import torch
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim


def load_global_shapelets(path):
    """
    Load global shapelets from various formats.
    
    Supports: list[Tensor], dict with 'cluster_centers' or 'global_shapelets'
    Returns: list[Tensor cpu float32], each tensor shape [L,1]
    """
    obj = torch.load(path, map_location="cpu")
    centers = None

    if isinstance(obj, list):
        centers = obj
    elif isinstance(obj, dict):
        if "cluster_centers" in obj:
            centers = obj["cluster_centers"]
        elif "global_shapelets" in obj:
            gs = obj["global_shapelets"]
            centers = list(gs.values()) if isinstance(gs, dict) else gs
        else:
            raise ValueError(f"Unsupported dict keys: {list(obj.keys())}")
    else:
        raise ValueError(f"Unsupported type: {type(obj)}")

    out = []
    for c in centers:
        if not isinstance(c, torch.Tensor):
            c = torch.as_tensor(c)
        c = c.detach().cpu().float()
        if c.ndim == 1:
            c = c.unsqueeze(1)
        out.append(c)
    return out


def z_norm_1d(x):
    """Z-normalize a 1D array."""
    mu = x.mean()
    sd = x.std()
    if sd == 0:
        return x - mu
    return (x - mu) / sd


def min_euclidean_sliding_1d(series_1d, shapelet_1d, z_norm_segment=True):
    """
    Compute minimum Euclidean distance using sliding window.
    Falls back to DTW if series is shorter than shapelet.
    """
    T = series_1d.shape[0]
    L = shapelet_1d.shape[0]

    if T >= L:
        s = z_norm_1d(shapelet_1d) if z_norm_segment else shapelet_1d
        best = np.inf
        for start in range(T - L + 1):
            w = series_1d[start:start+L]
            w = z_norm_1d(w) if z_norm_segment else w
            d = np.linalg.norm(w - s)
            if d < best:
                best = d
        return float(best)
    else:
        a = series_1d[:, None]
        b = shapelet_1d[:, None]
        d, _ = fastdtw(a, b, dist=euclidean)
        return float(d)


def series_to_shapelet_features(series, shapelets, z_norm_segment=True):
    """
    Convert time series to shapelet distance features.
    Returns: np.ndarray [K] where K is number of shapelets
    """
    if series.ndim == 1:
        s = series.detach().cpu().numpy()
    else:
        s = series[0].detach().cpu().numpy()
    feats = []
    for sh in shapelets:
        sh_np = sh.squeeze(1).numpy()
        feats.append(min_euclidean_sliding_1d(s, sh_np, z_norm_segment=z_norm_segment))
    return np.asarray(feats, dtype=np.float32)


def batch_to_numpy(batch, device):
    """Parse batch from DataLoader."""
    if isinstance(batch, (list, tuple)):
        if len(batch) >= 2:
            x, y = batch[:2]
        else:
            x, y = batch[0], None
    else:
        x, y = batch, None

    x = x.to(device, non_blocking=True)
    if x.ndim == 2:
        x = x.unsqueeze(1)
    return x, y


def shapelet_transform_from_loader(loader, shapelets, device, z_norm_segment=True):
    """
    Transform all samples in loader to shapelet features.
    Returns: X (N,K) np.float32, y (N,) np.int64 (or None)
    """
    X_list, y_list = [], []
    for batch in loader:
        x, y = batch_to_numpy(batch, device)
        print(x.shape)
        B, M, T = x.shape
        for i in range(B):
            feats = series_to_shapelet_features(x[i], shapelets, z_norm_segment=z_norm_segment)
            X_list.append(feats)
        if y is not None:
            if isinstance(y, torch.Tensor):
                y_list.extend(y.detach().cpu().numpy().tolist())
            else:
                y_list.extend(y)

    X = np.stack(X_list, axis=0) if X_list else np.zeros((0, len(shapelets)), dtype=np.float32)
    y = np.asarray(y_list, dtype=np.int64) if y_list else None
    return X, y


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, num_classes),
        )

    def forward(self, x):
        return self.model(x)


def train_eval_shapelet_mlp_classifier(train_loader, val_loader, test_loader, centers_path,
                                       device="cuda:0", z_norm_segment=True,
                                       hidden_dim=100, lr=1e-3, epochs=20):
    """Train and evaluate MLP classifier on shapelet features."""
    print(f"[load] global shapelets: {centers_path}")
    shapelets = load_global_shapelets(centers_path)
    K = len(shapelets)
    print(f"[info] #shapelets = {K}")

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    print("[transform] train ...")
    X_tr, y_tr = shapelet_transform_from_loader(train_loader, shapelets, device, z_norm_segment=z_norm_segment)

    print("[transform] val  ...")
    X_val, y_val = shapelet_transform_from_loader(val_loader, shapelets, device, z_norm_segment=z_norm_segment)

    print("[transform] test  ...")
    X_te, y_te = shapelet_transform_from_loader(test_loader, shapelets, device, z_norm_segment=z_norm_segment)

    print(f"[shape] X_tr={X_tr.shape}, X_te={X_te.shape}")
    if y_tr is None or y_te is None:
        raise ValueError("Labels not found in loaders.")

    X_tr_tensor = torch.tensor(X_tr, dtype=torch.float32).to(device)
    y_tr_tensor = torch.tensor(y_tr, dtype=torch.long).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)
    X_te_tensor = torch.tensor(X_te, dtype=torch.float32).to(device)

    num_classes = len(set(y_tr))
    model = SimpleMLP(input_dim=K, num_classes=num_classes, hidden_dim=hidden_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    best_state_dict = None

    print("[train] MLP ...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_tr_tensor)
        loss = criterion(output, y_tr_tensor)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_tensor)
            val_pred = val_logits.argmax(dim=1)
            val_acc = (val_pred == y_val_tensor).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = model.state_dict()

        if (epoch + 1) % 1 == 0 or epoch == 0:
            train_pred = output.argmax(dim=1)
            train_acc = (train_pred == y_tr_tensor).float().mean().item()
            print(f"Epoch {epoch+1}/{epochs}: loss={loss.item():.4f}, train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")

    print("[info] Loading best model from validation ...")
    model.load_state_dict(best_state_dict)

    print("[eval] test set ...")
    model.eval()
    with torch.no_grad():
        logits = model(X_te_tensor)
        pred_te = logits.argmax(dim=1).cpu().numpy()

    acc_te = accuracy_score(y_te, pred_te)
    print(f"[test] acc = {acc_te:.4f}")
    print(classification_report(y_te, pred_te, digits=4))

    return {
        "shapelets": shapelets,
        "X_tr": X_tr, "y_tr": y_tr,
        "X_te": X_te, "y_te": y_te,
        "mlp": model,
        "acc": acc_te,
        "val_acc": best_val_acc
    }


def train_eval_shapelet_classifier(train_loader, test_loader, centers_path,
                                   device="cuda:0", z_norm_segment=True, clf_C=1.0):
    """Train and evaluate logistic regression on shapelet features."""
    print(f"[load] global shapelets: {centers_path}")
    shapelets = load_global_shapelets(centers_path)
    K = len(shapelets)
    print(f"[info] #shapelets = {K}")

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    print("[transform] train ...")
    X_tr, y_tr = shapelet_transform_from_loader(train_loader, shapelets, device, z_norm_segment=z_norm_segment)
    print("[transform] test  ...")
    X_te, y_te = shapelet_transform_from_loader(test_loader, shapelets, device, z_norm_segment=z_norm_segment)

    print(f"[shape] X_tr={X_tr.shape}, X_te={X_te.shape}")
    if y_tr is None or y_te is None:
        raise ValueError("Labels not found in loaders.")

    clf = LogisticRegression(C=clf_C, max_iter=1000, n_jobs=-1, multi_class="auto")
    clf.fit(X_tr, y_tr)

    pred = clf.predict(X_te)
    acc = accuracy_score(y_te, pred)
    print(f"[test] acc = {acc:.4f}")
    print(classification_report(y_te, pred, digits=4))

    return {
        "shapelets": shapelets,
        "X_tr": X_tr, "y_tr": y_tr,
        "X_te": X_te, "y_te": y_te,
        "clf": clf,
        "acc": acc
    }


def train_eval_shapelet_classifier_xgb(
    train_loader,
    val_loader,
    test_loader,
    centers_path,
    device="cuda:0",
    z_norm_segment=True,
    xgb_params=None,
    early_stopping_rounds=50,
):
    """Train and evaluate XGBoost classifier on shapelet features with early stopping."""
    import numpy as np
    import torch
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score, classification_report

    print(f"[load] global shapelets: {centers_path}")
    shapelets = load_global_shapelets(centers_path)
    K = len(shapelets)
    print(f"[info] #shapelets = {K}")

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    print("[transform] train ...")
    X_tr, y_tr = shapelet_transform_from_loader(
        train_loader, shapelets, device, z_norm_segment=z_norm_segment
    )

    if val_loader is None:
        raise ValueError("val_loader is None. Validation set required for early stopping.")

    print("[transform] valid ...")
    X_val, y_val = shapelet_transform_from_loader(
        val_loader, shapelets, device, z_norm_segment=z_norm_segment
    )

    print("[transform] test  ...")
    X_te, y_te = shapelet_transform_from_loader(
        test_loader, shapelets, device, z_norm_segment=z_norm_segment
    )

    print(f"[shape] X_tr={X_tr.shape}, X_val={X_val.shape}, X_te={X_te.shape}")
    if y_tr is None or y_val is None or y_te is None:
        raise ValueError("Labels not found in one of loaders (train/val/test).")

    y_unique = np.unique(y_tr)
    num_class = len(y_unique)
    is_binary = (num_class == 2)

    base_params = {
        "n_estimators": 2000,
        "learning_rate": 0.03,
        "max_depth": 3,
        "min_child_weight": 2,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
        "gamma": 0.0,
        "objective": "binary:logistic" if is_binary else "multi:softprob",
        "eval_metric": "logloss" if is_binary else ["mlogloss", "merror"],
        "n_jobs": -1,
        "tree_method": "hist",
        "random_state": 42,
        "verbosity": 1,
        "early_stopping_rounds": early_stopping_rounds,
    }
    if not is_binary:
        base_params["num_class"] = num_class

    if xgb_params is not None:
        base_params.update(xgb_params)

    clf = XGBClassifier(**base_params)
    clf.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    print(f"[xgb] best_iteration={getattr(clf, 'best_iteration', None)} "
          f"best_score={getattr(clf, 'best_score', None)}")

    pred = clf.predict(X_te)
    acc = accuracy_score(y_te, pred)
    print(f"[test] acc = {acc:.4f}")
    print(classification_report(y_te, pred, digits=4))

    return {
        "shapelets": shapelets,
        "X_tr": X_tr, "y_tr": y_tr,
        "X_val": X_val, "y_val": y_val,
        "X_te": X_te, "y_te": y_te,
        "clf": clf,
        "acc": acc
    }