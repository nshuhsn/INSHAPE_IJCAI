import torch
import torch.nn.functional as F
import numpy as np

def segment(
    x: torch.Tensor,        # (B, M, T)
    min_len: int = 30,
    pen:     float = 0.1,
    model:   str = "rbf",
    n_jobs:  int = -1,
    algorithm: str = 'Pelt'
):
    """
    Segment time series using change point detection.
    
    Args:
        x: (B, M, T) input tensor
        min_len: minimum segment length
        pen: penalty for change point detection
        model: model type for ruptures
        n_jobs: number of parallel jobs (-1 for max cores)
        algorithm: 'Pelt' or 'spline'
    
    Returns:
        roi_time_mask: Tensor (B, M, Rmax, T) float32
                       - each (b,m,r,t) is 1.0 if belongs to that ROI
        roi_valid: Tensor (B, M, Rmax) bool
                   - each (b,m,r) is True if ROI exists
        L_max: int (longest ROI length)
    """
    import ruptures as rpt
    from joblib import Parallel, delayed
    from scipy.interpolate import UnivariateSpline

    B, M, T = x.shape
    # 1) Create 1D time series list for PELT algorithm
    ts_list = x.reshape(-1, T).cpu().numpy()  # (B*M, T)

    # Normalize algorithm string to lowercase for comparison
    algo_lower = algorithm.lower()
    
    if algo_lower == 'pelt':
        # 2) Parallel PELT segmentation
        def _run(ts):
            return rpt.Pelt(model=model, min_size=min_len)\
                    .fit(ts).predict(pen=pen)
        all_bkps = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_run)(ts) for ts in ts_list
        )
    elif algo_lower == 'spline':
        # 2) Parallel spline break point extraction
        def _get_breakpoints(ts):
            x_range = np.arange(len(ts))
            spline = UnivariateSpline(x_range, ts, s=1)
            break_points = spline.get_knots()
            return break_points
        all_bkps = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_get_breakpoints)(ts) for ts in ts_list
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Must be 'pelt' or 'spline'")

    # 3) Count ROIs per time series, get max count Rmax
    lens = torch.tensor([len(b) for b in all_bkps], dtype=torch.long)  # (B*M,)
    Rmax = int(lens.max().item())

    # 4) Pad breakpoints to length Rmax (fill remaining with T)
    bkps_pad = torch.full((B*M, Rmax), T, dtype=torch.long)
    for i, bk in enumerate(all_bkps):
        bkps_pad[i, :lens[i]] = torch.tensor(bk, dtype=torch.long)
    # bkps_pad[i] = [e1,e2,...,eK, T, T, ...]

    # 5) Calculate start points: first segment starts at 0, rest from previous break-pt
    starts = torch.cat([
        torch.zeros((B*M,1), dtype=torch.long),  # s0 = 0
        bkps_pad[:, :-1]                         # previous endpoints
    ], dim=1)  # (B*M, Rmax)

    # 6) Calculate lengths and valid mask
    lengths    = bkps_pad - starts            # (B*M, Rmax) segment lengths
    valid_mask = lengths > 0                  # False for empty segments

    # 7) Get longest segment length
    L_max = int(lengths[valid_mask].max().item())

    # 8) Create roi_time_mask: (B*M, Rmax, T) boolean -> float
    #    Set 1.0 where starts[:,r] <= t < bkps_pad[:,r]
    ar = torch.arange(T, device=x.device).reshape(1,1,T)            # (1,1,T)
    st = starts.to(x.device).reshape(B*M, Rmax, 1)                 # (B*M,Rmax,1)
    ed = bkps_pad.to(x.device).reshape(B*M, Rmax, 1)               # (B*M,Rmax,1)
    mask = (ar >= st) & (ar < ed)                              # (B*M,Rmax,T)
    roi_time_mask = mask.reshape(B, M, Rmax, T).float()           # (B,M,Rmax,T)

    # 9) roi_valid: simply reshape valid_mask
    roi_valid = valid_mask.reshape(B, M, Rmax)                    # (B,M,Rmax)

    return roi_time_mask, roi_valid, L_max


def pack_valid_roi_fast(
    x: torch.Tensor,             # (B, M, T)
    roi_time_mask: torch.Tensor, # (B, M, R, T)
    roi_valid: torch.Tensor,     # (B, M, R)
    L_max: int,
):
    """Pack valid ROIs into a single tensor (fast version)."""
    B, M, R, T = roi_time_mask.shape

    # 1) Extract coordinates + idx_map
    coords = roi_valid.nonzero(as_tuple=False)      # (N,3)
    idx_map = { (int(b),int(m),int(r)): i
                for i,(b,m,r) in enumerate(coords) }
    N = coords.size(0)

    # 2) Flatten preparation
    x_flat    = x.reshape(B*M, T)                     # (BM, T)
    mask_flat  = (roi_time_mask.reshape(B*M*R, T) > 0)        # (BM*R, T)
    valid_flat= roi_valid.reshape(B*M*R)              # (BM*R,)

    # 3) Select only valid ROIs
    x_sel  = x_flat.unsqueeze(1).expand(-1, R, -1).reshape(-1, T)  # (BM*R, T)
    x_sel  = x_sel[valid_flat]                                    # (N, T)
    mask_sel = mask_flat[valid_flat]                              # (N, T)

    # 4) Compress values for each ROI to front positions
    #    mask_sel.cumsum(dim=1)-1 generates 0,1,2,... sequence for mask=1 positions
    positions = mask_sel.long().cumsum(dim=1) - 1   # (N, T), mask_sel=1 positions get 0..l-1
    positions = positions * mask_sel        # (N, T) mask_sel=0 positions become 0

    # 5) Fill values using scatter
    seg = torch.zeros(N, L_max, device=x.device)           # (N, L_max)
    batch_idx = torch.arange(N, device=x.device).unsqueeze(1).expand(-1, T)  # (N, T)
    seg[batch_idx[mask_sel], positions[mask_sel]] = x_sel[mask_sel]

    # 6) Add dimension, generate pad_mask
    seg = seg.unsqueeze(-1)                                # (N, L_max, 1)
    pad_mask = torch.arange(L_max, device=x.device).unsqueeze(0) < \
               (mask_sel.sum(dim=1).unsqueeze(1))          # (N, L_max)
    
    pad_mask = ~pad_mask

    return seg, pad_mask, idx_map