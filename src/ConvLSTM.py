from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any

from sklearn.metrics import root_mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# -------------------------
# Dense cube utilities
# -------------------------

def _infer_bounds(df: pd.DataFrame) -> Dict[str, int]:
    return {
        "d_min": int(df["d"].min()),
        "d_max": int(df["d"].max()),
        "t_min": int(df["t"].min()),
        "t_max": int(df["t"].max()),
        "x_min": int(df["x"].min()),
        "x_max": int(df["x"].max()),
        "y_min": int(df["y"].min()),
        "y_max": int(df["y"].max()),
    }


def build_dense_cube(
    df: pd.DataFrame,
    *,
    use_log1p: bool = True,
    dtype: np.dtype = np.float16,
    bounds: Optional[Dict[str, int]] = None,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    每一天 D、每一個 time slot T，都有一張「人流 Heat map」，大小 H×W
    (D:多少天, T:48個time slot, H:X格網大小, W:Y格網大小)

    Build a dense cube:
        cube[d_idx, t, x_idx, y_idx] = log1p(count) (default) or count
    where indices are shifted to compact ranges using bounds.

    Returns:
        cube: (D, T, H, W)
        meta: dict with shifts and sizes:
              d_min, x_min, y_min, D, T, H, W
    """
    if bounds is None:
        bounds = _infer_bounds(df)

    d_min, d_max = bounds["d_min"], bounds["d_max"]
    t_min, t_max = bounds["t_min"], bounds["t_max"]
    x_min, x_max = bounds["x_min"], bounds["x_max"]
    y_min, y_max = bounds["y_min"], bounds["y_max"]

    D = d_max - d_min + 1
    T = t_max - t_min + 1
    H = x_max - x_min + 1
    W = y_max - y_min + 1

    cube = np.zeros((D, T, H, W), dtype=dtype)

    d = (df["d"].to_numpy(np.int64) - d_min)
    t = (df["t"].to_numpy(np.int64) - t_min)
    x = (df["x"].to_numpy(np.int64) - x_min)
    y = (df["y"].to_numpy(np.int64) - y_min)

    v = df["count"].to_numpy(np.float32)
    if use_log1p:
        v = np.log1p(v)

    ok = (d >= 0) & (d < D) & (t >= 0) & (t < T) & (x >= 0) & (x < H) & (y >= 0) & (y < W)
    d, t, x, y, v = d[ok], t[ok], x[ok], y[ok], v[ok]

    np.add.at(cube, (d, t, x, y), v.astype(dtype, copy=False))

    meta = {
        "d_min": d_min,
        "t_min": t_min,
        "x_min": x_min,
        "y_min": y_min,
        "D": D,
        "T": T,
        "H": H,
        "W": W,
        "dtype": str(np.dtype(dtype)),
    }
    return cube, meta


def extract_patch_from_frame(frame_hw: np.ndarray, x: int, y: int, radius: int) -> np.ndarray:
    """
    frame_hw: (H,W), x/y in [0..H-1]/[0..W-1]
    Returns (K,K) patch, K=2r+1, with zero padding near borders.
    """
    H, W = frame_hw.shape
    K = 2 * radius + 1

    x0, x1 = x - radius, x + radius + 1
    y0, y1 = y - radius, y + radius + 1

    xs0, xs1 = max(x0, 0), min(x1, H)
    ys0, ys1 = max(y0, 0), min(y1, W)

    patch = np.zeros((K, K), dtype=frame_hw.dtype)
    px0 = xs0 - x0
    py0 = ys0 - y0
    patch[px0:px0 + (xs1 - xs0), py0:py0 + (ys1 - ys0)] = frame_hw[xs0:xs1, ys0:ys1]
    return patch


# -------------------------
# Dataset: patch sequence + embeddings
# -------------------------

class SeqPatchDatasetEmbed(Dataset):
    """
    Each sample = one row in df, with input sequence:
        x_seq: (L,1,K,K), each step is a patch at SAME t-slot across day-lags.
    Returns tuple compatible with your existing training loops:
        (x_seq, weekday, t_id, x_id, y_id, is_weekend(1,), base_log(1,), y)
    """
    def __init__(self, df: pd.DataFrame, seq_len: int, patch_radius: int,
                 cube: np.ndarray, meta: Dict[str, int]):
        self.df = df.reset_index(drop=True)
        self.seq_len = int(seq_len)
        self.patch_radius = int(patch_radius)
        self.cube = cube
        self.meta = meta

        self.d = self.df["d"].to_numpy(np.int64)
        self.t = self.df["t"].to_numpy(np.int64)
        self.x = self.df["x"].to_numpy(np.int64)
        self.y = self.df["y"].to_numpy(np.int64)

        # embedding ids (global ids)
        self.weekday = self.df["weekday"].to_numpy(np.int64)
        self.t_id = self.df["t"].to_numpy(np.int64)
        self.x_id = self.df["x"].to_numpy(np.int64)
        self.y_id = self.df["y"].to_numpy(np.int64)
        self.is_weekend = self.df["is_weekend"].to_numpy(np.float32)

        # target：若有 y_target 就用（residual learning），否則維持原本 log1p(count)
        self.base_log = self.df["base_log"].to_numpy(np.float32) if "base_log" in self.df.columns else None

        if "y_target" in self.df.columns:
            self.target = self.df["y_target"].to_numpy(np.float32)
        else:
            self.target = np.log1p(self.df["count"].to_numpy(np.float32))

    def __len__(self):
        return len(self.target)

    def __getitem__(self, i):
        d = int(self.d[i]); t = int(self.t[i]); x = int(self.x[i]); y = int(self.y[i])

        d0 = d - self.meta["d_min"]
        t0 = t - self.meta["t_min"]
        x0 = x - self.meta["x_min"]
        y0 = y - self.meta["y_min"]

        L = self.seq_len
        r = self.patch_radius
        K = 2 * r + 1

        x_seq = np.zeros((L, 1, K, K), dtype=np.float32)  # float32 to model

        for j, k in enumerate(range(L, 0, -1)):
            d_lag = d0 - k
            if d_lag < 0 or d_lag >= self.cube.shape[0]:
                continue
            if t0 < 0 or t0 >= self.cube.shape[1]:
                continue
            frame = self.cube[d_lag, t0].astype(np.float32, copy=False)  # (H,W)
            if 0 <= x0 < frame.shape[0] and 0 <= y0 < frame.shape[1]:
                x_seq[j, 0] = extract_patch_from_frame(frame, x0, y0, r)

        base_log_i = 0.0 if self.base_log is None else float(self.base_log[i])

        return (
            torch.from_numpy(x_seq),
            torch.tensor(self.weekday[i], dtype=torch.long),
            torch.tensor(self.t_id[i], dtype=torch.long),
            torch.tensor(self.x_id[i], dtype=torch.long),
            torch.tensor(self.y_id[i], dtype=torch.long),
            torch.tensor([self.is_weekend[i]], dtype=torch.float32),
            torch.tensor([base_log_i], dtype=torch.float32),
            torch.tensor(self.target[i], dtype=torch.float32),
        )


# -------------------------
# ConvLSTM
# -------------------------

class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch: int, hid_ch: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2 #卷積範圍
        self.hid_ch = int(hid_ch)
        self.conv = nn.Conv2d(in_ch + hid_ch, 4 * hid_ch, kernel_size, padding=padding)

    def forward(self, x, h, c):
        if h is None:
            B, _, H, W = x.shape
            h = torch.zeros((B, self.hid_ch, H, W), device=x.device, dtype=x.dtype)
            c = torch.zeros((B, self.hid_ch, H, W), device=x.device, dtype=x.dtype)

        # i:input, f:forget, o:output, g:candidate
        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i); f = torch.sigmoid(f); o = torch.sigmoid(o); g = torch.tanh(g) # activation不建議換

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTMEncoder(nn.Module):
    """Single-layer ConvLSTM encoder: x_seq (B,L,1,K,K) -> last hidden map (B,Hid,K,K)."""
    def __init__(self, hid_ch: int = 32, kernel_size: int = 3):
        super().__init__()
        self.cell = ConvLSTMCell(in_ch=1, hid_ch=hid_ch, kernel_size=kernel_size)

    def forward(self, x_seq):
        h, c = None, None
        L = x_seq.size(1)
        for t in range(L):
            x = x_seq[:, t]
            h, c = self.cell(x, h, c)
        return h


class ConvLSTMRegEmbed(nn.Module):
    """
    ConvLSTM over patch sequence + categorical embeddings.
    Predicts in log-space (log1p(count) or residual log-space depending on your target).
    Signature matches your LSTMRegEmbed:
        forward(x_seq, weekday, t_id, x_id, y_id, is_weekend) -> (B,)
    """
    def __init__(self, hid_ch: int = 32, kernel_size: int = 3,
                 n_weekday: int = 7, n_t: int = 48, n_x: int = 2000, n_y: int = 2000,
                 emb_wd: int = 2, emb_t: int = 8, emb_x: int = 16, emb_y: int = 16,
                 mlp: int = 128):
        super().__init__()
        self.encoder = ConvLSTMEncoder(hid_ch=hid_ch, kernel_size=kernel_size)

        self.emb_weekday = nn.Embedding(n_weekday, emb_wd)
        self.emb_t = nn.Embedding(n_t, emb_t)
        self.emb_x = nn.Embedding(n_x, emb_x)
        self.emb_y = nn.Embedding(n_y, emb_y)

        static_dim = emb_wd + emb_t + emb_x + emb_y + 1

        self.head = nn.Sequential(
            nn.Linear(hid_ch + static_dim, mlp),
            nn.ReLU(),
            nn.Linear(mlp, 1),
        )

    def forward(self, x_seq, weekday, t_id, x_id, y_id, is_weekend):
        h_map = self.encoder(x_seq)  # (B,hid,K,K)
        K = h_map.size(-1)
        cx = K // 2; cy = K // 2
        h_center = h_map[:, :, cx, cy]  # (B,hid)

        e = torch.cat([
            self.emb_weekday(weekday),
            self.emb_t(t_id),
            self.emb_x(x_id),
            self.emb_y(y_id),
            is_weekend,
        ], dim=1)

        z = torch.cat([h_center, e], dim=1)
        return self.head(z).squeeze(1)


# -------------------------
# Train / Predict
# -------------------------

def train_convlstm_embed(
    model: nn.Module,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    seq_len: int,
    patch_radius: int = 4,
    sample_n: int = 200_000,
    batch_size: int = 256,
    epochs: int = 20,
    lr: float = 1e-3,
    seed: int = 42,
    loss: str = "mse",
    huber_beta: float = 1.0,
    use_residual: bool = False,
    lookup_df: Optional[pd.DataFrame] = None,
    cube_dtype: np.dtype = np.float16,
    cube_bounds: Optional[Dict[str, int]] = None,
):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if lookup_df is None:
        lookup_df = pd.concat([train_df, val_df], ignore_index=True)

    cube, meta = build_dense_cube(
        lookup_df[["d", "t", "x", "y", "count"]],
        use_log1p=True,
        dtype=cube_dtype,
        bounds=cube_bounds,
    )

    tr = train_df.sample(n=min(sample_n, len(train_df)), random_state=seed).reset_index(drop=True)

    ds_tr = SeqPatchDatasetEmbed(tr, seq_len=seq_len, patch_radius=patch_radius, cube=cube, meta=meta)
    ds_va = SeqPatchDatasetEmbed(val_df, seq_len=seq_len, patch_radius=patch_radius, cube=cube, meta=meta)

    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss(beta=huber_beta) if loss == "huber" else nn.MSELoss()

    best_rmse = float("inf")
    best_state = None
    patience, bad = 3, 0

    for ep in range(1, epochs + 1):
        model.train()
        for xseq, wd, tid, xid, yid, isw, base_log, y in dl_tr:
            xseq, wd, tid, xid, yid, isw, base_log, y = (
                xseq.to(device, non_blocking=True),
                wd.to(device, non_blocking=True),
                tid.to(device, non_blocking=True),
                xid.to(device, non_blocking=True),
                yid.to(device, non_blocking=True),
                isw.to(device, non_blocking=True),
                base_log.to(device, non_blocking=True),
                y.to(device, non_blocking=True),
            )
            pred = model(xseq, wd, tid, xid, yid, isw)
            l = loss_fn(pred, y)
            opt.zero_grad()
            l.backward()
            opt.step()

        model.eval()
        preds, ys = [], []
        with torch.no_grad():
            for xseq, wd, tid, xid, yid, isw, base_log, y in dl_va:
                xseq, wd, tid, xid, yid, isw, base_log, y = (
                    xseq.to(device, non_blocking=True),
                    wd.to(device, non_blocking=True),
                    tid.to(device, non_blocking=True),
                    xid.to(device, non_blocking=True),
                    yid.to(device, non_blocking=True),
                    isw.to(device, non_blocking=True),
                    base_log.to(device, non_blocking=True),
                    y.to(device, non_blocking=True),
                )
                pred_log = model(xseq, wd, tid, xid, yid, isw)
                if use_residual:
                    base = base_log.squeeze(1)
                    pred_log_full = pred_log + base
                    y_log_full = y + base
                else:
                    pred_log_full = pred_log
                    y_log_full = y

                preds.append(torch.expm1(pred_log_full).cpu().numpy())
                ys.append(torch.expm1(y_log_full).cpu().numpy())

        yhat = np.concatenate(preds)
        ytrue = np.concatenate(ys)
        rmse = float(root_mean_squared_error(ytrue, yhat))
        print(f"epoch {ep}: val RMSE={rmse:.4f}")

        if rmse < best_rmse - 1e-4:
            best_rmse = rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print("Early stop.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model._cube_meta = meta  # optional
    return model


def predict_convlstm_embed(
    model: nn.Module,
    df_split: pd.DataFrame,
    seq_len: int,
    patch_radius: int = 4,
    batch_size: int = 512,
    use_residual: bool = False,
    lookup_df: Optional[pd.DataFrame] = None,
    cube_dtype: np.dtype = np.float16,
    cube_bounds: Optional[Dict[str, int]] = None,
):
    device = next(model.parameters()).device

    if lookup_df is None:
        lookup_df = df_split

    cube, meta = build_dense_cube(
        lookup_df[["d", "t", "x", "y", "count"]],
        use_log1p=True,
        dtype=cube_dtype,
        bounds=cube_bounds,
    )

    ds = SeqPatchDatasetEmbed(df_split, seq_len=seq_len, patch_radius=patch_radius, cube=cube, meta=meta)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model.eval()
    preds = []
    with torch.no_grad():
        for xseq, wd, tid, xid, yid, isw, base_log, _y in dl:
            xseq, wd, tid, xid, yid, isw, base_log = (
                xseq.to(device, non_blocking=True),
                wd.to(device, non_blocking=True),
                tid.to(device, non_blocking=True),
                xid.to(device, non_blocking=True),
                yid.to(device, non_blocking=True),
                isw.to(device, non_blocking=True),
                base_log.to(device, non_blocking=True),
            )
            pred_log = model(xseq, wd, tid, xid, yid, isw)
            if use_residual:
                pred_log = pred_log + base_log.squeeze(1)
            preds.append(torch.expm1(pred_log).cpu().numpy())

    yhat = np.concatenate(preds)
    return np.clip(yhat, 0, None)


def make_df_pred_convlstm_embed(
    df_feat: pd.DataFrame,
    model: nn.Module,
    cfg: Dict[str, Any],
    batch_size: int = 512,
    use_residual: bool = False,
    lookup_df: Optional[pd.DataFrame] = None,
):
    seq_len = int(cfg["seq_len"])
    patch_radius = int(cfg.get("patch_radius", 4))

    cube_dtype = np.float16 if cfg.get("cube_dtype", "float16") == "float16" else np.float32
    cube_bounds = cfg.get("cube_bounds", None)

    yhat = predict_convlstm_embed(
        model=model,
        df_split=df_feat,
        seq_len=seq_len,
        patch_radius=patch_radius,
        batch_size=batch_size,
        use_residual=use_residual,
        lookup_df=lookup_df,
        cube_dtype=cube_dtype,
        cube_bounds=cube_bounds,
    )
    df_pred = df_feat[["d", "t", "x", "y"]].copy()
    df_pred["score"] = yhat
    return df_pred


# -------------------------
# Save / Load
# -------------------------

def save_convlstm_pkl(model: nn.Module, path_pkl: str, config: Dict[str, Any]):
    payload = {"state_dict": model.state_dict(), "config": config}
    torch.save(payload, path_pkl)


def load_convlstm_embed(path_pkl: str, map_location: str = "cpu"):
    ckpt = torch.load(path_pkl, map_location=map_location)
    cfg = ckpt["config"]

    model = ConvLSTMRegEmbed(
        hid_ch=cfg.get("hid_ch", 32),
        kernel_size=cfg.get("kernel_size", 3),
        n_weekday=cfg["n_weekday"],
        n_t=cfg["n_t"],
        n_x=cfg["n_x"],
        n_y=cfg["n_y"],
        emb_wd=cfg["emb_wd"],
        emb_t=cfg["emb_t"],
        emb_x=cfg["emb_x"],
        emb_y=cfg["emb_y"],
        mlp=cfg.get("mlp", 128),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, cfg
