from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any

from sklearn.metrics import root_mean_squared_error, r2_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -------------------------
# Dense cube utilities (OPT)
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
    Build dense cube cube[d_idx, t, x_idx, y_idx] = log1p(count) or count

    OPT:
    - Use flatten index + np.bincount instead of np.add.at (usually much faster)
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

    d = (df["d"].to_numpy(np.int64) - d_min)
    t = (df["t"].to_numpy(np.int64) - t_min)
    x = (df["x"].to_numpy(np.int64) - x_min)
    y = (df["y"].to_numpy(np.int64) - y_min)

    v = df["count"].to_numpy(np.float32)
    if use_log1p:
        v = np.log1p(v)

    ok = (d >= 0) & (d < D) & (t >= 0) & (t < T) & (x >= 0) & (x < H) & (y >= 0) & (y < W)
    d, t, x, y, v = d[ok], t[ok], x[ok], y[ok], v[ok]

    # Flatten index: (((d*T)+t)*H + x)*W + y
    flat = (((d * T) + t) * H + x) * W + y
    size = D * T * H * W

    acc = np.bincount(flat, weights=v.astype(np.float32, copy=False), minlength=size)
    cube = acc.reshape(D, T, H, W).astype(dtype, copy=False)

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


def pad_cube_hw(cube: np.ndarray, pad: int) -> np.ndarray:
    """
    cube: (D,T,H,W) -> pad H/W with zeros -> (D,T,H+2p,W+2p)
    """
    if pad <= 0:
        return cube
    return np.pad(cube, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant", constant_values=0)


# -------------------------
# Dataset: patch sequence + embeddings (OPT)
# -------------------------

class SeqPatchDatasetEmbed(Dataset):
    """
    Each sample = one row in df, with input sequence:
        x_seq: (L,1,K,K), each step is a patch at SAME t-slot across day-lags.

    OPT:
    - Use padded cube -> patch extraction is pure slicing (no border checks / no manual zero-pad)
    """
    def __init__(
        self,
        df: pd.DataFrame,
        seq_len: int,
        patch_radius: int,
        cube: np.ndarray,
        meta: Dict[str, int],
        *,
        cube_padded: Optional[np.ndarray] = None,
        t_window: int = 1
    ):
        self.df = df.reset_index(drop=True)
        self.seq_len = int(seq_len)
        self.patch_radius = int(patch_radius)
        self.meta = meta

        self.d = self.df["d"].to_numpy(np.int64)
        self.t = self.df["t"].to_numpy(np.int64)
        self.x = self.df["x"].to_numpy(np.int64)
        self.y = self.df["y"].to_numpy(np.int64)

        self.weekday = self.df["weekday"].to_numpy(np.int64)
        self.t_id = self.df["t"].to_numpy(np.int64)
        self.x_id = self.df["x"].to_numpy(np.int64)
        self.y_id = self.df["y"].to_numpy(np.int64)
        self.is_weekend = self.df["is_weekend"].to_numpy(np.float32)

        self.base_log = self.df["base_log"].to_numpy(np.float32) if "base_log" in self.df.columns else None
        if "y_target" in self.df.columns:
            self.target = self.df["y_target"].to_numpy(np.float32)
        else:
            self.target = np.log1p(self.df["count"].to_numpy(np.float32))

        self.cube = cube
        self.cube_pad = cube_padded if cube_padded is not None else pad_cube_hw(cube, self.patch_radius)

        self.K = 2 * self.patch_radius + 1
        self.pad = self.patch_radius

        self.t_window = t_window
        self.C = 2 * t_window + 1

    def __len__(self):
        return len(self.target)

    def __getitem__(self, i):
        d = int(self.d[i]); t = int(self.t[i]); x = int(self.x[i]); y = int(self.y[i])

        d0 = d - self.meta["d_min"]
        t0 = t - self.meta["t_min"]
        x0 = x - self.meta["x_min"]
        y0 = y - self.meta["y_min"]

        # shift due to padding
        xp = x0 + self.pad
        yp = y0 + self.pad

        L = self.seq_len
        K = self.K

        x_seq = np.zeros((L, self.C, K, K), dtype=np.float32)

        # use frames from d0-L ... d0-1, same t0
        # enumerate range(L,0,-1) keeps your original ordering
        for j, k in enumerate(range(L, 0, -1)):
            d_lag = d0 - k
            if d_lag < 0 or d_lag >= self.cube_pad.shape[0]:
                continue
            if t0 < 0 or t0 >= self.cube_pad.shape[1]:
                continue

            for c, dt in enumerate(range(-self.t_window, self.t_window + 1)):

                t_lag = t0 + dt
        
                if t_lag < 0 or t_lag >= self.cube_pad.shape[1]:
                 continue

                frame = self.cube_pad[d_lag, t_lag].astype(np.float32, copy=False)

                x_seq[j, c] = frame[
                    xp - self.pad: xp + self.pad + 1,
                    yp - self.pad: yp + self.pad + 1
                ]

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
        padding = kernel_size // 2
        self.hid_ch = int(hid_ch)
        self.conv = nn.Conv2d(in_ch + hid_ch, 4 * hid_ch, kernel_size, padding=padding)

    def forward(self, x, h, c):
        if h is None:
            B, _, H, W = x.shape
            h = torch.zeros((B, self.hid_ch, H, W), device=x.device, dtype=x.dtype)
            c = torch.zeros((B, self.hid_ch, H, W), device=x.device, dtype=x.dtype)

        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTMEncoder(nn.Module):
    def __init__(self, in_ch: int = 3, hid_ch: int = 32, kernel_size: int = 3):
        super().__init__()
        self.cell = ConvLSTMCell(in_ch=in_ch, hid_ch=hid_ch, kernel_size=kernel_size)

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
    forward(x_seq, weekday, t_id, x_id, y_id, is_weekend) -> (B,)
    """
    def __init__(
        self,
        hid_ch: int = 32,
        kernel_size: int = 3,
        n_weekday: int = 7,
        n_t: int = 48,
        n_x: int = 2000,
        n_y: int = 2000,
        emb_wd: int = 2,
        emb_t: int = 8,
        emb_x: int = 16,
        emb_y: int = 16,
        mlp: int = 128
    ):
        super().__init__()
        self.encoder = ConvLSTMEncoder(in_ch=3, hid_ch=hid_ch, kernel_size=kernel_size)

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
        cx = K // 2
        cy = K // 2
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
# Train / Predict (OPT)
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
    *,
    amp: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int = 4,
    persistent_workers: bool = True,
):
    torch.manual_seed(seed)

    device = next(model.parameters()).device

    if lookup_df is None:
        lookup_df = pd.concat([train_df, val_df], ignore_index=True)

    cube, meta = build_dense_cube(
        lookup_df[["d", "t", "x", "y", "count"]],
        use_log1p=True,
        dtype=cube_dtype,
        bounds=cube_bounds,
    )
    cube_pad = pad_cube_hw(cube, patch_radius)

    tr = train_df.sample(n=min(sample_n, len(train_df)), random_state=seed).reset_index(drop=True)

    ds_tr = SeqPatchDatasetEmbed(tr, seq_len=seq_len, patch_radius=patch_radius, cube=cube, meta=meta, cube_padded=cube_pad, t_window=1)
    ds_va = SeqPatchDatasetEmbed(val_df, seq_len=seq_len, patch_radius=patch_radius, cube=cube, meta=meta, cube_padded=cube_pad, t_window=1)

    dl_tr = DataLoader(
        ds_tr,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(persistent_workers and num_workers > 0),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    dl_va = DataLoader(
        ds_va,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(persistent_workers and num_workers > 0),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss(beta=huber_beta) if loss == "huber" else nn.MSELoss()

    scaler = torch.amp.GradScaler("cuda", enabled=(amp and device.type == "cuda"))

    best_rmse = float("inf")
    best_state = None
    patience, bad = 3, 0

    for ep in range(1, epochs + 1):
        model.train()

        for xseq, wd, tid, xid, yid, isw, base_log, y in dl_tr:
            xseq = xseq.to(device, non_blocking=True)
            wd = wd.to(device, non_blocking=True)
            tid = tid.to(device, non_blocking=True)
            xid = xid.to(device, non_blocking=True)
            yid = yid.to(device, non_blocking=True)
            isw = isw.to(device, non_blocking=True)
            base_log = base_log.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(amp and device.type == "cuda")):
                pred = model(xseq, wd, tid, xid, yid, isw)
                l = loss_fn(pred, y)

            scaler.scale(l).backward()
            scaler.step(opt)
            scaler.update()

        # ---- validation ----
        model.eval()
        preds, ys = [], []
        with torch.no_grad():
            for xseq, wd, tid, xid, yid, isw, base_log, y in dl_va:
                xseq = xseq.to(device, non_blocking=True)
                wd = wd.to(device, non_blocking=True)
                tid = tid.to(device, non_blocking=True)
                xid = xid.to(device, non_blocking=True)
                yid = yid.to(device, non_blocking=True)
                isw = isw.to(device, non_blocking=True)
                base_log = base_log.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=(amp and device.type == "cuda")):
                    pred_log = model(xseq, wd, tid, xid, yid, isw)

                    if use_residual:
                        base = base_log.squeeze(1)
                        pred_log_full = pred_log + base
                        y_log_full = y + base
                    else:
                        pred_log_full = pred_log
                        y_log_full = y

                preds.append(torch.expm1(pred_log_full).float().cpu().numpy())
                ys.append(torch.expm1(y_log_full).float().cpu().numpy())

        yhat = np.concatenate(preds)
        ytrue = np.concatenate(ys)

        rmse = float(root_mean_squared_error(ytrue, yhat))
        r2 = float(r2_score(ytrue, yhat))

        print(f"epoch {ep}: val RMSE={rmse:.4f} | R2={r2:.4f}")

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

    model._cube_meta = meta
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
    *,
    amp: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int = 4,
    persistent_workers: bool = True,
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
    cube_pad = pad_cube_hw(cube, patch_radius)

    ds = SeqPatchDatasetEmbed(df_split, seq_len=seq_len, patch_radius=patch_radius, cube=cube, meta=meta, cube_padded=cube_pad, t_window=1)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(persistent_workers and num_workers > 0),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )

    model.eval()
    preds = []
    with torch.no_grad():
        for xseq, wd, tid, xid, yid, isw, base_log, _y in dl:
            xseq = xseq.to(device, non_blocking=True)
            wd = wd.to(device, non_blocking=True)
            tid = tid.to(device, non_blocking=True)
            xid = xid.to(device, non_blocking=True)
            yid = yid.to(device, non_blocking=True)
            isw = isw.to(device, non_blocking=True)
            base_log = base_log.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=(amp and device.type == "cuda")):
                pred_log = model(xseq, wd, tid, xid, yid, isw)
                if use_residual:
                    pred_log = pred_log + base_log.squeeze(1)

            preds.append(torch.expm1(pred_log).float().cpu().numpy())

    yhat = np.concatenate(preds)
    return np.clip(yhat, 0, None)


def make_df_pred_convlstm_embed(
    df_feat: pd.DataFrame,
    model: nn.Module,
    cfg: Dict[str, Any],
    batch_size: int = 512,
    use_residual: bool = False,
    lookup_df: Optional[pd.DataFrame] = None,
    *,
    amp: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
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
        amp=amp,
        num_workers=num_workers,
        pin_memory=pin_memory,
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