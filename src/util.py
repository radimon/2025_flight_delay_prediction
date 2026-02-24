import numpy as np
import pandas as pd

def circle_radius_by_mass(cand_xy: np.ndarray, p_pred: np.ndarray, x0: float, y0: float, alpha: float):
    """
    固定圓心在 (x0,y0)，找最小半徑 r 使得圓內累積 p_pred >= alpha
    回傳：r、以及圓內 indices
    """
    dist = np.hypot(cand_xy[:,0] - x0, cand_xy[:,1] - y0)
    order = np.argsort(dist)
    cum = np.cumsum(p_pred[order])
    k = int(np.searchsorted(cum, alpha, side="left")) + 1
    k = min(k, len(order))
    r = float(dist[order[k-1]])
    idx_circle = order[:k]
    return r, idx_circle

def true_mass_in_indices(p_true: np.ndarray, idx: np.ndarray) -> float:
    return float(p_true[idx].sum())

def softmax(x: np.ndarray, temp: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=float) / max(temp, 1e-9)
    x = x - np.max(x)
    e = np.exp(x)
    s = e.sum()
    return e / s if s > 0 else np.ones_like(e) / len(e)

def normalize_nonneg(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.clip(x, 0, None)
    s = x.sum()
    return x / s if s > 0 else np.ones_like(x) / len(x)

def dist2d(x1, y1, x2, y2) -> float:
    return float(np.hypot(x1 - x2, y1 - y2))

def select_mass_region(cells_xy: np.ndarray, probs: np.ndarray, alpha: float):
    """
    Return indices of smallest set S such that sum(probs[S]) >= alpha
    """
    order = np.argsort(-probs)
    cum = np.cumsum(probs[order])
    k = int(np.searchsorted(cum, alpha, side="left")) + 1
    idx = order[:k]
    return idx

# 補零
def densify_topk_series(df_raw: pd.DataFrame, top_k=20000):
    """
    df_raw 需要欄位: d,t,x,y,count
    回傳：只含 top_k 個 (x,y,t) 且已補齊所有 d 的 DataFrame
    """
    df = df_raw[["d","t","x","y","count"]].copy()

    # 選最活躍的 (x,y,t)：用總量或出現天數都可以
    key_sum = df.groupby(["x","y","t"])["count"].sum().sort_values(ascending=False)
    top_keys = key_sum.head(top_k).index

    df = df.set_index(["x","y","t"]).loc[top_keys].reset_index()

    dmin, dmax = int(df["d"].min()), int(df["d"].max())
    all_d = np.arange(dmin, dmax + 1, dtype=int)

    out = []
    for (x,y,t), g in df.groupby(["x","y","t"], sort=False):
        g2 = g.set_index("d").reindex(all_d)
        g2["count"] = g2["count"].fillna(0.0)
        g2["d"] = all_d
        g2["x"] = x; g2["y"] = y; g2["t"] = t
        out.append(g2[["d","t","x","y","count"]])

    return pd.concat(out, ignore_index=True)

#切資料
def split_by_day(df, test_days=7, val_days=7):
    max_day = int(df["d"].max())
    test_start = max_day - test_days + 1
    val_start  = test_start - val_days

    train_df = df[df["d"] < val_start].copy()
    val_df   = df[(df["d"] >= val_start) & (df["d"] < test_start)].copy()
    test_df  = df[df["d"] >= test_start].copy()

    return train_df, val_df, test_df