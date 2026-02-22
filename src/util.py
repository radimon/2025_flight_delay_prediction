import numpy as np

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