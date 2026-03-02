import numpy as np
import pandas as pd
import folium
from branca.colormap import linear
from branca.element import Template, MacroElement
from folium.plugins import HeatMap

# === 1. 數學與資料處理工具 (保留你原本提供的內容) ===

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

def normalize_nonneg(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.clip(x, 0, None)
    s = x.sum()
    return x / s if s > 0 else np.ones_like(x) / len(x)

def softmax(x: np.ndarray, temp: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=float) / max(temp, 1e-9)
    x = x - np.max(x)
    e = np.exp(x)
    s = e.sum()
    return e / s if s > 0 else np.ones_like(e) / len(e)

def densify_topk_series(df_raw: pd.DataFrame, top_k=20000):
    """補齊時間序列缺失值 (原本用於訓練前處理)"""
    df = df_raw[["d","t","x","y","count"]].copy()
    key_sum = df.groupby(["x","y","t"])["count"].sum().sort_values(ascending=False)
    top_keys = key_sum.head(top_k).index
    df = df.set_index(["x","y","t"]).loc[top_keys].reset_index()
    dmin, dmax = int(df["d"].min()), int(df["d"].max())
    all_d = np.arange(dmin, dmax + 1, dtype=int)
    out = []
    for (x,y,t), g in df.groupby(["x","y","t"], sort=False):
        g2 = g.set_index("d").reindex(all_d)
        g2["count"] = g2["count"].fillna(0.0); g2["d"] = all_d
        g2["x"] = x; g2["y"] = y; g2["t"] = t
        out.append(g2[["d","t","x","y","count"]])
    return pd.concat(out, ignore_index=True)

def split_by_day(df, test_days=7, val_days=7):
    """切割訓練、驗證、測試集"""
    max_day = int(df["d"].max())
    test_start = max_day - test_days + 1
    val_start  = test_start - val_days
    return df[df["d"] < val_start].copy(), \
           df[(df["d"] >= val_start) & (df["d"] < test_start)].copy(), \
           df[df["d"] >= test_start].copy()

# === 2. 繪圖工具 (從 Notebook 移植過來) ===

def plot_query_on_map(resp, mapper, df_pred, city_bounds_dict):
    """
    將 ConfidenceEngine 的結果繪製在 Folium 地圖上
    """
    x0 = resp["query"]["center_grid"]["x"]
    y0 = resp["query"]["center_grid"]["y"]
    d  = resp["query"]["d_used"]
    t  = resp["query"]["t_slot"]

    # 轉換經緯度
    center = mapper.transform(pd.DataFrame([{"x":x0,"y":y0}])).iloc[0]
    lat0, lng0 = float(center.lat), float(center.lng)

    m = folium.Map(location=[lat0, lng0], zoom_start=14)

    # 畫城市邊界 (CITY_BOUNDS)
    folium.Rectangle(
        bounds=[[city_bounds_dict["lat_min"], city_bounds_dict["lng_min"]], 
                [city_bounds_dict["lat_max"], city_bounds_dict["lng_max"]]],
        color="black", weight=2, fill=False
    ).add_to(m)

    # 畫信心圓圈
    colors = {0.5: "red", 0.8: "orange", 0.95: "blue"}
    # 這裡假設一格約 250m，或你可以呼叫 cell_size_m_at
    for c in resp["circles"]:
        folium.Circle(
            location=[lat0, lng0],
            radius=float(c["radius_cells"] * 250), 
            color=colors.get(c["alpha"], "black"),
            fill=True, fill_opacity=0.1,
            popup=f"alpha={c['alpha']}"
        ).add_to(m)

    # 繪製熱力圖層 (抓取預測切片)
    sl = df_pred[(df_pred["d"] == d) & (df_pred["t"] == t)].copy()
    if not sl.empty:
        ll = mapper.transform(sl[["x","y"]].copy())
        heat_data = [[ll.iloc[i].lat, ll.iloc[i].lng, float(sl.iloc[i].score)] for i in range(len(ll))]
        HeatMap(heat_data, radius=15, blur=20).add_to(m)

    m.save("predict_confidence.html")