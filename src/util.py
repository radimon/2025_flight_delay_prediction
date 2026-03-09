import numpy as np
import pandas as pd
import folium
from branca.colormap import linear
from branca.element import Template, MacroElement
from folium.plugins import HeatMap

from src.geo.grid_to_latlng import cell_size_m_at


# ============================================================
# 1) Math helpers
# ============================================================

def circle_radius_by_mass(cand_xy, p_pred, x0, y0, alpha):
    dist = np.hypot(cand_xy[:, 0] - x0, cand_xy[:, 1] - y0)
    order = np.argsort(dist)
    cum = np.cumsum(p_pred[order])
    k = int(np.searchsorted(cum, alpha, side="left")) + 1
    k = min(k, len(order))
    r = float(dist[order[k - 1]])
    idx_circle = order[:k]
    return r, idx_circle


def normalize_nonneg(x):
    x = np.asarray(x, dtype=float)
    x = np.clip(x, 0, None)
    s = x.sum()
    return x / s if s > 0 else np.ones_like(x) / len(x)

def dataset_stats(y):
    y = np.asarray(y, float)
    return {
        "n": len(y),
        "zero_ratio": float((y==0).mean()),
        "mean": float(y.mean()),
        "std": float(y.std()),
        "p90": float(np.percentile(y, 90)),
        "p99": float(np.percentile(y, 99)),
    }


# ============================================================
# 2) Data preprocess
# ============================================================
def softmax(x: np.ndarray, temp: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=float) / max(temp, 1e-9)
    x = x - np.max(x)
    e = np.exp(x)
    s = e.sum()
    return e / s if s > 0 else np.ones_like(e) / len(e)

def select_mass_region(cells_xy: np.ndarray, probs: np.ndarray, alpha: float):
    """
    Return indices of smallest set S such that sum(probs[S]) >= alpha
    """
    order = np.argsort(-probs)
    cum = np.cumsum(probs[order])
    k = int(np.searchsorted(cum, alpha, side="left")) + 1
    idx = order[:k]
    return idx

# ============================================================
# 3) UI map (With Precise Positioning Support)
# ============================================================

def plot_query_on_map(
    resp,
    mapper,
    df_pred,
    city_bounds_dict,
    save_path="predict_confidence2.html",
    add_heatmap=False,
    orig_lat=None,  # 新增：Google 得到的精確緯度
    orig_lng=None   # 新增：Google 得到的精確經度
):
    """
    視覺化預測結果：
    - 如果有 orig_lat/lng，會 Pin 在真實位置
    - 點點顏色表示 score
    - 信心圓以真實位置為圓心繪製（若無則用格網中心）
    """

    x0 = float(resp["query"]["center_grid"]["x"])
    y0 = float(resp["query"]["center_grid"]["y"])
    d = int(resp["query"]["d_used"])
    t = int(resp["query"]["t_slot"])

    # 從格網推算的中心點（格子中心）
    center_box = mapper.transform(pd.DataFrame([{"x": x0, "y": y0}])).iloc[0]
    lat_box, lng_box = float(center_box.lat), float(center_box.lng)

    # 決定最終顯示的中心點 (優先使用原始座標)
    draw_lat = orig_lat if orig_lat is not None else lat_box
    draw_lng = orig_lng if orig_lng is not None else lng_box

    # cell size around center
    cell_m = float(cell_size_m_at(mapper, x0, y0))
    cell_area_km2 = (cell_m * cell_m) / 1e6

    # predicted slice
    sl = df_pred[(df_pred["d"] == d) & (df_pred["t"] == t)][["x", "y", "score"]].copy()
    if sl.empty:
        raise ValueError("pred_slice is empty for this (d,t).")

    # window 限制
    c95 = next((c for c in resp["circles"] if abs(float(c["alpha"]) - 0.95) < 1e-6), None)
    win = int(c95["window_cells"]) if (c95 and "window_cells" in c95) else int(resp["query"].get("radius_cells", 20))

    dist_grid = np.hypot(sl["x"].to_numpy() - x0, sl["y"].to_numpy() - y0)
    sl = sl.loc[dist_grid <= win + 1e-9].copy()

    # 重算 95% 圓以做 mask
    cand_xy = sl[["x", "y"]].to_numpy()
    p_pred = normalize_nonneg(sl["score"].to_numpy())
    r95, idx95 = circle_radius_by_mass(cand_xy, p_pred, x0, y0, 0.95)
    achieved95 = float(p_pred[idx95].sum())

    # 建立地圖
    m = folium.Map(location=[draw_lat, draw_lng], zoom_start=15, control_scale=True)

    # 城市範圍邊界
    folium.Rectangle(
        bounds=[
            [city_bounds_dict["lat_min"], city_bounds_dict["lng_min"]],
            [city_bounds_dict["lat_max"], city_bounds_dict["lng_max"]],
        ],
        color="black", weight=2, fill=False, tooltip="CITY BOUNDS"
    ).add_to(m)

    # --- Marker 繪製邏輯 ---
    # 1. 畫出 Google 的精確位置 (紅色 Pin)
    if orig_lat is not None:
        folium.Marker(
            [orig_lat, orig_lng],
            tooltip="Google 精確定位",
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(m)
        
    # 2. 畫出對應的格網中心點 (小藍點，幫助 Debug 位移)
    folium.CircleMarker(
        [lat_box, lng_box],
        radius=3,
        color="blue",
        fill=True,
        tooltip=f"模型格網中心 ({x0:.0f}, {y0:.0f})"
    ).add_to(m)

    # 繪製信心圓
    alpha_color = {0.5: "red", 0.8: "orange", 0.95: "blue"}
    for c in sorted(resp["circles"], key=lambda c: float(c["alpha"])):
        a = float(c["alpha"])
        a_key = min(alpha_color.keys(), key=lambda k: abs(k - a))
        color = alpha_color[a_key]
        radius_m = float(c["radius_cells"]) * cell_m
        dash = {0.5: None, 0.8: "6,6", 0.95: "1,6"}
        
        folium.Circle(
            location=[draw_lat, draw_lng], # 以精確座標為圓心
            radius=radius_m,
            dash_array=dash[a_key],
            color=color,
            fill=True,
            fill_opacity=0.08,
            popup=f"Confidence {a*100:.0f}%: radius≈{radius_m:.0f}m",
        ).add_to(m)

    # 顏色映射與預測點繪製
    smin, smax = float(sl["score"].quantile(0.05)), float(sl["score"].quantile(0.95))
    sl["score_vis"] = sl["score"].clip(smin, smax)
    cmap = linear.YlOrRd_09.scale(smin, smax)

    ll = mapper.transform(sl[["x", "y"]].copy())
    ll = ll.assign(score=sl["score"].values, x=sl["x"].values, y=sl["y"].values)
    dist_to_center = np.hypot(ll["x"].to_numpy() - x0, ll["y"].to_numpy() - y0)
    in95 = dist_to_center <= r95 + 1e-9

    points_group = folium.FeatureGroup(name="Grid Predictions (Score)")
    for i in range(len(ll)):
        lat, lng = float(ll.iloc[i].lat), float(ll.iloc[i].lng)
        score = float(ll.iloc[i].score)
        color = cmap(score)
        rr = 2 + 6 * (score - smin) / (smax - smin + 1e-12)

        folium.CircleMarker(
            location=[lat, lng],
            radius=float(rr),
            color="#000000" if in95[i] else "#999999",
            weight=1.5 if in95[i] else 0.5,
            fill=True,
            fill_color=color,
            fill_opacity=0.8 if in95[i] else 0.2,
        ).add_to(points_group)
    points_group.add_to(m)

    # UI Panel (摘要資訊)
    avg_score_win = float(sl["score"].mean())
    avg_density_win_km2 = avg_score_win / (cell_area_km2 + 1e-12)
    time_str = f"d={d}, t={t} ({t//2:02d}:{(t%2)*30:02d})"
    
    info_html = f"""
    {{% macro html(this, kwargs) %}}
    <div style="position: fixed; top: 12px; left: 12px; z-index: 9999; background: white; 
                border: 2px solid grey; border-radius: 6px; padding: 10px; font-size: 12px;">
      <b>Crowd Analysis Summary</b><br>
      <b>Location</b>: {resp['query']['start_place']}<br>
      <b>Time</b>: {time_str}<br>
      <hr>
      <b>Cell Size</b>: ~{cell_m:.0f}m ({cell_area_km2:.3f} km²)<br>
      <b>Avg Score</b>: {avg_score_win:.3f} /cell<br>
      <b>Avg Density</b>: {avg_density_win_km2:.1f} /km²
    </div>
    {{% endmacro %}}
    """
    macro = MacroElement()
    macro._template = Template(info_html)
    m.get_root().add_child(macro)

    cmap.caption = "Crowd Prediction Score"
    cmap.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    m.save(save_path)
    print(f"✓ Map saved to: {save_path}")