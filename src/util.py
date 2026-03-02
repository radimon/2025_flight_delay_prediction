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


# ============================================================
# 2) Old UI map (points + panels + legend)
# ============================================================

def plot_query_on_map(
    resp,
    mapper,
    df_pred,
    city_bounds_dict,
    save_path="predict_confidence.html",
    add_heatmap=False,   # 你想要舊 UI：預設 False
):
    """
    舊版 UI：
    - 點點顏色表示 score
    - 左上 summary panel
    - 右上 legend
    - 信心圓
    """

    x0 = float(resp["query"]["center_grid"]["x"])
    y0 = float(resp["query"]["center_grid"]["y"])
    d = int(resp["query"]["d_used"])
    t = int(resp["query"]["t_slot"])

    # grid -> lat/lng
    center = mapper.transform(pd.DataFrame([{"x": x0, "y": y0}])).iloc[0]
    lat0, lng0 = float(center.lat), float(center.lng)

    # cell size around center
    cell_m = float(cell_size_m_at(mapper, x0, y0))
    cell_area_km2 = (cell_m * cell_m) / 1e6

    # predicted slice for (d,t)
    sl = df_pred[(df_pred["d"] == d) & (df_pred["t"] == t)][["x", "y", "score"]].copy()
    if sl.empty:
        raise ValueError("pred_slice is empty for this (d,t).")

    # 用 95% 圈的 window_cells 來限制顯示範圍（如果有）
    c95 = next((c for c in resp["circles"] if abs(float(c["alpha"]) - 0.95) < 1e-6), None)
    if c95 is not None and "window_cells" in c95:
        win = int(c95["window_cells"])
    else:
        # 沒有 window_cells 就用 radius_cells 當 fallback
        win = int(resp["query"].get("radius_cells", 20))

    dist_grid = np.hypot(sl["x"].to_numpy() - x0, sl["y"].to_numpy() - y0)
    sl = sl.loc[dist_grid <= win + 1e-9].copy()
    if sl.empty:
        raise ValueError("No cells in the selected window. Try larger radius/window.")

    # 用同一批候選點重算 95% 圓（方便做 in95 mask）
    cand_xy = sl[["x", "y"]].to_numpy()
    p_pred = normalize_nonneg(sl["score"].to_numpy())
    r95, idx95 = circle_radius_by_mass(cand_xy, p_pred, x0, y0, 0.95)
    achieved95 = float(p_pred[idx95].sum())

    # Map
    m = folium.Map(location=[lat0, lng0], zoom_start=14, control_scale=True)

    # City bounds rectangle (lat/lng)
    folium.Rectangle(
        bounds=[
            [city_bounds_dict["lat_min"], city_bounds_dict["lng_min"]],
            [city_bounds_dict["lat_max"], city_bounds_dict["lng_max"]],
        ],
        color="black",
        weight=2,
        fill=False,
        tooltip="CITY QUERY BOUNDS",
    ).add_to(m)

    # Center marker
    folium.Marker([lat0, lng0], tooltip=f"center grid=({x0:.0f},{y0:.0f})").add_to(m)

    # Circles
    alpha_color = {0.5: "red", 0.8: "orange", 0.95: "blue"}
    for c in resp["circles"]:
        a = float(c["alpha"])
        # 允許浮點誤差
        a_key = min(alpha_color.keys(), key=lambda k: abs(k - a))
        color = alpha_color[a_key]

        radius_m = float(c["radius_cells"]) * cell_m
        folium.Circle(
            location=[lat0, lng0],
            radius=radius_m,
            color=color,
            fill=True,
            fill_opacity=0.10,
            popup=f"alpha={a:.2f}  r_cells={float(c['radius_cells']):.2f}  r_m≈{radius_m:.0f}",
        ).add_to(m)

    # Colormap for points
    smin, smax = float(sl["score"].min()), float(sl["score"].max())
    cmap = linear.YlOrRd_09.scale(smin, smax)

    # grid -> lat/lng for points
    ll = mapper.transform(sl[["x", "y"]].copy())
    ll = ll.assign(score=sl["score"].values, x=sl["x"].values, y=sl["y"].values)

    # in95 mask
    dist_to_center = np.hypot(ll["x"].to_numpy() - x0, ll["y"].to_numpy() - y0)
    in95 = dist_to_center <= r95 + 1e-9

    # Optional heatmap layer (關掉才是你要的舊 UI)
    if add_heatmap:
        heat_data = [[float(ll.iloc[i].lat), float(ll.iloc[i].lng), float(ll.iloc[i].score)] for i in range(len(ll))]
        HeatMap(
            heat_data,
            name="Density HeatMap (score)",
            min_opacity=0.20,
            radius=22,
            blur=28,
            max_zoom=16,
        ).add_to(m)

    # Points layer
    points_group = folium.FeatureGroup(name="Predicted points (score)", show=True)
    for i in range(len(ll)):
        lat, lng = float(ll.iloc[i].lat), float(ll.iloc[i].lng)
        score = float(ll.iloc[i].score)
        color = cmap(score)

        rr = 2 + 6 * (score - smin) / (smax - smin + 1e-12)

        folium.CircleMarker(
            location=[lat, lng],
            radius=float(rr),
            color="#000000" if in95[i] else "#666666",
            weight=2 if in95[i] else 1,
            fill=True,
            fill_color=color,
            fill_opacity=0.9 if in95[i] else 0.30,
            popup=f"score={score:.3f}  in95={bool(in95[i])}",
        ).add_to(points_group)
    points_group.add_to(m)

    # Summary stats
    avg_score_win = float(sl["score"].mean())
    avg_density_win_km2 = avg_score_win / (cell_area_km2 + 1e-12)
    avg_score_95 = float(sl.loc[in95, "score"].mean()) if in95.any() else np.nan
    avg_density_95_km2 = (avg_score_95 / (cell_area_km2 + 1e-12)) if np.isfinite(avg_score_95) else np.nan

    time_str = f"d={d}, t={t} (approx {t//2:02d}:{(t%2)*30:02d})"
    info_html = f"""
    {{% macro html(this, kwargs) %}}
    <div style="
        position: fixed;
        top: 12px; left: 12px;
        z-index: 9999;
        background: rgba(255,255,255,0.92);
        border: 1px solid #999;
        border-radius: 10px;
        padding: 10px 12px;
        font-size: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        max-width: 340px;
    ">
      <div style="font-weight:700; font-size:13px; margin-bottom:6px;">Crowd / Density Summary</div>
      <div><b>Query</b>: center=({x0:.0f},{y0:.0f}) | {time_str}</div>
      <div><b>Window</b>: win={win} cells (≈ {win*cell_m:.0f} m)</div>
      <hr style="margin:8px 0;">
      <div><b>Avg score (window)</b>: {avg_score_win:.3f} /cell</div>
      <div><b>Avg density (window)</b>: {avg_density_win_km2:.1f} /km²</div>
      <div style="margin-top:6px;"><b>95% circle</b>: r95={r95:.2f} cells (≈ {r95*cell_m:.0f} m), achieved={achieved95:.3f}</div>
      <div><b>Avg score (in 95%)</b>: {avg_score_95:.3f} /cell</div>
      <div><b>Avg density (in 95%)</b>: {avg_density_95_km2:.1f} /km²</div>
      <div style="color:#666; margin-top:6px;">
        Note: grid cell is ~{cell_m:.0f}m × {cell_m:.0f}m.
      </div>
    </div>
    {{% endmacro %}}
    """
    macro = MacroElement()
    macro._template = Template(info_html)
    m.get_root().add_child(macro)

    # Legend + layer control
    cmap.caption = "predicted score"
    cmap.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    m.save(save_path)