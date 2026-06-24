import os
import datetime as dt
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd

from dotenv import load_dotenv
from openai import OpenAI

from src.geo.grid_to_latlng import GridLatLngMapper, cell_size_m_at
from src.confidence_engine import ConfidenceEngine
from src.parks_choose import build_candidates, choose_parking
from src.parks_routing import create_poi_parking, routing_algorithm
from src.preprocess import prepare_base_df, add_lags_and_rollings
from src.ConvLSTM import load_convlstm_embed, make_df_pred_convlstm_embed

# ------------------------------------------------
# 設定
# ------------------------------------------------

load_dotenv()

CONVLSTM_PATH      = "models/convlstm_sapporo.pkl"
PARQUET_PATH       = "data/processed/sapporo_density.parquet"
PARKING_PARAM_PATH = "data/parking/grid_parking_params.parquet"
START_DATE         = "2019-09-15"
T_BUFFER           = 4
CITY_BOUNDS        = (42.9, 140.7, 43.9, 141.6)

ANCHORS = [
    {"x": 24, "y": 151, "lat": 43.0691833, "lng": 141.3514707},
    {"x": 24, "y": 148, "lat": 43.0794037, "lng": 141.3422559},
    {"x": 26, "y": 153, "lat": 43.0579859, "lng": 141.3540211},
    {"x": 52, "y": 81,  "lat": 43.1982317, "lng": 140.9940363},
    {"x": 50, "y": 41,  "lat": 43.1880641, "lng": 140.7945541},
    {"x": 182,"y": 186, "lat": 43.8536095, "lng": 141.5234881},
]

# 停車場顏色對應（卡片 + 地圖一致）
PARK_COLORS      = ["#E74C3C", "#2980B9", "#8E44AD"]
PARK_FOLIUM_COLS = ["red", "blue", "purple"]
PARK_ICONS       = ["1️⃣", "2️⃣", "3️⃣"]

# ------------------------------------------------
# 快取：模型與資料只載入一次
# ------------------------------------------------

@st.cache_resource
def load_model_and_data():
    df_true = pd.read_parquet(PARQUET_PATH).copy()
    model, cfg = load_convlstm_embed(CONVLSTM_PATH)
    seq_len    = int(cfg["seq_len"])
    df_feat    = add_lags_and_rollings(df_true.copy(), seq_len=seq_len, start_date=START_DATE)
    need_cols  = [f"lag_{k}" for k in range(1, seq_len + 1)]
    df_feat    = df_feat.dropna(subset=need_cols).copy()
    mapper     = GridLatLngMapper(ANCHORS)
    return model, cfg, df_feat, mapper

# ------------------------------------------------
# 輔助函數
# ------------------------------------------------

def get_neighbor_grids(x, y, radius=2):
    return [(x + dx, y + dy)
            for dx in range(-radius, radius + 1)
            for dy in range(-radius, radius + 1)]

def radius_m_to_cells(radius_m, mapper, x0, y0):
    cell_m = cell_size_m_at(mapper, x0, y0)
    return max(1, min(int(np.ceil(radius_m / max(cell_m, 1e-6))), 3))

def filter_prediction_area_range(df, t_start, t_end, grids):
    base    = df[(df["t"] >= t_start) & (df["t"] <= t_end)].copy()
    grid_df = pd.DataFrame(sorted(set(grids)), columns=["x", "y"])
    return base.merge(grid_df, on=["x", "y"], how="inner")

def prob_color(p):
    """依機率回傳綠/橘/紅色標籤"""
    if p >= 0.6:
        return "#27AE60", "高"
    elif p >= 0.3:
        return "#F39C12", "中"
    else:
        return "#E74C3C", "低"

# ------------------------------------------------
# CSS
# ------------------------------------------------

st.set_page_config(page_title="AI Smart Parking", layout="wide", page_icon="🅿️")

st.markdown("""
<style>
/* 全體背景 */
[data-testid="stAppViewContainer"] {
    background: #0F1117;
}
[data-testid="stSidebar"] { display: none; }

/* 標題區 */
.hero {
    background: linear-gradient(135deg, #1a1f2e 0%, #16213e 100%);
    border-radius: 16px;
    padding: 32px 36px 24px;
    margin-bottom: 28px;
    border: 1px solid #2a3050;
}
.hero h1 {
    font-size: 2rem;
    font-weight: 700;
    color: #FFFFFF;
    margin: 0 0 6px;
}
.hero p {
    color: #8892a4;
    font-size: 0.95rem;
    margin: 0;
}

/* 查詢區塊 */
.query-box {
    background: #1a1f2e;
    border-radius: 12px;
    padding: 24px 28px;
    margin-bottom: 24px;
    border: 1px solid #2a3050;
}
.section-title {
    color: #A8B4C8;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 14px;
}

/* 停車場卡片 */
.park-card {
    background: #1a1f2e;
    border-radius: 12px;
    padding: 18px 22px;
    margin-bottom: 14px;
    border: 1px solid #2a3050;
    transition: border-color 0.2s;
}
.park-card:hover { border-color: #4a6fa5; }
.park-card-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 14px;
}
.park-rank {
    font-size: 1.3rem;
}
.park-name {
    font-size: 1.05rem;
    font-weight: 600;
    color: #E8EDF5;
}
.park-badge {
    margin-left: auto;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
}
.park-stats {
    display: flex;
    gap: 24px;
}
.stat-item {
    display: flex;
    flex-direction: column;
    gap: 2px;
}
.stat-label {
    font-size: 0.72rem;
    color: #606880;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.stat-value {
    font-size: 1.0rem;
    font-weight: 600;
    color: #C8D4E8;
}
.prob-bar-bg {
    height: 5px;
    background: #2a3050;
    border-radius: 3px;
    margin-top: 10px;
}
.prob-bar-fill {
    height: 5px;
    border-radius: 3px;
}

/* 地圖區塊 */
.map-section {
    background: #1a1f2e;
    border-radius: 12px;
    padding: 20px;
    border: 1px solid #2a3050;
    margin-top: 24px;
}
.map-title {
    color: #A8B4C8;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 14px;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# Hero
# ------------------------------------------------

st.markdown("""
<div class="hero">
    <h1>🅿️ AI Smart Parking Navigation</h1>
    <p>ConvLSTM-based crowd-flow prediction · LLM query parsing · Multi-criteria route optimization · Sapporo, Japan</p>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------
# API key 檢查 & 模型載入
# ------------------------------------------------

api_key = os.getenv("GOOGLE_MAPS_API_KEY")
if not api_key:
    st.error("❌ 未找到 GOOGLE_MAPS_API_KEY，請確認 .env 檔案設定")
    st.stop()

with st.spinner("載入模型與資料中..."):
    model, cfg, df_feat, mapper = load_model_and_data()

# ------------------------------------------------
# 查詢輸入
# ------------------------------------------------

st.markdown('<div class="section-title">輸入查詢</div>', unsafe_allow_html=True)

col_input, col_btn = st.columns([5, 1])
with col_input:
    user_input = st.text_input(
        label="query",
        label_visibility="collapsed",
        placeholder="星期六下午2點從札幌車站到丘珠空港"
    )
with col_btn:
    run_btn = st.button("🔍 搜尋", type="primary", use_container_width=True)

if run_btn and user_input:

    with st.spinner("解析查詢中..."):
        client     = OpenAI()
        engine_tmp = ConfidenceEngine(
            df_feat, mapper,
            openai_client=client,
            google_api_key=api_key
        )
        q        = engine_tmp.parse_route_query_llm(user_input)
        resolved = engine_tmp.resolve_two_locations(q, CITY_BOUNDS)

    if resolved is None:
        st.error("❌ 地點解析失敗，請確認輸入的地名是否正確")
        st.stop()

    start_info, dest_info = resolved
    start_x, start_y     = start_info["x"], start_info["y"]
    start_lat, start_lng = start_info["lat"], start_info["lng"]
    dest_x, dest_y       = dest_info["x"], dest_info["y"]
    dest_lat, dest_lng   = dest_info["lat"], dest_info["lng"]
    target_t             = engine_tmp.time_to_slot(q.hhmm)

    # 解析結果小標籤
    c1, c2, c3 = st.columns(3)
    c1.success(f"🟢 起點　{q.start_place}")
    c2.success(f"🔴 終點　{q.dest_place}")
    c3.info(f"🕐 時段　{q.hhmm}　（slot {target_t}）")

    # 人流預測
    with st.spinner("產出人流預測中..."):
        radius_cells = radius_m_to_cells(float(q.radius_m or 400), mapper, dest_x, dest_y)
        target_grids = (get_neighbor_grids(start_x, start_y, radius=radius_cells) +
                        get_neighbor_grids(dest_x,  dest_y,  radius=radius_cells))
        df_feat_small = filter_prediction_area_range(
            df_feat,
            t_start=target_t,
            t_end=min(target_t + T_BUFFER, 47),
            grids=target_grids
        )
        if df_feat_small.empty:
            st.error("該時間與範圍內沒有可用預測資料")
            st.stop()
        df_pred = make_df_pred_convlstm_embed(df_feat_small, model, cfg)

    # 停車機率計算
    with st.spinner("計算停車需求與停車機率..."):
        df_parks, df_with_a, df_prob = create_poi_parking(
            df_pred, 42.9, 140.7, 43.9, 141.6,
            mapper, START_DATE, refetch=False
        )

    # 候選停車場
    from src.parks_routing import aggregate_df_prob_same_weekday
    query_date = dt.datetime.today()
    w0         = (query_date.weekday() + 1) % 7
    df_prob_w  = aggregate_df_prob_same_weekday(df_prob, agg="median")

    prefs = {
        "risk_aversion": 0.1,
        "min_prob": 0.01,
        "max_walk_min": 120,
        "max_detour_min": 120,
        "w_prob": 0.4,
        "w_walk": 0.2,
        "w_detour": 0.2,
        "w_drive": 0.3,
        "w_price": 0.5,
    }

    cand  = build_candidates(df_prob_w, mapper=mapper, w0=w0, t0=target_t,
                             O=(start_x, start_y), D=(dest_x, dest_y))
    parks = choose_parking(cand, prefs=prefs, top_k=3)

    if parks.empty or "score" not in parks.columns:
        st.error("找不到符合條件的停車場")
        st.stop()

    parks_sorted = parks.sort_values("score", ascending=False).reset_index(drop=True)

    st.session_state["parks_sorted"] = parks_sorted
    st.session_state["query_info"]   = {
        "s_lat": start_lat, "s_lng": start_lng,
        "d_lat": dest_lat,  "d_lng": dest_lng,
        "w0": w0, "target_t": target_t,
    }

# ------------------------------------------------
# 停車場卡片 + 勾選
# ------------------------------------------------

if "parks_sorted" in st.session_state:
    parks_sorted = st.session_state["parks_sorted"]
    info         = st.session_state["query_info"]

    st.markdown('<div class="section-title" style="margin-top:28px;">推薦停車場</div>',
                unsafe_allow_html=True)
    st.markdown("<p style='color:#606880; font-size:0.88rem; margin-bottom:16px;'>"
                "勾選想納入路線的停車場，系統將串聯開車路線依序嘗試。</p>",
                unsafe_allow_html=True)

    selected = []

    for i, row in parks_sorted.iterrows():
        p          = float(row["p_avail"])
        badge_col, badge_txt = prob_color(p)
        card_color = PARK_COLORS[i % len(PARK_COLORS)]

        st.markdown(f"""
        <div class="park-card" style="border-left: 4px solid {card_color};">
            <div class="park-card-header">
                <span class="park-rank">{PARK_ICONS[i % len(PARK_ICONS)]}</span>
                <span class="park-name">停車場 #{i+1}</span>
                <span class="park-badge" style="background:{badge_col}22; color:{badge_col};">
                    可用性：{badge_txt}
                </span>
            </div>
            <div class="park-stats">
                <div class="stat-item">
                    <span class="stat-label">可停機率</span>
                    <span class="stat-value" style="color:{badge_col};">{p:.1%}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">步行時間</span>
                    <span class="stat-value">{row['walk_time']:.1f} 分</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">繞路時間</span>
                    <span class="stat-value">{row['detour_time']:.1f} 分</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">綜合評分</span>
                    <span class="stat-value">{row['score']:.3f}</span>
                </div>
            </div>
            <div class="prob-bar-bg">
                <div class="prob-bar-fill" style="width:{p*100:.1f}%; background:{badge_col};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.checkbox(f"選擇停車場 #{i+1}", key=f"park_{i}"):
            selected.append(i)

    st.markdown("<br>", unsafe_allow_html=True)
    gen_btn = st.button("🗺️ 產生路線地圖", type="primary", use_container_width=True)

    if gen_btn and selected:

        selected_parks = parks_sorted.iloc[selected].copy()

        with st.spinner("呼叫 Google Maps API 產生路線中..."):
            from src.parks_routing import (
                get_google_route, draw_route,
                next_datetime_for_weekday,
                SLOT_MIN
            )
            import folium

            s_lat    = info["s_lat"]; s_lng    = info["s_lng"]
            d_lat    = info["d_lat"]; d_lng    = info["d_lng"]
            w0       = info["w0"];    target_t = info["target_t"]

            fmap = folium.Map(location=[d_lat, d_lng], zoom_start=14)

            folium.Marker([s_lat, s_lng], popup="起點",
                          icon=folium.Icon(color="green", icon="play")).add_to(fmap)
            folium.Marker([d_lat, d_lng], popup="終點",
                          icon=folium.Icon(color="red", icon="flag")).add_to(fmap)

            departure_time = next_datetime_for_weekday(w0, target_t, SLOT_MIN)

            park_coords = [(float(r["lat"]), float(r["lng"]))
                           for _, r in selected_parks.iterrows()]

            # 串聯開車路線
            drive_waypoints  = park_coords[:-1] if len(park_coords) > 1 else None
            last_plat, last_plng = park_coords[-1]

            coords_drive, dist_drive, dur_drive = get_google_route(
                api_key, s_lat, s_lng, last_plat, last_plng,
                mode="driving", departure_time=departure_time,
                waypoints=drive_waypoints
            )

            draw_route(fmap, coords_drive, "#2C3E50",
                       tooltip=f"開車總計: {dur_drive/60:.1f}分 / {dist_drive/1000:.1f}km",
                       weight=6)

            all_points = [(s_lat, s_lng), (d_lat, d_lng)] + list(coords_drive)

            for idx, (i, row) in enumerate(selected_parks.iterrows()):
                plat  = float(row["lat"])
                plng  = float(row["lng"])
                color = PARK_FOLIUM_COLS[idx % len(PARK_FOLIUM_COLS)]
                rank  = i + 1

                coords_walk, dist_walk, dur_walk = get_google_route(
                    api_key, plat, plng, d_lat, d_lng, mode="walking"
                )

                draw_route(fmap, coords_walk, color,
                           tooltip=f"步行至終點: {dur_walk/60:.1f}分",
                           dash_array="6,10", weight=4)

                folium.Marker(
                    [plat, plng],
                    icon=folium.Icon(color=color, icon="car", prefix="fa"),
                    tooltip=folium.Tooltip(
                        f"<b>停車場 #{rank}</b><br>"
                        f"🚶 步行 {dur_walk/60:.0f} 分（{dist_walk}m）<br>"
                        f"🅿️ 可停機率 {row['p_avail']:.0%}",
                        permanent=True, direction="top", offset=(0, -8)
                    ),
                    popup=(f"<b>停車場 #{rank}</b><br>"
                           f"可停機率: {row['p_avail']:.2%}<br>"
                           f"步行至終點: {dur_walk/60:.0f} 分 / {dist_walk}m")
                ).add_to(fmap)

                all_points.extend([(plat, plng)] + list(coords_walk))

            lats, lngs = zip(*all_points)
            fmap.fit_bounds([[min(lats), min(lngs)], [max(lats), max(lngs)]])

            fmap.save("parking_route.html")
            html_content = open("parking_route.html", "r", encoding="utf-8").read()

        st.markdown('<div class="map-title" style="margin-top:28px;">導航地圖</div>',
                    unsafe_allow_html=True)
        components.html(html_content, height=620, scrolling=False)
        st.success("✓ 地圖已產生，並儲存至 parking_route.html")

    elif gen_btn:
        st.warning("請至少勾選一個停車場")