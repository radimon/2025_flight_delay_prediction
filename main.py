import os
import requests
import polyline
import folium
import pandas as pd
import numpy as np
import datetime as dt

from dotenv import load_dotenv
from openai import OpenAI

from src.geo.grid_to_latlng import GridLatLngMapper, cell_size_m_at
from src.parking_engine import ParkingEngine
from src.routing_engine import RoutingEngine
from src.confidence_engine import ConfidenceEngine

from src.parks_choose import *
from src.parks_create import *
from src.parks_routing import *

from src.preprocess import *
from src.ConvLSTM import *

CONVLSTM_PATH = "models/convlstm_sapporo.pkl"
PARQUET_PATH = "data/processed/sapporo_density.parquet"
SAVED_FILLED_ZERO_DATA_PATH = "data/processed/sapporo_density.parquet"
PARKING_PARAM_PATH = "data/parking/grid_parking_params.parquet"

NEED_TRAIN = False
REBUILD_DATA = False

START_DATE = "2019-09-15"
T_BUFFER = 4

# ------------------------------------------------
# Google Directions API
# ------------------------------------------------

def get_google_route(api_key, origin_lat, origin_lng, dest_lat, dest_lng):

    url = "https://maps.googleapis.com/maps/api/directions/json"

    params = {
        "origin": f"{origin_lat},{origin_lng}",
        "destination": f"{dest_lat},{dest_lng}",
        "key": api_key
    }

    r = requests.get(url, params=params, timeout=15)
    data = r.json()

    if data.get("status") != "OK":
        raise RuntimeError(f"Google API error: {data.get('status')}")

    route = data["routes"][0]
    poly = route["overview_polyline"]["points"]

    coords = polyline.decode(poly)

    leg = route["legs"][0]
    dist = leg["distance"]["value"]
    dur = leg["duration"]["value"]

    return coords, dist, dur


def draw_route(fmap, coords, color):

    folium.PolyLine(
        coords,
        color=color,
        weight=5,
        opacity=0.8
    ).add_to(fmap)


# ------------------------------------------------
# Grid helpers
# ------------------------------------------------

def get_neighbor_grids(x, y, radius=2):

    grids = []

    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            grids.append((x + dx, y + dy))

    return grids


def radius_m_to_cells(radius_m, mapper, x0, y0):

    cell_m = cell_size_m_at(mapper, x0, y0)

    raw = int(np.ceil(radius_m / max(cell_m, 1e-6)))

    return max(1, min(raw, 3))


def filter_prediction_area_range(df, t_start, t_end, grids):

    base = df[(df["t"] >= t_start) & (df["t"] <= t_end)].copy()

    grid_df = pd.DataFrame(sorted(set(grids)), columns=["x", "y"])

    return base.merge(grid_df, on=["x", "y"], how="inner")


# ------------------------------------------------
# Main
# ------------------------------------------------

def main():

    load_dotenv()

    api_key = os.getenv("GOOGLE_MAPS_API_KEY")

    if not api_key:
        raise ValueError("❌ Google Maps API key not found")

    print("\n===== AI Parking Navigation =====\n")

    user_input = input("請輸入查詢：").strip()

    print("\n正在載入模型與原始資料...")

    df_raw = prepare_base_df(PARQUET_PATH, remove_sentinel=True)

    if REBUILD_DATA:

        df_true = densify_topk_series(df_raw, top_k=20000)

        save_filled_zero_data(df_true, SAVED_FILLED_ZERO_DATA_PATH)

    else:

        df_true = pd.read_parquet(SAVED_FILLED_ZERO_DATA_PATH).copy()

    model, cfg = load_convlstm_embed(CONVLSTM_PATH)

    seq_len = int(cfg["seq_len"])

    print("執行特徵工程中...")

    df_feat = add_lags_and_rollings(
        df_true.copy(),
        seq_len=seq_len,
        start_date="2019-09-15"
    )

    need_cols = [f"lag_{k}" for k in range(1, seq_len + 1)]

    df_feat = df_feat.dropna(subset=need_cols).copy()

    # ------------------------------------------------
    # Grid anchors
    # ------------------------------------------------

    anchors = [
        {"x": 24, "y": 151, "lat": 43.0691833, "lng": 141.3514707},
        {"x": 24, "y": 148, "lat": 43.0794037, "lng": 141.3422559},
        {"x": 26, "y": 153, "lat": 43.0579859, "lng": 141.3540211},
        {"x": 52, "y": 81, "lat": 43.1982317, "lng": 140.9940363},
        {"x": 50, "y": 41, "lat": 43.1880641, "lng": 140.7945541},
        {"x": 182, "y": 186, "lat": 43.8536095, "lng": 141.5234881},
    ]

    mapper = GridLatLngMapper(anchors)

    # ------------------------------------------------
    # Query parsing
    # ------------------------------------------------

    print("\n解析查詢...")

    client = OpenAI()

    engine_tmp = ConfidenceEngine(
        df_feat,
        mapper,
        openai_client=client,
        google_api_key=api_key
    )

    city_bounds = (42.9, 140.7, 43.9, 141.6)

    q = engine_tmp.parse_route_query_llm(user_input)

    print("LLM parsing result:", q)

    resolved = engine_tmp.resolve_two_locations(q, city_bounds)

    if resolved is None:
        raise RuntimeError("❌ 地點解析失敗")

    start_info, dest_info = resolved

    start_x, start_y = start_info["x"], start_info["y"]
    start_lat, start_lng = start_info["lat"], start_info["lng"]

    dest_x, dest_y = dest_info["x"], dest_info["y"]
    dest_lat, dest_lng = dest_info["lat"], dest_info["lng"]

    print(f"起點 grid: ({start_x}, {start_y})")
    print(f"終點 grid: ({dest_x}, {dest_y})")

    # ------------------------------------------------
    # Time slot
    # ------------------------------------------------

    target_t = engine_tmp.time_to_slot(q.hhmm)

    radius_cells = radius_m_to_cells(
        float(q.radius_m or 400),
        mapper,
        dest_x,
        dest_y
    )

    start_grids = get_neighbor_grids(start_x, start_y, radius=radius_cells)
    dest_grids = get_neighbor_grids(dest_x, dest_y, radius=radius_cells)

    target_grids = start_grids + dest_grids

    df_feat_small = filter_prediction_area_range(
        df_feat,
        t_start=target_t,
        t_end=min(target_t + T_BUFFER, 47),
        grids=target_grids
    )

    print(f"預測 time slot: {target_t}")
    print(f"radius_cells: {radius_cells}")
    print(f"prediction rows: {len(df_feat_small)}")

    if df_feat_small.empty:
        raise RuntimeError("該時間與範圍內沒有可用預測資料")

    # ------------------------------------------------
    # Crowd prediction
    # ------------------------------------------------

    print("\n產出人流預測中...")

    df_pred = make_df_pred_convlstm_embed(
        df_feat_small,
        model,
        cfg
    )

    print("✓ 人流預測完成")

    # ------------------------------------------------
    # Parking probability
    # ------------------------------------------------

    print("\n計算停車需求與停車機率...")

    south = 42.9
    north = 43.9
    west = 140.7
    east = 141.6

    df_parks, df_with_a, df_prob = create_poi_parking(
        df_pred,
        south,
        west,
        north,
        east,
        mapper,
        START_DATE,
        refetch=False
    )

    print("✓ 停車機率計算完成")

    # ------------------------------------------------
    # Routing
    # ------------------------------------------------

    print("\n進行停車推薦...")

    # 暫時用今天
    query_date = dt.datetime.today()

    topK = 3

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

    # 傳入 Google Geocoding 的精準座標
    fmap = routing_algorithm(
        mapper,
        df_prob,
        query_date,
        start_x,
        start_y,
        dest_x,
        dest_y,
        topK,
        target_t,
        api_key,
        prefs,
        origin_start_lat=start_lat,
        origin_start_lng=start_lng,
        origin_dest_lat=dest_lat,
        origin_dest_lng=dest_lng
    )

    fmap.save("parking_route.html")

    print("\n✓ 導航地圖已輸出: parking_route.html")


if __name__ == "__main__":
    main()