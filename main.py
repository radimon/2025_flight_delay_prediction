import os
import requests
import polyline
import folium
import pandas as pd
import numpy as np
from dotenv import load_dotenv

from src.geo.grid_to_latlng import GridLatLngMapper, cell_size_m_at
from src.parking_engine import ParkingEngine
from src.routing_engine import RoutingEngine
from src.confidence_engine import ConfidenceEngine

from src.preprocess import *
from src.ConvLSTM import *


CONVLSTM_PATH = "models/convlstm_sapporo.pkl"
PARQUET_PATH = "data/processed/sapporo_density.parquet"
SAVED_FILLED_ZERO_DATA_PATH = "data/processed/sapporo_density_filled_zero.parquet"
PARKING_PARAM_PATH = "data/parking/grid_parking_params.parquet"

NEED_TRAIN = False
REBUILD_DATA = False


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


def filter_prediction_area(df, t_slot, grids):

    base = df[df["t"] == t_slot].copy()

    grid_df = pd.DataFrame(sorted(set(grids)), columns=["x", "y"])

    out = base.merge(grid_df, on=["x", "y"], how="inner")

    return out


def main():

    load_dotenv()

    api_key = os.getenv("GOOGLE_MAPS_API_KEY")

    if not api_key:
        raise ValueError("Google Maps API key not found")

    print("\n===== AI Parking Navigation =====\n")

    user_input = input("請輸入查詢（例如：小樽車站 小樽運河 週六 18:00 附近）：").strip()

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
        start_date="2023-01-01"
    )

    need_cols = [f"lag_{k}" for k in range(1, seq_len + 1)]

    df_feat = df_feat.dropna(subset=need_cols).copy()

    anchors = [
        {"x": 24, "y": 151, "lat": 43.0691833, "lng": 141.3514707},
        {"x": 24, "y": 148, "lat": 43.0794037, "lng": 141.3422559},
        {"x": 26, "y": 153, "lat": 43.0579859, "lng": 141.3540211},
        {"x": 52, "y": 81, "lat": 43.1982317, "lng": 140.9940363},
        {"x": 50, "y": 41, "lat": 43.1880641, "lng": 140.7945541},
        {"x": 182, "y": 186, "lat": 43.8536095, "lng": 141.5234881},
    ]

    mapper = GridLatLngMapper(anchors)

    # -----------------------------
    # Query parsing
    # -----------------------------

    print("\n解析查詢...")

    engine_tmp = ConfidenceEngine(df_feat, mapper)

    city_bounds = (42.9, 140.7, 43.9, 141.6)

    q = engine_tmp.parse_route_query(user_input)

    resolved = engine_tmp.resolve_two_locations(q, city_bounds)

    if resolved is None:
        raise RuntimeError("地點解析失敗")

    start_info, dest_info = resolved

    start_x, start_y = start_info["x"], start_info["y"]
    start_lat, start_lng = start_info["lat"], start_info["lng"]

    dest_x, dest_y = dest_info["x"], dest_info["y"]
    dest_lat, dest_lng = dest_info["lat"], dest_info["lng"]

    print(f"起點 grid: ({start_x}, {start_y})")
    print(f"終點 grid: ({dest_x}, {dest_y})")

    target_t = engine_tmp.time_to_slot(q.hhmm)

    radius_cells = radius_m_to_cells(float(q.radius_m or 400), mapper, dest_x, dest_y)

    start_grids = get_neighbor_grids(start_x, start_y, radius=radius_cells)
    dest_grids = get_neighbor_grids(dest_x, dest_y, radius=radius_cells)

    target_grids = start_grids + dest_grids

    df_feat_small = filter_prediction_area(
        df_feat,
        target_t,
        target_grids
    )

    print(f"預測 time slot: {target_t}")
    print(f"radius_cells: {radius_cells}")
    print(f"prediction rows: {len(df_feat_small)}")

    if df_feat_small.empty:
        raise RuntimeError("該時間與範圍內沒有可用預測資料")

    print("\n產出人流預測中...")

    df_pred = make_df_pred_convlstm_embed(
        df_feat_small,
        model,
        cfg
    )

    print("人流預測完成")

    # -----------------------------
    # 停車需求模型
    # -----------------------------

    print("\n計算停車需求與停車機率...")

    parking_engine = ParkingEngine(PARKING_PARAM_PATH)

    df_parking = parking_engine.run(df_pred)

    print("停車機率計算完成")

    # -----------------------------
    # 停車推薦
    # -----------------------------

    print("\n進行停車推薦...")

    routing_engine = RoutingEngine(mapper)

    best, candidates = routing_engine.recommend_parking(
        df_parking,
        start_lat,
        start_lng,
        dest_lat,
        dest_lng
    )

    print("\n推薦停車位置:")
    print(best)

    park_lat = float(best["lat"])
    park_lng = float(best["lng"])

    print("\n呼叫 Google Directions API...")

    coords1, dist1, dur1 = get_google_route(
        api_key,
        start_lat,
        start_lng,
        park_lat,
        park_lng
    )

    coords2, dist2, dur2 = get_google_route(
        api_key,
        park_lat,
        park_lng,
        dest_lat,
        dest_lng
    )

    fmap = folium.Map(
        location=[dest_lat, dest_lng],
        zoom_start=14
    )

    draw_route(fmap, coords1, "blue")
    draw_route(fmap, coords2, "green")

    folium.Marker(
        [start_lat, start_lng],
        popup=f"Start: {q.start_place}"
    ).add_to(fmap)

    folium.Marker(
        [park_lat, park_lng],
        popup="Parking"
    ).add_to(fmap)

    folium.Marker(
        [dest_lat, dest_lng],
        popup=f"Destination: {q.dest_place}"
    ).add_to(fmap)

    fmap.save("parking_route.html")

    print("\n✓ 地圖已輸出: parking_route.html")

    print(f"第一段距離: {dist1/1000:.2f} km, 時間: {dur1/60:.1f} 分")
    print(f"第二段距離: {dist2/1000:.2f} km, 時間: {dur2/60:.1f} 分\n")


if __name__ == "__main__":
    main()