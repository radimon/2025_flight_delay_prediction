import os
import pandas as pd
import numpy as np
import datetime as dt

from dotenv import load_dotenv
from openai import OpenAI

from src.geo.grid_to_latlng import GridLatLngMapper
from src.parks_create import *
from src.parks_choose import *
from src.parks_routing import *
from src.preprocess import *
from src.ConvLSTM import *
from src.confidence_engine import ConfidenceEngine

CONVLSTM_PATH = "models/convlstm_sapporo.pkl"
PARQUET_PATH = "data/processed/sapporo_density.parquet"
START_DATE = "2019-09-15"
T_BUFFER = 4

anchors = [
    {"x": 24, "y": 151, "lat": 43.0691833, "lng": 141.3514707},
    {"x": 24, "y": 148, "lat": 43.0794037, "lng": 141.3422559},
    {"x": 26, "y": 153, "lat": 43.0579859, "lng": 141.3540211},
    {"x": 52, "y": 81,  "lat": 43.1982317, "lng": 140.9940363},
    {"x": 50, "y": 41,  "lat": 43.1880641, "lng": 140.7945541},
    {"x": 182,"y": 186, "lat": 43.8536095, "lng": 141.5234881},
]

SCENARIOS = [
    {"name": "Weekday Morning",   "weekday": 2, "hhmm": "10:00"},
    {"name": "Weekday Evening",   "weekday": 2, "hhmm": "19:00"},
    {"name": "Weekend Afternoon", "weekday": 6, "hhmm": "14:00"},
]

TOPK = 3
PREFS = {
    "risk_aversion": 0.4,
    "min_prob": 0.3,
    "max_walk_min": 30,
    "max_detour_min": 15,
    "w_prob": 0.45,
    "w_walk": 0.25,
    "w_detour": 0.15,
    "w_drive": 0.10,
    "w_price": 0.05,
}

CITY_BOUNDS = (42.9, 140.7, 43.9, 141.6)

def hhmm_to_slot(hhmm: str, slot_min=30) -> int:
    h, m = map(int, hhmm.split(":"))
    return (h * 60 + m) // slot_min

def run_scenario(df_feat, model, cfg, mapper, engine, start_info, dest_info, scenario):
    name    = scenario["name"]
    weekday = scenario["weekday"]
    t0      = hhmm_to_slot(scenario["hhmm"])

    start_x, start_y = start_info["x"], start_info["y"]
    dest_x,  dest_y  = dest_info["x"],  dest_info["y"]

    # 取目標格網周邊資料
    dest_grids  = [(dest_x  + dx, dest_y  + dy) for dx in range(-4,5) for dy in range(-4,5)]
    start_grids = [(start_x + dx, start_y + dy) for dx in range(-4,5) for dy in range(-4,5)]
    target_grids = list(set(start_grids + dest_grids))

    grid_df  = pd.DataFrame(sorted(target_grids), columns=["x","y"])
    df_small = df_feat[(df_feat["t"] >= t0) & (df_feat["t"] <= min(t0+T_BUFFER, 47))].copy()
    df_small = df_small.merge(grid_df, on=["x","y"], how="inner")

    if df_small.empty:
        print(f"[{name}] 無資料，跳過")
        return None

    # 人流預測
    df_pred = make_df_pred_convlstm_embed(df_small, model, cfg)

    # 停車機率
    df_parks, df_with_a, df_prob = create_poi_parking(
        df_pred, *CITY_BOUNDS,
        mapper, START_DATE, refetch=False
    )

    # 聚合同weekday
    df_prob_w = aggregate_df_prob_same_weekday(df_prob, agg="median")
    df_prob_w = df_prob_w[df_prob_w["w"] == weekday]

    # 建候選
    cand = build_candidates(
        df_prob_w, mapper=mapper,
        w0=weekday, t0=t0,
        O=(start_x, start_y),
        D=(dest_x,  dest_y)
    )

    # 選Top-K
    result = choose_parking(cand, prefs=PREFS, top_k=TOPK)

    if result.empty:
        print(f"[{name}] 無符合條件的停車場")
        return None

    result = result.reset_index(drop=True)
    result["rank"]    = result.index + 1
    result["scenario"] = name
    result["time"]    = scenario["hhmm"]

    return result[["scenario", "time", "rank", "park_id",
                   "lat", "lng", "p_avail",
                   "drive_time", "walk_time", "detour_time", "score"]]

def main():
    load_dotenv()
    api_key    = os.getenv("GOOGLE_MAPS_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    print("載入模型與資料...")
    df_true  = pd.read_parquet(PARQUET_PATH).copy()
    model, cfg = load_convlstm_embed(CONVLSTM_PATH)
    seq_len  = int(cfg["seq_len"])

    df_feat  = add_lags_and_rollings(df_true.copy(), seq_len=seq_len, start_date=START_DATE)
    need_cols = [f"lag_{k}" for k in range(1, seq_len+1)]
    df_feat  = df_feat.dropna(subset=need_cols).copy()

    mapper = GridLatLngMapper(anchors)
    client = OpenAI(api_key=openai_key)

    engine = ConfidenceEngine(df_feat, mapper, openai_client=client, google_api_key=api_key)

    # 解析起訖點（只做一次）
    print("\n解析地點：札幌車站 → 丘珠空港")
    start_info = engine.resolve_place_to_grid("札幌駅", CITY_BOUNDS)
    dest_info  = engine.resolve_place_to_grid("北海道大学", CITY_BOUNDS)

    if start_info is None or dest_info is None:
        raise RuntimeError("地點解析失敗")

    print(f"起點 grid: ({start_info['x']}, {start_info['y']})")
    print(f"終點 grid: ({dest_info['x']}, {dest_info['y']})")

    # 跑三個情境
    all_results = []
    for scenario in SCENARIOS:
        print(f"\n執行情境：{scenario['name']}...")
        res = run_scenario(df_feat, model, cfg, mapper, engine, start_info, dest_info, scenario)
        if res is not None:
            all_results.append(res)

    if all_results:
        df_final = pd.concat(all_results, ignore_index=True)
        df_final.to_csv("scenario_comparison.csv", index=False)
        print("\n✓ 結果已存到 scenario_comparison.csv")
        print(df_final.to_string(index=False))
    else:
        print("所有情境均無結果")

if __name__ == "__main__":
    main()