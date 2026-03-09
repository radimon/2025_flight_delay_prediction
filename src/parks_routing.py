from src.parks_create import *
from src.parks_choose import *
from src.geo.grid_to_latlng import GridLatLngMapper
import numpy as np
import pandas as pd
import requests
import polyline
import folium
import datetime as dt

# 設定時間槽長度，確保與 ConfidenceEngine 一致
SLOT_MIN = 30

# 呼叫 google map api找出最短路徑
def get_google_route(api_key, origin_lat, origin_lng, dest_lat, dest_lng,
                     mode="driving", departure_time=None, traffic_model="best_guess"):
    url = "https://maps.googleapis.com/maps/api/directions/json"
    params = {
        "origin": f"{origin_lat},{origin_lng}",
        "destination": f"{dest_lat},{dest_lng}",
        "mode": mode,            # driving / walking
        "key": api_key,
        "alternatives": "false",
        "language": "zh-TW"
    }

    # 只有 driving 才能帶 departure_time / traffic_model
    if mode == "driving" and departure_time is not None:
        if isinstance(departure_time, dt.datetime):
            params["departure_time"] = int(departure_time.timestamp())
        else:
            # 你也可以直接傳 int unix timestamp
            params["departure_time"] = int(departure_time)
        params["traffic_model"] = traffic_model
    
    try:
        r = requests.get(url, params=params, timeout=15)
        data = r.json()
        if data.get("status") != "OK":
            return [(origin_lat, origin_lng), (dest_lat, dest_lng)], 0, 0
        route = data["routes"][0]
        poly = route["overview_polyline"]["points"]
        coords = polyline.decode(poly)
        leg = route["legs"][0]
        dist = leg["distance"]["value"] # meters
        dur  = leg["duration"]["value"] # seconds
        return coords, dist, dur
    except Exception as e:
        return [(origin_lat, origin_lng), (dest_lat, dest_lng)], 0, 0

def draw_route(layer, coords, color, tooltip=None, dash_array=None, weight=5):
    folium.PolyLine(
        coords,
        color=color,
        weight=weight,
        opacity=0.75, 
        dash_array=dash_array,
        tooltip=tooltip
    ).add_to(layer)

# ============================================================
# Parking Data Processing
# ============================================================

# 用抓取 POI 並隨機模擬容量的方式生成一份矩形區塊內的停車場資料與機率
def create_poi_parking(
    df_pred: pd.DataFrame,      # 預測出的人流
    south,                      # 最小緯度 
    west,                       # 最小經度
    north,                      # 最大緯度
    east,                       # 最大經度
    mapper: GridLatLngMapper,
    start_date: str,
    city_q: float = 0.7,        # 市區與郊區區分的百分位
    seed: int = 2025,           # 合成模擬停車場的 random seed
):
    g = urban_score(df_pred) 
    df_poi = fetch_parking_pois_overpass(south, west, north, east)

    # 保留公有停車場
    df_poi_public = df_poi[df_poi["access"].isin([None, "yes", "customers", "permissive"])]

    # NaN > 80% 蛋雕
    df_poi_public = df_poi[df_poi["access"].isin([None, "yes", "customers", "permissive"])].copy()    
    if "name" in df_poi_public.columns:
        df_poi_public = df_poi_public.drop(columns=["name"])

    # 補上NaN
    df_poi_public["parking"] = df_poi_public["parking"].fillna("surface")
    df_poi_public["access"] = df_poi_public["access"].fillna("yes")
    df_poi_public = df_poi_public.rename(columns={"parking": "type"})

    # 用百分位切
    city_threshold = g["urban_score"].quantile(city_q)

    df_parks = synthesize_parking_lots_from_poi(
        g, 
        df_poi=df_poi_public, 
        mapper=mapper,
        seed=seed, 
        city_threshold=city_threshold
    )

    # 生成矩形區塊內所有格網的人流與車流轉換比
    df_with_a = build_a(df_pred, g, start_date=start_date)

    # 用人流與車流轉換比與模擬停車場的資料透過機率模型算出每個停車場在每個時間點(d,t)的可停車機率
    df_prob = build_parking_prob_baseline(df_with_a, df_parks, city_threshold=city_threshold)

    # 把停車場位置/容量/市區標籤合併回來
    df_prob = df_prob.merge(
        df_parks[["park_id", "park_x", "park_y", "capacity"] + 
                 [c for c in ["is_city", "urban_score"] if c in df_parks.columns]],
        on="park_id",
        how="left"
    )

    return df_parks, df_with_a, df_prob

# t => hhmm
def slot_to_time(t_idx, slot_min=30):
    m = t_idx * slot_min
    return dt.time(m//60, m%60)

# 依查詢日期的 weekday 把所有同樣 d 抽出來，再對每個 parks 做 mean/median 聚合，在不越界的前提下回答任意年份日期的查詢
def aggregate_df_prob_same_weekday(df_prob: pd.DataFrame, agg="median") -> pd.DataFrame:
    df = df_prob.copy()
    df["w"] = (df["d"].astype(int) % 7)   # Sun=0..Sat=6

    # 動態欄位做聚合
    val_cols = [c for c in ["p_avail","demand","pressure"] if c in df.columns]
    
    # "median" 或 "mean" 字串，不要傳 np.median
    df_agg = (df.groupby(["w", "t", "park_id"], as_index=False)[val_cols].agg(agg)) 

    # 靜態欄位保留(只需 drop_duplicates，不需要 groupby)
    static_cols = [c for c in df.columns if c not in (["d", "w"] + val_cols)]
    df_static = (df[static_cols].drop_duplicates(subset=["t", "park_id"]))
    
    # merge 回來
    return df_agg.merge(df_static, on=["t", "park_id"], how="left")

# ============================================================
# Routing Algorithm
# ============================================================

# 把最佳路徑畫在地圖
def routing_algorithm(
    mapper: GridLatLngMapper,
    df_prob: pd.DataFrame,
    query_date: dt.date,       # 日期，將其映射為訓練資料的某一天
    start_x, start_y, 
    dest_x, dest_y, 
    topK, t, 
    api_key,
    prefs: dict,
    origin_start_lat=None,     # 原始定位的起點與終點的確切經緯度
    origin_start_lng=None,
    origin_dest_lat=None,
    origin_dest_lng=None
):
    # 將訓練資料中同 weekday 的停車場資訊用中位數聚合(考慮預先計算再傳入)
    df_prob_w = aggregate_df_prob_same_weekday(df_prob, agg="median")

    # 把日期轉為 weekday
    w0 = (query_date.weekday() + 1) % 7

    # 給予起點、終點與 weekday，輸出候選停車場
    cand = build_candidates(df_prob_w, mapper=mapper, w0=w0, t0=t, O=(start_x,start_y), D=(dest_x,dest_y))

    # 依據偏好選擇最適合的幾個停車場
    parks = choose_parking(cand, prefs=prefs, top_k=topK)

    if parks.empty or "score" not in parks.columns:
        return None

    grid_s_lat, grid_s_lng = mapper.grid_to_latlng(start_x, start_y)
    grid_d_lat, grid_d_lng = mapper.grid_to_latlng(dest_x, dest_y)

    s_lat = origin_start_lat if origin_start_lat is not None else grid_s_lat
    s_lng = origin_start_lng if origin_start_lng is not None else grid_s_lng
    d_lat = origin_dest_lat if origin_dest_lat is not None else grid_d_lat
    d_lng = origin_dest_lng if origin_dest_lng is not None else grid_d_lng

    # 建立 Folium 地圖物件，用終點當中心
    fmap = folium.Map(location=[d_lat, d_lng], zoom_start=14)

    # 起點與終點 marker
    folium.Marker([s_lat, s_lng], popup="精確起點", icon=folium.Icon(color='green', icon='play')).add_to(fmap)
    folium.Marker([d_lat, d_lng], popup="精確終點", icon=folium.Icon(color='red', icon='flag')).add_to(fmap)

    # 抵達時間(date + hhmm)
    departure_time = dt.datetime.combine(query_date, slot_to_time(t, SLOT_MIN))
    # 如果時間已過，推到下週同一天(google map 只能查未來的時間)
    if departure_time <= dt.datetime.now():
        departure_time += dt.timedelta(days=7)
        print(f"查詢時間已過，改用下週同星期：{departure_time}")

    colors = ["red", "blue", "purple", "orange", "darkgreen"]
    all_points = [(s_lat, s_lng), (d_lat, d_lng)]

    # 顯示全部候選
    for i, row in parks.sort_values("score", ascending=False).reset_index(drop=True).iterrows():
        pid = row["park_id"]
        plat, plng = float(row["lat"]), float(row["lng"])
        color = colors[i % len(colors)]

        # 可以把不同候選停車場的路線分成不同 group，搭配 LayerControl 讓使用者勾選顯示/隱藏
        fg = folium.FeatureGroup(name=f"推薦 #{i+1} (Score: {row['score']:.2f})")

        # departure_time 如果不需要即時路況也可以不傳
        coords_drive, dist1, dur1 = get_google_route(
            api_key, s_lat, s_lng, plat, plng,
            mode="driving", departure_time=departure_time
        )

        coords_walk, dist2, dur2 = get_google_route(
            api_key, plat, plng, d_lat, d_lng,
            mode="walking"
        )

        draw_route(fg, coords_drive, color, tooltip=f"開車: {dur1/60:.1f}分", weight=5)
        draw_route(fg, coords_walk, color, tooltip=f"走路: {dur2/60:.1f}分", dash_array="6,10", weight=4)

        folium.Marker(
            [plat, plng],
            icon=folium.Icon(color=color, icon="car", prefix="fa"),
            popup=(f"<b>候選 #{i+1}</b><br> ID: {pid}<br>"
                   f"可停機率: {row['p_avail']:.2%}<br>"
                   f"步行距離: {dist2}m")
        ).add_to(fg)

        fg.add_to(fmap)
        all_points.extend([(plat, plng)] + coords_drive + coords_walk)

    # 自動縮放到全部路線都看得到
    lats, lngs = zip(*all_points)
    fmap.fit_bounds([[min(lats), min(lngs)], [max(lats), max(lngs)]])
    folium.LayerControl(collapsed=False).add_to(fmap)

    return fmap