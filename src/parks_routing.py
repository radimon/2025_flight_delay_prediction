from src.parks_create import *
from src.parks_choose import *
from src.geo.grid_to_latlng import GridLatLngMapper
import numpy as np
import pandas as pd
import requests
import polyline
import folium
import datetime as dt

# 呼叫 google map API找出最短路徑
def get_google_route(api_key, origin_lat, origin_lng, dest_lat, dest_lng,
                     mode="driving", departure_time=None, traffic_model="best_guess"):

    url = "https://maps.googleapis.com/maps/api/directions/json"

    params = {
        "origin": f"{origin_lat},{origin_lng}",
        "destination": f"{dest_lat},{dest_lng}",
        "mode": mode,              # driving / walking
        "key": api_key,
        "alternatives": "false",
    }

    # 只有 driving 才能帶 departure_time / traffic_model
    if mode == "driving" and departure_time is not None:
        if isinstance(departure_time, dt.datetime):
            params["departure_time"] = int(departure_time.timestamp())
        else:
            # 你也可以直接傳 int unix timestamp
            params["departure_time"] = int(departure_time)
        params["traffic_model"] = traffic_model

    r = requests.get(url, params=params, timeout=15)
    data = r.json()

    if data.get("status") != "OK":
        raise RuntimeError(f"Google API error: {data.get('status')} - {data.get('error_message')}")

    route = data["routes"][0]
    poly = route["overview_polyline"]["points"]
    coords = polyline.decode(poly)  # [(lat,lng), ...]

    leg = route["legs"][0]
    dist = leg["distance"]["value"]     # meters
    dur  = leg["duration"]["value"]     # seconds

    return coords, dist, dur

def draw_route(layer, coords, color, tooltip=None, dash_array=None, weight=5):

    folium.PolyLine(
        coords,
        color=color,
        weight=weight,
        opacity=0.85,
        dash_array=dash_array,   # 例如 "6,10" 畫虛線
        tooltip=tooltip
    ).add_to(layer)

# 用抓取 POI 並隨機模擬容量的方式生成一份矩形區塊內的停車場資料
def create_poi_parking(
    df_pred:pd.DataFrame,    # 預測出的人流
    south,                   # 最小緯度 
    west,                    # 最小經度
    north,                   # 最大緯度
    east,                    # 最大經度
    mapper:GridLatLngMapper,
    start_date:str,
    city_q:int = 0.7,        # 市區與郊區區分的百分位
    seed:int = 2025,         # 合成模擬停車場的 random seed
):
    g = urban_score(df_pred)

    df_poi = fetch_parking_pois_overpass(south, west, north, east)

    # 保留公有停車場
    df_poi_public = df_poi[df_poi["access"].isin([None, "yes", "customers", "permissive"])]

    # NaN > 80% 蛋雕
    df_poi_public = df_poi_public.drop(columns=["name"])

    # 補上NaN
    df_poi_public["parking"] = df_poi_public["parking"].fillna("surface")
    df_poi_public["access"] = df_poi_public["access"].fillna("yes")

    df_poi_public = df_poi_public.rename(columns={"parking" : "type"})

    # 用百分位切
    city_q = city_q
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
    df_prob = build_parking_prob_baseline(df_with_a, df_parks)

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
    df["w"] = (df["d"].astype(int) % 7)  # Sun=0..Sat=6

    # 動態欄位做聚合
    val_cols = [c for c in ["p_avail","demand","pressure"] if c in df.columns]

    # "median" 或 "mean" 字串，不要傳 np.median
    df_agg = (df.groupby(["w", "t", "park_id"], as_index=False)[val_cols].agg(agg)) 

    # 靜態欄位保留(只需 drop_duplicates，不需要 groupby)
    static_cols = [c for c in df.columns if c not in (["d", "w"] + val_cols)]

    df_static = (df[static_cols].drop_duplicates(subset=["t", "park_id"]))

    # merge 回來
    df_w = df_agg.merge(df_static, on=["t", "park_id"], how="left")

    return df_w

# 把最佳路徑畫在地圖
def routing_algorithm(
    mapper:GridLatLngMapper,
    df_prob:pd.DataFrame,
    query_date: dt.date,    # # 日期，將其映射為訓練資料的某一天
    start_x, 
    start_y, 
    dest_x, 
    dest_y, 
    topK, 
    t, 
    api_key,
    prefs:dict,
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
        print(f"沒有符合條件的候選停車場，請放寬 prefs 條件")
        return None

    start_lat, start_lng = mapper.grid_to_latlng(start_x, start_y)
    dest_lat, dest_lng = mapper.grid_to_latlng(dest_x, dest_y)

    # 抵達時間(date + hhmm)
    departure_time = dt.datetime.combine(query_date, slot_to_time(t, SLOT_MIN))
    # 如果時間已過，推到下週同一天
    if departure_time <= dt.datetime.now():
        departure_time += dt.timedelta(days=7)
        print(f"查詢時間已過，改用下週同星期：{departure_time}")

    # 建立 Folium 地圖物件，用終點當中心
    fmap = folium.Map(location=[dest_lat, dest_lng], zoom_start=14)

    # 起點與終點 marker
    folium.Marker([start_lat, start_lng], popup=f"Start").add_to(fmap)
    folium.Marker([dest_lat,  dest_lng],  popup=f"Destination").add_to(fmap)

    colors = ["red", "blue", "purple", "orange", "green"]

    all_points = [(start_lat, start_lng), (dest_lat, dest_lng)]  # for fit_bounds

    for i, row in parks.sort_values("score", ascending=False).reset_index(drop=True).iterrows():
        pid  = row["park_id"]
        plat = float(row["lat"])
        plng = float(row["lng"])
        color = colors[i % len(colors)]

        # 可以把不同候選停車場的路線分成不同 group，搭配 LayerControl 讓使用者勾選顯示/隱藏
        fg = folium.FeatureGroup(name=f"#{i+1} park_id={pid} score={row['score']:.3f}")

        coords_drive, dist1, dur1 = get_google_route(
            api_key, start_lat, start_lng, plat, plng,
            mode="driving", departure_time=departure_time
        )

        coords_walk, dist2, dur2 = get_google_route(
            api_key, plat, plng, dest_lat, dest_lng,
            mode="walking"
        )

        draw_route(fg, coords_drive, color,
                tooltip=f"[DRIVE] #{i+1} park={pid} / {dur1/60:.1f}min / {dist1/1000:.2f}km",
                dash_array=None, weight=5)

        draw_route(fg, coords_walk, color,
                tooltip=f"[WALK] #{i+1} park={pid} / {dur2/60:.1f}min / {dist2/1000:.2f}km",
                dash_array="6,10", weight=4)

        folium.Marker(
            [plat, plng],
            icon=folium.Icon(color=color, icon="car", prefix="fa"),
            popup=(f"#{i+1} park_id = {pid}<br>"
                f"score = {row['score']:.3f}<br>"
                f"p_avail = {row['p_avail']:.3f}<br>"
                f"drive ≈ {row['drive_time']:.2f} / walk ≈ {row['walk_time']:.2f}")
        ).add_to(fg)

        fg.add_to(fmap)
        all_points.extend([(plat, plng)] + coords_drive + coords_walk)

    # 自動縮放到全部路線都看得到
    min_lat = min(p[0] for p in all_points); max_lat = max(p[0] for p in all_points)
    min_lng = min(p[1] for p in all_points); max_lng = max(p[1] for p in all_points)
    fmap.fit_bounds([[min_lat, min_lng], [max_lat, max_lng]])

    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.save("parking_topk_routes.html")

    return fmap