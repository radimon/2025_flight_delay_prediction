import numpy as np
import pandas as pd
from src.geo.grid_to_latlng import *

SLOT_MIN = 30
SLOTS_PER_DAY = 48

# 計算通勤時間(min)
def time_minutes_from_meters(dist_m, speed_kmh):
    # dist_m: meters, speed_kmh: km/h
    return (dist_m * 60) / (speed_kmh * 1000.0)

# 輸入起點與終點，先初步算出當天所有可停車的停車場候選
def build_candidates(
    df_prob, 
    mapper:GridLatLngMapper, 
    w0, # 出發的 weekday
    t0, 
    O, 
    D,
    v_car_kmh=35, 
    v_walk_kmh=4.8, 
    top_parks_limit=None
):
    ''' 
        O : 起點, D : 終點, d0 : 天數
    '''

    # 先取當天資料
    day = df_prob[df_prob["w"] == w0].copy()

    # 先拿停車場靜態位置（每個 park_id 一筆）
    parks = (day.sort_values(["t"])
               .drop_duplicates("park_id")[["park_id","park_x","park_y"]])
    
    # 限制候選停車場數量
    if top_parks_limit is not None and len(parks) > top_parks_limit:
        parks = parks.head(top_parks_limit).copy()

    # 把 O/D/parks 座標轉成經緯度
    od = pd.DataFrame([{"x": O[0], "y": O[1]}, {"x": D[0], "y": D[1]}])
    od_lat_lng = mapper.transform(od)
    O_lat, O_lng = float(od_lat_lng.loc[0, "lat"]), float(od_lat_lng.loc[0, "lng"])
    D_lat, D_lng = float(od_lat_lng.loc[1, "lat"]), float(od_lat_lng.loc[1, "lng"])

    parks_xy = parks.rename(columns={"park_x": "x", "park_y": "y"})[["x", "y"]]
    parks_lat_lng = mapper.transform(parks_xy)
    parks = parks.copy()
    parks["lat"] = parks_lat_lng["lat"].to_numpy()
    parks["lng"] = parks_lat_lng["lng"].to_numpy()

    # 算起點到停車場的開車時間(min)與距離(m)
    dist_op_m = haversine_m_vec(O_lat, O_lng, parks["lat"].to_numpy(), parks["lng"].to_numpy())
    parks["drive_time"] = time_minutes_from_meters(dist_op_m, v_car_kmh)

    # 抵達 time slot
    slot_offset = np.ceil(parks["drive_time"] / SLOT_MIN).astype(int)
    t_arrive_raw = t0 + slot_offset

    # 抵達 weekday（跨日就 + days 再 %7）
    day_offset = (t_arrive_raw // SLOTS_PER_DAY).astype(int)
    parks["w_arrive"] = ((w0 + day_offset) % 7).astype(int)
    parks["t_arrive"] = (t_arrive_raw % SLOTS_PER_DAY).astype(int)

    # 某個 park_id，在 (w_arrive, t_arrive) 時，抓 df_prob 的 p_avail
    need = parks[["park_id","w_arrive","t_arrive"]].rename(columns={"w_arrive":"w","t_arrive":"t"})
    prob_slice = df_prob[df_prob["park_id"].isin(parks["park_id"])][["w","t","park_id","p_avail"]]

    cand = need.merge(prob_slice, on=["w","t","park_id"], how="left").merge(
        parks[["park_id","park_x","park_y","lat","lng","drive_time"]], on="park_id", how="left"
    )

    # 算停車場到終點的走路時間
    dist_pd_m = haversine_m_vec(cand["lat"].to_numpy(), cand["lng"].to_numpy(), D_lat, D_lng)
    cand["walk_time"] = time_minutes_from_meters(dist_pd_m, v_walk_kmh)

    # 起點到終點的直達開車時間
    direct_od_m = float(haversine_m_vec(O_lat, O_lng, D_lat, D_lng))
    direct_drive = time_minutes_from_meters(direct_od_m, v_car_kmh)

    # 繞路（開車+走路 vs 直達開車）
    cand["detour_time"] = (cand["drive_time"] + cand["walk_time"]) - direct_drive

    return cand

# normalize
def minmax_norm(s: pd.Series) -> pd.Series:
    mn, mx = s.min(), s.max()
    return (s - mn) / (mx - mn + 1e-9)

def reliability(p: pd.Series, risk_aversion: float) -> pd.Series:
    # risk_aversion:風險偏好(越接近 1 越保守)
    eps = 1e-9
    # 線性插值
    neutral = p.clip(0, 1)
    cautious = -np.log(1 - p.clip(0, 1) + eps) # Convex function:越接近 1 的區域分數提升得越快，達到機率越低越被厭惡的效果
    return (1 - risk_aversion) * neutral + risk_aversion * cautious

# 從所有候選停車場中，依據偏好參數計算每個停車場的 score，篩選最適合的topK停車場
def choose_parking(
    candidates: pd.DataFrame,  # 每列一個停車場，至少要有 park_id, drive_time, walk_time, detour_time, p_avail
    prefs: dict,
    top_k: int = 5,
) -> pd.DataFrame:
    df = candidates.copy()

    # Hard filters
    min_prob = prefs.get("min_prob", 0.3) # 可停車機率最低門檻
    max_walk = prefs.get("max_walk_min", 30) # 最多走幾分鐘
    max_detour = prefs.get("max_detour_min", 15) # 最多繞幾分鐘

    df = df[df["p_avail"] >= min_prob]
    df = df[df["walk_time"] <= max_walk]
    df = df[df["detour_time"] <= max_detour]

    if len(df) == 0:
        return df  # 沒候選就空表回傳，讓上層做 fallback（放寬條件或直接改成最靠近者）

    # Normalize costs 
    df["C_walk"] = minmax_norm(df["walk_time"])
    df["C_detour"] = minmax_norm(df["detour_time"])
    df["C_drive"] = minmax_norm(df["drive_time"])

    if "price" in df.columns:
        df["C_price"] = minmax_norm(df["price"])
    else:
        df["C_price"] = 0.0

    # 謹慎性
    risk = float(prefs.get("risk_aversion", 0.4))
    df["R"] = reliability(df["p_avail"], risk)

    # 權重 
    w_prob  = prefs.get("w_prob", 0.45)   # 重視有位子的機率
    w_walk  = prefs.get("w_walk", 0.25)   # 重視不要走太久
    w_det   = prefs.get("w_detour", 0.15) # 重視不要繞太遠
    w_drive = prefs.get("w_drive", 0.10)  # 重視不要開車太久
    w_price = prefs.get("w_price", 0.05)  # 重視停車費便宜

    # + 加分項 , - 扣分項
    df["score"] = (
        w_prob  * df["R"]
        - w_walk  * df["C_walk"]
        - w_det   * df["C_detour"]
        - w_drive * df["C_drive"]
        - w_price * df["C_price"]
    )

    # 排序 + 回傳 Top-K
    cols = ["park_id", "lat", "lng", "p_avail",
            "drive_time", "walk_time", "detour_time",
            "score", "R", "C_walk", "C_detour", "C_drive"]
    if "price" in df.columns:
        cols.insert(5, "price")
    return df.sort_values("score", ascending=False).head(top_k)[cols]