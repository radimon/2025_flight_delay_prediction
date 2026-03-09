import numpy as np
import pandas as pd
import requests
import hashlib
from src.geo.grid_to_latlng import GridLatLngMapper

# 以預測人流給予格網市區分數(越高越像市區)
def urban_score(df_pred:pd.DataFrame):
    # df_pred 欄位: d, t, x, y, score
    df = df_pred.copy()

    # 型別保險
    for c in ["d","t","x","y"]:
        df[c] = df[c].astype(int)
    df["score"] = df["score"].astype(float)

    # 聚合到每個格網
    g = (df.groupby(["x","y"], as_index=False)
        .agg(mean_score=("score","mean"), 
                max_score=("score","max"),
                std_score=("score","std"),
                n=("score","size")))

    # Min-Max normalization 
    mn = g["mean_score"].quantile(0.02)
    mx = g["mean_score"].quantile(0.98)
    eps = 1e-9
    g["urban_score"] = (g["mean_score"] - mn) / (mx - mn + eps)

    return g

# 用Overpass API 抓指定範圍內的停車場資料(傳入四組經緯度)
def fetch_parking_pois_overpass(south, west, north, east, timeout=180):
    query = f"""
        [out:json][timeout:{timeout}];
        (
        node["amenity"="parking"]({south},{west},{north},{east});
        way["amenity"="parking"]({south},{west},{north},{east});
        relation["amenity"="parking"]({south},{west},{north},{east});
        );
        out center tags;
    """

    url = "https://overpass-api.de/api/interpreter"
    r = requests.post(url, data={"data": query}, timeout=timeout)
    r.raise_for_status()
    data = r.json()

    rows = []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        # node(點狀停車場) 有 lat/lon；way/relation(線/面狀停車場) 用 center
        if el["type"] == "node":
            lat, lon = el.get("lat"), el.get("lon")
        else:
            c = el.get("center", {})
            lat, lon = c.get("lat"), c.get("lon")
        if lat is None or lon is None:
            continue

        rows.append({
            "poi_id": f'{el["type"]}/{el["id"]}',
            "lat": float(lat),
            "lng": float(lon),
            "name": tags.get("name"),
            "parking": tags.get("parking"),         # 種類如 surface / underground / multistory
            "access": tags.get("access"),           # private/public
            "capacity_osm": tags.get("capacity"),   
        })

    df_pois = pd.DataFrame(rows).drop_duplicates(subset=["poi_id"])
    return df_pois

# poi_id 產生穩定 seed（不受 Python hash 隨機化影響）
def stable_seed_from_poi_id(poi_id: str) -> int:
    h = hashlib.md5(poi_id.encode("utf-8")).hexdigest()
    return int(h[:8], 16) 

# 合成虛擬停車場資料
def synthesize_parking_lots_from_poi(
    g:pd.DataFrame,                         # x, y, n, urban_score
    df_poi:pd.DataFrame,                    # poi_id, lat, lng, type, access
    mapper:GridLatLngMapper,                # need latlng_to_grid
    seed=42,
    city_cap_range=(20, 40),                # 市區格數範圍
    suburb_cap_range=(60, 120),             # 郊區格數範圍
    city_threshold=0.3,                     # 超過當市區
    drop_outside_grid: bool = True,         # POI 若不在 g 的格網範圍，是否丟掉
    keep_access_yes: bool = False,          # 是否只保留 access==yes 的 POI
    apply_density_sampling: bool = False,   # 是否開啟抽樣
    keep_prob_min: float = 0.35,            # 郊區保留率
    keep_prob_max: float = 0.95,            # 市區保留率
    keep_gamma: float = 1.6,                # > 1 讓市區更容易被保留
):
    rng = np.random.default_rng(seed) #可重現停車場資料

    grid = g[["x","y","urban_score"]].copy()
    pois = df_poi.copy()
    u = grid["urban_score"].to_numpy()

    # 篩選 access
    if keep_access_yes and "access" in pois.columns:
        pois = pois[pois["access"].fillna("").str.lower().eq("yes")].copy()

    # 把 POI 經緯度轉換成格網座標偏移量
    park_xy = pois.apply(
        lambda r: mapper.latlng_to_grid(float(r["lat"]), float(r["lng"]), mode="float"),
        axis=1
    )
    pois["park_x"] = park_xy.map(lambda t: float(t[0]))
    pois["park_y"] = park_xy.map(lambda t: float(t[1]))
    pois["grid_x"] = pois["park_x"].round().astype(int)
    pois["grid_y"] = pois["park_y"].round().astype(int)

    # 把 POI 落在哪個格網的 urban_score merge 回來
    pois = pois.merge(grid,left_on=["grid_x", "grid_y"],right_on=["x", "y"],how="left",suffixes=("", "_g"))

    # 蛋雕格網外的
    if drop_outside_grid:
        pois = pois[pois["urban_score"].notna()].copy()
    else:
        # 不丟掉就把缺的當郊區
        pois["urban_score"] = pois["urban_score"].fillna(0.0)

    # 市區/郊區判斷
    pois["is_city"] = (pois["urban_score"] >= city_threshold).astype(int)

    # 密度抽樣：市區保留率高、郊區保留率低
    if apply_density_sampling:
        rng_global = np.random.default_rng(seed)
        u = pois["urban_score"].to_numpy(float)
        keep_prob = keep_prob_min + (keep_prob_max - keep_prob_min) * np.power(u, keep_gamma)
        keep_mask = rng_global.random(len(pois)) < keep_prob
        pois = pois[keep_mask].copy()

    # 隨機生成容量
    def sample_capacity_simple(row) -> int:
        is_city = int(row["is_city"]) == 1
        cap_lo, cap_hi = city_cap_range if is_city else suburb_cap_range
        return int(rng.integers(cap_lo, cap_hi + 1))

    # 缺值隨機補
    pois["capacity"] = pois["capacity_osm"].where(
        pois["capacity_osm"].notna(), 
        other=pois.apply(sample_capacity_simple, axis=1)
    )

    df_parks = pd.DataFrame({
        "park_id": pois["poi_id"].astype(str),
        "grid_x": pois["grid_x"].astype(int),
        "grid_y": pois["grid_y"].astype(int),
        "park_x": pois["park_x"].astype(float),
        "park_y": pois["park_y"].astype(float),
        "park_lat": pois["lat"].astype(float),
        "park_lng": pois["lng"].astype(float),
        "urban_score": pois["urban_score"].astype(float),
        "is_city": pois["is_city"].astype(int),
        "capacity": pois["capacity"].astype(int),
    })

    # 其他資訊可保留或蛋雕
    for col in ["type", "access"]:
        if col in pois.columns:
            df_parks[col] = pois[col].values

    return df_parks

# 用區域特性、當天內時段加成與局部事件加成為每個格網計算出參數 a(人流車流轉換比)
def build_a(df_pred, g, start_date:str,step_minutes=30,
            a_city=0.10, a_suburb=0.35, gamma=2.0,
            A_m=0.15, A_e=0.15, A_w=0.10,
            sig_m=1.0, sig_e=1.75, sig_w=4.0,
            beta=0.08, a_min=0.02, a_max=0.70):
    ''' 
        baseline f(x) =>
            a_city : 市區人流與車流轉換率 , a_suburb : 郊區人流與車流轉換率

        time factor f(x) =>
            A_m : 早上通勤尖峰加成 , A_e : 傍晚通勤尖峰加成 , A_w : 周末通勤尖峰加成
            sig_m : 早上通勤尖峰長度 , sig_e : 傍晚通勤尖峰長度 , sig_w : 周末通勤尖峰長度

        event boost f(x) =>
            z : 網格人流當下的 Z-Score , beta : 事件加成倍率
    '''

    df = df_pred.copy()
    df["score"] = df["score"].astype(float)

    # merge urban_score
    df = df.merge(g[["x","y","urban_score"]], on=["x","y"], how="left")
    df["urban_score"] = df["urban_score"].fillna(df["urban_score"].median())

    # ===== (A) a_base from urban_score =====
    sub = 1.0 - df["urban_score"].to_numpy() # sub 越接近 1 越像郊區
    a_base = a_city + (a_suburb - a_city) * np.power(sub, gamma)

    # ===== (B) time factor =====
    hour = df["t"].to_numpy() * (step_minutes / 60.0)

    # 週末判斷
    df["date"] = pd.to_datetime(start_date) + pd.to_timedelta(df["d"], unit="D")
    df["weekday"] = df["date"].dt.weekday
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    is_weekend = df["is_weekend"].to_numpy(dtype=float)

    # 高斯常態分佈(mu : 最高峰的時間)
    def gauss(h, mu, sig):
        return np.exp(-0.5 * ((h - mu) / sig) ** 2)

    # 1 是基底之倍率
    f_time = (
        1.0
        + A_m * gauss(hour, 8.0,  sig_m)
        + A_e * gauss(hour, 17.0, sig_e)
        + is_weekend * A_w * gauss(hour, 13.0, sig_w)
    )

    # ===== (C) event boost from z-score per grid =====
    grid_stats = (df.groupby(["x","y"])["score"]
                    .agg(["mean","std"])
                    .rename(columns={"mean":"mu","std":"sd"})
                    .reset_index())
    df = df.merge(grid_stats, on=["x","y"], how="left")
    z = (df["score"].to_numpy() - df["mu"].to_numpy()) / (df["sd"].to_numpy() + 1e-9) # z=0:平均值,z=1:比平常高一個std,z<0:比平常低不須加成 
    df['z'] = z    
    f_event = 1.0 + beta * np.maximum(0.0, z)

    # ===== final a =====
    # a_base : 區域特性 , f_time : 2; 當天內時段加成 , f_event : 局部事件加成
    a = a_base * f_time * f_event
    a = np.clip(a, a_min, a_max)

    df["a"] = a
    return df.drop(columns=["mu","sd"])

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# 把車流需求用機率模型算出每個停車場在每個時間點(d,t)的可停車機率
def build_parking_prob_baseline(
    df_with_a: pd.DataFrame,     # d,t,x,y,score,a
    df_parks: pd.DataFrame,      # park_id, park_x, park_y, capacity
    *,
    kappa: float = 1.0,          # car_demand = kappa * a * score (白話文：1人流單位有多少台車)
    R_choice: float = 3.0,       # R_choice內的停車場人流會較偏好去停 
    sigma: float = 1.5,          # 控制距離衰減速度
    sigma_choice: float = 1.5,   # 控制距離偏好衰減（越小越只選近的）
    theta0: float = 2.0,         # sigmoid 截距(控制整體偏高/偏低，越大sigmoid越接近1，越容易有位)
    theta1: float = 6.0,         # sigmoid 斜率(控制停車需求對滿位的敏感程度)
    cap_beta: float = 0.3,       # 容量加成（大停車場更容易被選，0=不考慮）
) -> pd.DataFrame:

    df = df_with_a.copy()
    parks = df_parks.copy()

    # grid-level car demand
    df["car_demand"] = kappa * df["a"].astype(float) * df["score"].astype(float)

    # 建立格網索引，把每個(x,y)映射成整數索引
    grid_xy = df[["x","y"]].drop_duplicates().copy()
    grid_xy["grid_idx"] = np.arange(len(grid_xy), dtype=int)

    # 把 grid_idx merge 回 df
    df = df.merge(grid_xy, on=["x","y"], how="left")

    # park_id -> 連續索引 park_pos（方便用 numpy 陣列累加）
    parks["park_id"] = parks["park_id"].astype(str)
    parks = parks.sort_values("park_id").reset_index(drop=True)
    park_ids = parks["park_id"].to_numpy()

    px_all = parks["park_x"].to_numpy(dtype=float)
    py_all = parks["park_y"].to_numpy(dtype=float)
    cap_all = parks["capacity"].to_numpy(dtype=float) # 每個停車場的容量，不隨時間變化
    n_parks = len(parks)

    # 對每一個格網先找出它附近會被考慮的停車場候選集合，然後用距離(+容量)算出這個格網的車流會選哪個停車場的機率分布 
    grid_records = [(np.array([], dtype=int), np.array([], dtype=float)) for _ in range(len(grid_xy))]
    for row in grid_xy.itertuples(index=False):
        gx0, gy0, gidx = float(row.x), float(row.y), int(row.grid_idx)

        dx = px_all - gx0
        dy = py_all - gy0
        dist2 = dx*dx + dy*dy
        mask = dist2 <= R_choice*R_choice

        park_poss = np.where(mask)[0]
        if len(park_poss) == 0:
            # 若R_choice內都沒停車場則取最近 K 個停車場當候選
            K = 5
            cand_poss = np.argsort(dist2)[:K].astype(int)
        else:
            cand_poss = park_poss.astype(int)

        dist2_select = dist2[cand_poss]

        # 用高斯衰減計算吸引力權重，越近越容易被想到
        w_choice = np.exp(-dist2_select / (2.0 * sigma_choice * sigma_choice))

        # 用高斯衰減計算競爭權重，越大代表那個格網的車流需求，對該停車場的影響越高。可理解為競爭強度
        w_comp = np.exp(-dist2_select / (2.0 * sigma * sigma))

        # w_choice * w_comp * capacity加成
        attr = w_choice * w_comp * np.power(cap_all[cand_poss], cap_beta)

        # 正規化成機率 P(p|g)
        probs = attr / (attr.sum() + 1e-12)
        grid_records[gidx] = (cand_poss, probs.astype(float))

    # 每個格網分配到各停車場的機率矩陣 W (n_grids × n_parks) 
    n_grids = len(grid_xy)
    W = np.zeros((n_grids, n_parks), dtype=float)

    for gidx in range(n_grids):
        park_poss, probs = grid_records[gidx]
        if len(park_poss) > 0:
            W[gidx, park_poss] = probs   # 每列加總為 1

    # 建立 (d,t) 的連續索引
    dt_keys = df[["d","t"]].drop_duplicates().sort_values(["d","t"]).reset_index(drop=True)
    dt_keys["dt_idx"] = np.arange(len(dt_keys), dtype=int)
    n_dt = len(dt_keys)

    df = df.merge(dt_keys, on=["d","t"], how="left")

    # 聚合(dt_idx, grid_idx)的 car_demand 加總
    agg = df.groupby(["dt_idx","grid_idx"], as_index=False)["car_demand"].sum()

    # 每個時間點每個格網的車流 F (n_dt × n_grids)
    F = np.zeros((n_dt, n_grids), dtype=float)
    F[agg["dt_idx"].to_numpy(int), agg["grid_idx"].to_numpy(int)] = agg["car_demand"].to_numpy(float)

    # shape: (n_dt, n_grids) @ (n_grids, n_parks) = (n_dt, n_parks)
    demand_mat = F @ W   # 每個時間點、每個停車場的車流需求

    # 向量化算 pressure / p_avail
    pressure_mat = demand_mat / (cap_all[np.newaxis, :] + 1e-9)  # 對 cap_all 升維，使其對齊 demand_mat
    p_avail_mat  = sigmoid(theta0 - theta1 * pressure_mat)       # shape=(n_dt, n_parks)

    dt_idx_arr   = np.repeat(np.arange(n_dt), n_parks)        # [0,0,...,1,1,...,n_dt]
    park_id_arr = np.tile(park_ids, n_dt) 

    result_df = pd.DataFrame({
        "dt_idx": dt_idx_arr,
        "park_id": park_id_arr,
        "demand": demand_mat.ravel(),
        "pressure": pressure_mat.ravel(),
        "p_avail": p_avail_mat.ravel(),
    })

    # merge 回 d,t
    result_df = result_df.merge(dt_keys[["dt_idx","d","t"]], on="dt_idx", how="left")
    out_df = result_df[["d","t","park_id","demand","pressure","p_avail"]].reset_index(drop=True)

    return pd.DataFrame(out_df)