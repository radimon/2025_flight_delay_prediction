import re
import json
import datetime
import requests
import numpy as np
import pandas as pd
from typing import List
from dataclasses import dataclass, fields
from scipy.spatial import KDTree
from src.str_similarity import similarity, importance, best_label
from src.util import normalize_nonneg, circle_radius_by_mass
from src.geo.grid_to_latlng import cell_size_m_at, haversine_m

@dataclass
class CrowdQuery:
    city: str
    date: str | None 
    weekday: int | None      
    hhmm: str | None         
    radius_m: float | None
    place: str | None        
    place_variants: List[str] | None
    lat: float | None = None
    lon: float | None = None

class ConfidenceEngine:
    def __init__(self, df_pred, mapper, openai_client):
        self.df_pred = df_pred
        self.mapper = mapper
        self.client = openai_client
        self.tree, self.grid_xy_lookup = self._build_grid_kdtree()

    def _build_grid_kdtree(self):
        """建立 KDTree 以快速檢索最近網格"""
        grid_xy = self.df_pred[["x", "y"]].drop_duplicates().reset_index(drop=True)
        grid_ll = self.mapper.transform(grid_xy.copy()).reset_index(drop=True)
        tree = KDTree(np.c_[grid_ll["lat"].to_numpy(), grid_ll["lng"].to_numpy()])
        return tree, grid_xy

    def time_to_slot(self, hhmm: str) -> int:
        """將 HH:MM 轉為 30 分鐘一個單位的 slot (0-47)"""
        try:
            h, m = map(int, hhmm.split(':'))
            return (h * 60 + m) // 30
        except:
            return 24  # 預設中午

    def parse_query_llm(self, text: str, default_city="sapporo") -> CrowdQuery:
        """使用 LLM 解析使用者輸入，並嚴格過濾欄位"""
        TODAY = datetime.date.today()
        
        prompt = f"""你是查詢解析器。今天日期是 {TODAY}。把文字轉成 JSON。
        必須使用以下英文鍵名：city, date, weekday, hhmm, radius_m, place, place_variants。
        weekday 請給整數 (0=週一, 5=週六, 6=週日)。
        使用者輸入：{text}"""

        resp = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a data extractor. Only output JSON with English keys."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        raw_data = json.loads(resp.choices[0].message.content)
        
        # 1. 處理中文 Key 的對應
        mapping = {"地點": "place", "城市": "city", "時間": "hhmm", "日期": "date", "星期": "weekday", "別名": "place_variants"}
        processed_data = {}
        for k, v in raw_data.items():
            new_key = mapping.get(k, k)
            processed_data[new_key] = v

        # 2. 【核心修正】只留下 CrowdQuery 定義過的欄位，過濾掉「地點」等不認識的參數
        allowed_names = {f.name for f in fields(CrowdQuery)}
        final_init_data = {k: v for k, v in processed_data.items() if k in allowed_names}

        # 3. 補足必填缺失值
        if "city" not in final_init_data: final_init_data["city"] = default_city
        if "place_variants" not in final_init_data: final_init_data["place_variants"] = [final_init_data.get("place", "")]
        if "hhmm" not in final_init_data: final_init_data["hhmm"] = "12:00"

        return CrowdQuery(**final_init_data)

    def resolve_location(self, q: CrowdQuery, city_bounds):
        """將地名解析為網格座標"""
        best_overall = None
        best_score = (-1.0, -1.0)
        lat_min, lng_min, lat_max, lng_max = city_bounds

        # 如果 place_variants 是空的，就用 place
        search_list = q.place_variants if q.place_variants else [q.place]

        for v in search_list:
            params = {"q": v, "format": "json", "countrycodes": "jp", "viewbox": f"{lng_min},{lat_max},{lng_max},{lat_min}", "bounded": 1}
            try:
                r = requests.get("https://nominatim.openstreetmap.org/search", params=params, headers={"User-Agent": "crowd-demo/1.0"}, timeout=5)
                results = r.json()
            except:
                continue
            
            for it in results:
                lat, lng = float(it["lat"]), float(it["lon"])
                if not (lat_min <= lat <= lat_max and lng_min <= lng <= lng_max): continue
                
                sim = similarity(q.place, best_label(it))
                imp = float(it.get("importance") or 0.0)
                if (sim, imp) > best_score:
                    best_score = (sim, imp)
                    best_overall = it

        if not best_overall: return None
        
        lat_found, lng_found = float(best_overall["lat"]), float(best_overall["lon"])
        _, idx = self.tree.query([lat_found, lng_found])
        
        print(f"✓ 定位成功: {best_overall.get('display_name')[:30]}... 經緯度: ({lat_found:.4f}, {lng_found:.4f})")
        
        return int(self.grid_xy_lookup.loc[idx, "x"]), int(self.grid_xy_lookup.loc[idx, "y"])

    def calculate_circles(self, q: CrowdQuery, x0, y0, coverages=(0.5, 0.8, 0.95)):
        """計算信心圓圈"""
        t_slot = self.time_to_slot(q.hhmm)
        
        # 尋找資料中最近的一天（處理 weekday 匹配）
        target_weekday = q.weekday if q.weekday is not None else 0
        possible_days = self.df_pred[self.df_pred["d"] % 7 == target_weekday]["d"]
        
        if possible_days.empty:
            d_use = self.df_pred["d"].max()
        else:
            d_use = int(possible_days.max())
        
        pred_slice = self.df_pred[(self.df_pred["d"]==d_use) & (self.df_pred["t"]==t_slot)].copy()
        
        if pred_slice.empty:
            raise ValueError(f"找不到日期 d={d_use}, 時段 t={t_slot} 的預測資料")

        circles = []
        p = normalize_nonneg(pred_slice["score"].to_numpy())
        coords = pred_slice[["x","y"]].to_numpy()
        
        for alpha in coverages:
            r, idx_circle = circle_radius_by_mass(coords, p, x0, y0, alpha)
            circles.append({
                "alpha": alpha, 
                "radius_cells": r, 
                "achieved_mass": float(p[idx_circle].sum())
            })
            
        return {
            "query": {"center_grid": {"x": x0, "y": y0}, "d_used": d_use, "t_slot": t_slot}, 
            "circles": circles
        }