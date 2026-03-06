import requests
import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.spatial import KDTree

from src.str_similarity import similarity, best_label
from src.util import normalize_nonneg, circle_radius_by_mass
from src.geo.grid_to_latlng import cell_size_m_at


@dataclass
class RouteQuery:
    city: str
    start_place: str
    dest_place: str
    weekday: int | None
    hhmm: str | None
    radius_m: float | None


@dataclass
class CrowdQuery:
    city: str
    date: str | None
    weekday: int | None
    hhmm: str | None
    radius_m: float | None
    place: str | None


class ConfidenceEngine:
    def __init__(self, df_pred: pd.DataFrame, mapper, openai_client=None):
        self.df_pred = df_pred
        self.mapper = mapper
        self.client = openai_client

    def time_to_slot(self, hhmm: str) -> int:
        try:
            h, m = map(int, hhmm.split(":"))
            return (h * 60 + m) // 30
        except Exception:
            return 24

    def parse_route_query(self, text: str, default_city: str = "otaru") -> RouteQuery:
        """
        支援格式：
        小樽車站 小樽運河 週六 18:00 附近
        札幌站 小樽運河 週五 19:30 300
        """
        tokens = text.split()

        if len(tokens) < 4:
            raise ValueError("輸入格式錯誤，請使用：起點 終點 星期 時間 [附近/距離]")

        start_place = tokens[0]
        dest_place = tokens[1]

        weekday_map = {
            "週日": 0, "星期日": 0, "禮拜日": 0,
            "週一": 1, "星期一": 1, "禮拜一": 1,
            "週二": 2, "星期二": 2, "禮拜二": 2,
            "週三": 3, "星期三": 3, "禮拜三": 3,
            "週四": 4, "星期四": 4, "禮拜四": 4,
            "週五": 5, "星期五": 5, "禮拜五": 5,
            "週六": 6, "星期六": 6, "禮拜六": 6,
        }

        weekday = weekday_map.get(tokens[2], None)
        if weekday is None:
            raise ValueError("星期格式錯誤，例如：週六")

        hhmm = tokens[3]

        radius_m = 400.0
        if len(tokens) >= 5:
            tok = tokens[4].strip()
            if tok == "附近":
                radius_m = 400.0
            else:
                tok = tok.replace("公尺", "").replace("米", "").replace("m", "").replace("M", "")
                try:
                    radius_m = float(tok)
                except Exception:
                    radius_m = 400.0

        return RouteQuery(
            city=default_city,
            start_place=start_place,
            dest_place=dest_place,
            weekday=weekday,
            hhmm=hhmm,
            radius_m=radius_m,
        )

    def _search_place_best(self, place: str, city_bounds):
        """
        用 OSM Nominatim 搜尋地點，限制北海道小樽/札幌範圍。
        """
        lat_min, lng_min, lat_max, lng_max = city_bounds

        # 一些簡單別名，夠用就好，別把系統搞成香腸工廠
        variants = [
            place,
            f"{place} 北海道",
            f"{place} 小樽",
            f"{place} 札幌",
        ]

        best_overall = None
        best_score = (-1.0, -1.0)

        for v in variants:
            params = {
                "q": v,
                "format": "json",
                "countrycodes": "jp",
                "viewbox": f"{lng_min},{lat_max},{lng_max},{lat_min}",
                "bounded": 1,
            }
            try:
                r = requests.get(
                    "https://nominatim.openstreetmap.org/search",
                    params=params,
                    headers={"User-Agent": "crowd-demo/1.0"},
                    timeout=6,
                )
                results = r.json()
            except Exception:
                continue

            for it in results:
                lat = float(it["lat"])
                lng = float(it["lon"])

                if not (lat_min <= lat <= lat_max and lng_min <= lng <= lng_max):
                    continue

                sim = similarity(place, best_label(it))
                imp = float(it.get("importance") or 0.0)

                if (sim, imp) > best_score:
                    best_score = (sim, imp)
                    best_overall = it

        return best_overall

    def resolve_place_to_grid(self, place: str, city_bounds):
        """
        地名 -> (lat,lng) -> 反推 grid
        """
        best = self._search_place_best(place, city_bounds)
        if not best:
            return None

        lat_found = float(best["lat"])
        lng_found = float(best["lon"])

        x, y = self.mapper.latlng_to_grid(lat_found, lng_found)

        print(
            f"✓ 定位成功: {place} -> ({lat_found:.5f}, {lng_found:.5f}) -> grid({x}, {y})"
        )

        return {
            "place": place,
            "lat": lat_found,
            "lng": lng_found,
            "x": x,
            "y": y,
            "display_name": best.get("display_name", ""),
        }

    def resolve_two_locations(self, q: RouteQuery, city_bounds):
        start_info = self.resolve_place_to_grid(q.start_place, city_bounds)
        dest_info = self.resolve_place_to_grid(q.dest_place, city_bounds)

        if start_info is None or dest_info is None:
            return None

        return start_info, dest_info

    def calculate_circles(self, q: CrowdQuery, x0, y0, coverages=(0.5, 0.8, 0.95)):
        """
        保留舊功能：單點 crowd confidence circle
        """
        t_slot = self.time_to_slot(q.hhmm)

        tmp_days = self.df_pred[["d"]].drop_duplicates().copy()
        tmp_days["weekday"] = (tmp_days["d"] % 7)

        target_weekday = q.weekday if q.weekday is not None else 0
        cand_days = tmp_days[tmp_days["weekday"] == target_weekday]["d"].tolist()

        if not cand_days:
            raise ValueError("找不到對應 weekday 的資料日 d")

        d_use = int(cand_days[-1])

        pred_slice = self.df_pred[
            (self.df_pred["d"] == d_use) &
            (self.df_pred["t"] == t_slot)
        ][["x", "y", "score"]].copy()

        if pred_slice.empty:
            raise ValueError("該時段沒有預測資料")

        cell_m = cell_size_m_at(self.mapper, x0, y0)
        radius_m = 400 if q.radius_m is None else float(q.radius_m)
        radius_cells = int(np.ceil(radius_m / max(cell_m, 1e-6)))

        MAX_RADIUS_CELLS = 128
        MIN_CAND = 20

        cells_xy = pred_slice[["x", "y"]].to_numpy()
        dist = np.hypot(cells_xy[:, 0] - x0, cells_xy[:, 1] - y0)

        win0 = max(radius_cells, 3)

        while True:
            cand0 = pred_slice[dist <= win0 + 1e-9]
            if len(cand0) >= MIN_CAND or win0 >= MAX_RADIUS_CELLS:
                break
            win0 *= 2

        if len(cand0) < 5:
            raise ValueError("附近候選格太少，請加大 radius")

        circles = []

        for alpha in coverages:
            win = win0

            while True:
                cand = pred_slice[dist <= win + 1e-9]
                if len(cand) < 5:
                    raise ValueError("附近候選格太少")

                cand_xy = cand[["x", "y"]].to_numpy()
                p = normalize_nonneg(cand["score"].to_numpy())

                r, idx_circle = circle_radius_by_mass(cand_xy, p, x0, y0, alpha)
                achieved = float(p[idx_circle].sum())

                if achieved >= alpha or win >= MAX_RADIUS_CELLS:
                    break

                win *= 2

            circles.append({
                "alpha": alpha,
                "center_grid": {"x": float(x0), "y": float(y0)},
                "radius_cells": r,
                "n_cells": int(len(idx_circle)),
                "achieved_mass": achieved,
                "window_cells": win,
            })

        return {
            "query": {
                "city": q.city,
                "place": q.place,
                "weekday": q.weekday,
                "hhmm": q.hhmm,
                "t_slot": t_slot,
                "d_used": d_use,
                "center_grid": {"x": x0, "y": y0},
                "radius_m": float(radius_m),
                "radius_cells": radius_cells,
            },
            "circles": circles,
        }