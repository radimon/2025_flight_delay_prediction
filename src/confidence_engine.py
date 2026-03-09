import json
import requests
import numpy as np
import pandas as pd

from dataclasses import dataclass
from src.str_similarity import similarity, best_label
from src.util import normalize_nonneg, circle_radius_by_mass
from src.geo.grid_to_latlng import cell_size_m_at

# ===============================
# Query dataclass
# ===============================

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

# ===============================
# LLM Schema
# ===============================

QUERY_SCHEMA = {
    "name": "route_query",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "city": {"type": "string"},
            "start_place": {"type": "string"},
            "dest_place": {"type": "string"},
            "weekday": {"type": ["integer", "null"], "minimum": 0, "maximum": 6},
            "hhmm": {"type": ["string", "null"]},
            "radius_m": {"type": ["number", "null"]}
        },
        "required": ["start_place", "dest_place"]
    }
}

# ===============================
# Confidence Engine
# ===============================

class ConfidenceEngine:

    def __init__(self, df_pred: pd.DataFrame, mapper, openai_client=None, google_api_key=None):
        self.df_pred = df_pred
        self.mapper = mapper
        self.client = openai_client
        self.google_api_key = google_api_key  # 建議從外部傳入 API Key

    # --------------------------------
    # time slot
    # --------------------------------

    def time_to_slot(self, hhmm: str) -> int:
        try:
            h, m = map(int, hhmm.split(":"))
            return (h * 60 + m) // 30
        except Exception:
            return 24

    # --------------------------------
    # LLM parsing
    # --------------------------------

    def parse_route_query_llm(self, text: str) -> RouteQuery:
        if self.client is None:
            return self.parse_route_query(text)

        # 這裡根據你的截圖調整為正確的呼叫方式
        resp = self.client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {"role": "system", "content": "Extract route query information."},
                {"role": "user", "content": text}
            ],
            tools=[{
                "type": "function",
                "name": QUERY_SCHEMA["name"],
                "parameters": QUERY_SCHEMA["schema"]
            }],
            tool_choice={"type": "function", "name": "route_query"}
        )

        tool_call = resp.output[0]
        args = json.loads(tool_call.arguments)

        return RouteQuery(
            city=args.get("city", "sapporo"),
            start_place=args.get("start_place"),
            dest_place=args.get("dest_place"),
            weekday=args.get("weekday"),
            hhmm=args.get("hhmm"),
            radius_m=args.get("radius_m")
        )

    # --------------------------------
    # Google Geocoding API (精準定位)
    # --------------------------------

    def _search_place_best(self, place, city_bounds):
        """
        使用 Google Geocoding API 取得地點座標
        """

        if not self.google_api_key:
            print("❌ Google API key 未設定")
            return None

        lat_min, lng_min, lat_max, lng_max = city_bounds

        url = "https://maps.googleapis.com/maps/api/geocode/json"

        params = {
            "address": place,
            "key": self.google_api_key,
            "language": "zh-TW",
            "region": "jp"
        }

        try:

            r = requests.get(url, params=params, timeout=10)
            data = r.json()

            if data.get("status") != "OK":
                print(f"❌ Geocoding failed: {data.get('status')} ({place})")
                return None

            result = data["results"][0]

            lat = result["geometry"]["location"]["lat"]
            lng = result["geometry"]["location"]["lng"]

            print(f"Google Geocode: {place} -> {lat},{lng}")

            # optional: 檢查是否在城市範圍內
            if not (lat_min <= lat <= lat_max and lng_min <= lng <= lng_max):
                print(f"⚠ {place} 在範圍外 ({lat},{lng})")

            return {
                "lat": lat,
                "lon": lng,
                "display_name": result.get("formatted_address", place)
            }

        except Exception as e:

            print("❌ Google API error:", e)

            return None

    # --------------------------------
    # place -> grid (核心邏輯：將精準座標轉回格網)
    # --------------------------------

    def resolve_place_to_grid(self, place, city_bounds):
        # 呼叫上面修改過的 Google API
        best = self._search_place_best(place, city_bounds)

        if not best:
            return None

        lat = float(best["lat"])
        lng = float(best["lon"])

        # 這裡不會受到 API 更換影響，mapper 依然能正常運作
        x, y = self.mapper.latlng_to_grid(lat, lng)

        print(f"✓ {place} -> ({lat:.5f},{lng:.5f}) -> grid({x},{y})")

        return {
            "place": place,
            "lat": lat,
            "lng": lng,
            "x": x,
            "y": y,
            "display_name": best.get("display_name", "")
        }

    # --------------------------------
    # resolve start / dest
    # --------------------------------

    def resolve_two_locations(self, q, city_bounds):
        start = self.resolve_place_to_grid(q.start_place, city_bounds)
        dest = self.resolve_place_to_grid(q.dest_place, city_bounds)

        if start is None or dest is None:
            return None

        return start, dest

    # --------------------------------
    # fallback parser (保持不變)
    # --------------------------------

    def parse_route_query(self, text: str, default_city: str = "sapporo") -> RouteQuery:
        tokens = text.split()
        if len(tokens) < 4:
            raise ValueError("輸入格式錯誤：起點 終點 星期 時間")

        start_place = tokens[0]
        dest_place = tokens[1]
        weekday_map = {
            "週日": 0, "週一": 1, "週二": 2, "週三": 3, 
            "週四": 4, "週五": 5, "週六": 6
        }
        weekday = weekday_map.get(tokens[2], None)
        hhmm = tokens[3]
        radius_m = 400

        if len(tokens) >= 5:
            tok = tokens[4]
            if tok != "附近":
                tok = tok.replace("公尺", "").replace("m", "")
                try:
                    radius_m = float(tok)
                except:
                    radius_m = 400

        return RouteQuery(
            city=default_city,
            start_place=start_place,
            dest_place=dest_place,
            weekday=weekday,
            hhmm=hhmm,
            radius_m=radius_m
        )