import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


class GridLatLngMapper:
    def __init__(self, anchors):
        """
        anchors: [{"x": 24, "y": 151, "lat": 43.069, "lng": 141.351}, ...]
        """
        self.df_anchors = pd.DataFrame(anchors).copy()

        self.model_lat = LinearRegression().fit(
            self.df_anchors[["x", "y"]],
            self.df_anchors["lat"]
        )
        self.model_lng = LinearRegression().fit(
            self.df_anchors[["x", "y"]],
            self.df_anchors["lng"]
        )

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        輸入包含 x, y 欄位的 DataFrame，回傳加上 lat, lng 欄位的 DataFrame
        """
        res = df.copy()
        res["lat"] = self.model_lat.predict(df[["x", "y"]])
        res["lng"] = self.model_lng.predict(df[["x", "y"]])
        return res

    def grid_to_latlng(self, x: int, y: int) -> tuple[float, float]:
        """
        單點 grid -> lat/lng
        """
        df = pd.DataFrame([{"x": x, "y": y}])
        res = self.transform(df)
        lat = float(res.iloc[0]["lat"])
        lng = float(res.iloc[0]["lng"])
        return lat, lng

    def latlng_to_grid(self, lat: float, lng: float) -> tuple[int, int]:
        """
        反推 lat/lng -> grid
        利用兩個線性方程：
            lat = a*x + b*y + c
            lng = d*x + e*y + f
        解出 x, y
        """
        A = np.array([
            [self.model_lat.coef_[0], self.model_lat.coef_[1]],
            [self.model_lng.coef_[0], self.model_lng.coef_[1]],
        ], dtype=float)

        b = np.array([
            lat - float(self.model_lat.intercept_),
            lng - float(self.model_lng.intercept_),
        ], dtype=float)

        try:
            x, y = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # 退化情況用 least squares
            x, y = np.linalg.lstsq(A, b, rcond=None)[0]

        return int(round(x)), int(round(y))


def haversine_m(lat1, lon1, lat2, lon2):
    """計算兩點間球面距離（公尺）"""
    R = 6371000.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return float(2 * R * np.arcsin(np.sqrt(a)))


def cell_size_m_at(mapper: GridLatLngMapper, x0: int, y0: int):
    """估計特定 grid 一格在現實世界的大小（公尺）"""
    pts = pd.DataFrame([
        {"x": x0, "y": y0},
        {"x": x0 + 1, "y": y0},
        {"x": x0, "y": y0 + 1},
    ])
    pts = mapper.transform(pts)

    c = pts.iloc[0]
    px = pts.iloc[1]
    py = pts.iloc[2]

    dx = haversine_m(c.lat, c.lng, px.lat, px.lng)
    dy = haversine_m(c.lat, c.lng, py.lat, py.lng)

    return (dx + dy) / 2.0