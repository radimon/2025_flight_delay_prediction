import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

class GridLatLngMapper:
    def __init__(self, anchors):
        """
        anchors: 一個 list of dict，例如 [{"x": 24, "y": 151, "lat": 43.069, "lng": 141.351}, ...]
        """
        self.df_anchors = pd.DataFrame(anchors)
        
        # 訓練兩個簡單的線性模型：(x, y) -> lat 以及 (x, y) -> lng
        self.model_lat = LinearRegression().fit(self.df_anchors[['x', 'y']], self.df_anchors['lat'])
        self.model_lng = LinearRegression().fit(self.df_anchors[['x', 'y']], self.df_anchors['lng'])

    def transform(self, df):
        """
        輸入包含 x, y 欄位的 DataFrame，回傳加上 lat, lng 欄位的 DataFrame
        """
        res = df.copy()
        # 預測經緯度
        res['lat'] = self.model_lat.predict(df[['x', 'y']])
        res['lng'] = self.model_lng.predict(df[['x', 'y']])
        return res

def haversine_m(lat1, lon1, lat2, lon2):
    """計算兩點間的球面距離（公尺）"""
    R = 6371000.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return float(2*R*np.arcsin(np.sqrt(a)))

# 向量版
def haversine_m_vec(lat1, lon1, lat2, lon2):
    """
    lat1, lon1: scalar or array
    lat2, lon2: scalar or array
    return: meters (numpy array)
    """
    R = 6371000.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def cell_size_m_at(mapper, x0, y0):
    """計算特定網格在現實世界中的約略尺寸"""
    # 建立目前網格、右方網格、上方網格的三個點
    pts = pd.DataFrame([{"x":x0,"y":y0}, {"x":x0+1,"y":y0}, {"x":x0,"y":y0+1}])
    pts = mapper.transform(pts)
    
    c = pts.iloc[0]  # 中心點
    px = pts.iloc[1] # X 方向移動一格
    py = pts.iloc[2] # Y 方向移動一格
    
    # 計算 X 與 Y 方向的物理距離 (公尺)
    dx = haversine_m(c.lat, c.lng, px.lat, px.lng)
    dy = haversine_m(c.lat, c.lng, py.lat, py.lng)
    
    # 回傳平均一格的大小
    return (dx + dy) / 2