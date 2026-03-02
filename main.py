import joblib
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from src.geo.grid_to_latlng import GridLatLngMapper
from src.confidence_engine import ConfidenceEngine
from src.util import plot_query_on_map


def main():

    load_dotenv()
    client = OpenAI()

    print("正在載入模型與原始資料...")
    model = joblib.load("models/lgbm_sapporo_2.pkl")
    df = pd.read_parquet("data/processed/sapporo_density.parquet")

    print("執行特徵工程中...")

    df = df[(df["x"] != 999) | (df["y"] != 999)].copy()

    df["date"] = pd.to_datetime("2023-01-01") + pd.to_timedelta(df["d"], unit="D")
    df["weekday"] = df["date"].dt.weekday
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)

    df = df.sort_values(["x", "y", "t", "d"])

    df["lag_1"] = df.groupby(["x", "y", "t"])["count"].shift(1)
    df["lag_7"] = df.groupby(["x", "y", "t"])["count"].shift(7)
    df["rolling_3"] = df.groupby(["x", "y", "t"])["count"].transform(
        lambda x: x.shift(1).rolling(3).mean()
    )
    df["rolling_7"] = df.groupby(["x", "y", "t"])["count"].transform(
        lambda x: x.shift(1).rolling(7).mean()
    )

    df_feat = df.dropna().copy()

    features = [
        "weekday", "t", "x", "y",
        "is_weekend", "lag_1", "lag_7",
        "rolling_3", "rolling_7"
    ]

    print(f"預測中 (樣本數: {len(df_feat)})...")
    df_feat["score"] = model.predict(df_feat[features])
    df_feat["score"] = df_feat["score"].clip(lower=0)

    df_pred = df_feat[["d", "t", "x", "y", "score"]].copy()

    anchors = [
        {"x": 24, "y": 151, "lat": 43.069183, "lng": 141.351470},
        {"x": 26, "y": 153, "lat": 43.057985, "lng": 141.354021},
        {"x": 52, "y": 81, "lat": 43.198231, "lng": 140.994036}
    ]

    mapper = GridLatLngMapper(anchors)
    engine = ConfidenceEngine(df_pred, mapper, client)

    city_bounds = (42.9, 140.7, 43.9, 141.6)

    user_input = "札幌邱珠空港 週六 12:00 附近"
    print(f"解析並運算中: {user_input}")

    try:
        q_data = engine.parse_query_llm(user_input)
        xy = engine.resolve_location(q_data, city_bounds)

        if xy:
            resp = engine.calculate_circles(q_data, xy[0], xy[1])

            plot_query_on_map(
                resp,
                mapper,
                df_pred,
                {
                    "lat_min": city_bounds[0],
                    "lng_min": city_bounds[1],
                    "lat_max": city_bounds[2],
                    "lng_max": city_bounds[3],
                }
            )

            print("✓ 成功！地圖已儲存為 predict_confidence.html")
        else:
            print("找不到地點。")

    except Exception as e:
        print(f"發生錯誤: {e}")


if __name__ == "__main__":
    main()