import joblib
import pandas as pd
import sys
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

from src.geo.grid_to_latlng import GridLatLngMapper
from src.confidence_engine import ConfidenceEngine
from src.util import *
from src.preprocess import *
from src.LSTM import *
from src.ConvLSTM import *

LSTM_PATH = "models/lstm_sapporo.pkl"
CONVLSTM_PATH = "models/convlstm_sapporo.pkl"
PARQUET_PATH = "data/processed/sapporo_density.parquet"
SAVED_FILLED_ZERO_DATA_PATH = "data/processed/sapporo_density_filled_zero.parquet"

NEED_TRAIN = False
REBUILD_DATA = False

def main():

    load_dotenv()
    client = OpenAI()

    print("正在載入模型與原始資料...")
    df_raw = prepare_base_df(PARQUET_PATH, remove_sentinel=True)

    if REBUILD_DATA:
        df_true = densify_topk_series(df_raw, top_k=20000) # 對raw data進行補零
        save_filled_zero_data(df_true, SAVED_FILLED_ZERO_DATA_PATH)
    else:
        df_true = pd.read_parquet(SAVED_FILLED_ZERO_DATA_PATH).copy()

    model, cfg = load_lstm_embed(LSTM_PATH) #載入model
    SEQ_LEN = cfg["seq_len"]

    print("執行特徵工程中...")
    df_feat = add_lags_and_rollings(df_true.copy(), seq_len=SEQ_LEN, start_date="2023-01-01")

    need_lstm = [f"lag_{k}" for k in range(1, SEQ_LEN+1)]
    df_feat = df_feat.dropna(subset=need_lstm).copy()

    train_df, val_df, test_df = split_by_day(df_feat, test_days=7, val_days=7)
    print(len(train_df), len(val_df), len(test_df))

    #若需要重新訓練
    if NEED_TRAIN:
        # 用更大的 top_k 重新產資料，就應該 同步更新 cfg 的 n_x/n_y 再建模
        cfg["n_weekday"] = 7
        cfg["n_t"] = 48
        cfg["n_x"] = int(df_feat["x"].max()) + 1
        cfg["n_y"] = int(df_feat["y"].max()) + 1

        # 不要用 load 進來的舊 model，改用新 cfg 重建
        model = LSTMRegEmbed(
            hidden=cfg["hidden"], layers=cfg["layers"],
            n_weekday=cfg["n_weekday"], n_t=cfg["n_t"],
            n_x=cfg["n_x"], n_y=cfg["n_y"],
            emb_wd=cfg["emb_wd"], emb_t=cfg["emb_t"],
            emb_x=cfg["emb_x"], emb_y=cfg["emb_y"],
        )

        # train
        model = train_lstm_embed(
            model=model,
            train_df=train_df,
            val_df=val_df,
            seq_len=SEQ_LEN,
            epochs=20,
            lr=1e-3,
            batch_size=1024,
            seed=42,
            loss="huber", 
            huber_beta=2.0,
            use_residual=True
        )

        save_lstm_pkl(model, LSTM_PATH, cfg)

    # 產出預測的人流
    df_pred = make_df_pred_lstm_embed(df_feat, model, cfg)
   

    anchors = [
        {"x": 24, "y": 151, "lat": 43.06918333153887, "lng": 141.35147072116592},  # 札幌站 
        {"x": 24, "y": 148, "lat": 43.07940372979633, "lng": 141.34225589803765},  # 北海道大學
        {"x": 26, "y": 153, "lat": 43.05798589528942, "lng": 141.35402112326315},  # 狸小路商店街
        {"x": 52, "y": 81, "lat": 43.1982317547878, "lng": 140.99403634015297},  # 小樽站
        {"x": 50, "y": 41, "lat": 43.188064114901195, "lng": 140.79455411455163},  # 余市站
        {"x": 182, "y": 186, "lat": 43.85360951281324, "lng": 141.52348814480132},  # 増毛町文化センター
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