import torch
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from src.geo.grid_to_latlng import GridLatLngMapper
from src.confidence_engine import ConfidenceEngine
from src.util import *
from src.preprocess import *
from src.ConvLSTM import *

# ==============================
# 路徑設定
# ==============================

CONVLSTM_PATH = "models/convlstm_sapporo.pkl"
PARQUET_PATH = "data/processed/sapporo_density.parquet"
SAVED_FILLED_ZERO_DATA_PATH = "data/processed/sapporo_density_filled_zero.parquet"

NEED_TRAIN = True     # GPU
REBUILD_DATA = True   # CPU


# ==============================
# GPU 強制檢查
# ==============================

def force_cuda():

    if not torch.cuda.is_available():
        raise RuntimeError("❌ CUDA 不可用，但你要求 GPU")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    print("🔥 使用 GPU:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)

    return device


# ==============================
# GPU sanity test
# ==============================

def sanity_gpu_test(device):

    print("🔍 GPU 計算測試...")

    x = torch.randn(1500, 1500, device=device)
    y = torch.matmul(x, x)

    torch.cuda.synchronize()

    print("✅ GPU matrix multiply 成功")


# ==============================
# GPU memory status
# ==============================

def gpu_status(tag=""):

    if torch.cuda.is_available():

        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2

        print(f"[GPU {tag}] allocated={allocated:.1f} MB | reserved={reserved:.1f} MB")

    else:

        print("CUDA not available")


# ==============================
# 主程式
# ==============================

def main():

    load_dotenv()
    client = OpenAI()

    # ==============================
    # GPU setup
    # ==============================

    device = force_cuda()

    sanity_gpu_test(device)

    gpu_status("after sanity test")

    # ==============================
    # 載入資料
    # ==============================

    print("正在載入原始資料...")

    df_raw = prepare_base_df(
        PARQUET_PATH,
        remove_sentinel=True
    )

    if REBUILD_DATA:

        print("重新補零資料中...")

        df_true = densify_topk_series(
            df_raw,
            top_k=50000
        )

        save_filled_zero_data(
            df_true,
            SAVED_FILLED_ZERO_DATA_PATH
        )

    else:

        df_true = pd.read_parquet(
            SAVED_FILLED_ZERO_DATA_PATH
        ).copy()

    # ==============================
    # 載入 ConvLSTM
    # ==============================

    print("載入 ConvLSTM 模型...")

    model, cfg = load_convlstm_embed(CONVLSTM_PATH)

    model = model.to(device)

    print("模型所在裝置:", next(model.parameters()).device)

    gpu_status("after model load")

    SEQ_LEN = int(cfg["seq_len"])

    # ==============================
    # 特徵工程
    # ==============================

    print("執行特徵工程...")

    df_feat = add_lags_and_rollings(
        df_true.copy(),
        seq_len=SEQ_LEN,
        start_date="2023-01-01"
    )

    need_cols = [f"lag_{k}" for k in range(1, SEQ_LEN + 1)]

    df_feat = df_feat.dropna(
        subset=need_cols
    ).copy()

    train_df, val_df, test_df = split_by_day(
        df_feat,
        test_days=7,
        val_days=7
    )

    print("Train / Val / Test size:")

    print(len(train_df), len(val_df), len(test_df))

    # ==============================
    # 訓練 ConvLSTM
    # ==============================

    if NEED_TRAIN:

        print("重新訓練 ConvLSTM...")

        model = ConvLSTMRegEmbed(
            hid_ch=cfg.get("hid_ch", 32),
            kernel_size=cfg.get("kernel_size", 3),
            n_weekday=7,
            n_t=48,
            n_x=int(df_feat["x"].max()) + 1,
            n_y=int(df_feat["y"].max()) + 1,
            emb_wd=cfg["emb_wd"],
            emb_t=cfg["emb_t"],
            emb_x=cfg["emb_x"],
            emb_y=cfg["emb_y"],
            mlp=cfg.get("mlp", 128),
        ).to(device)

        print("訓練前模型裝置:", next(model.parameters()).device)

        gpu_status("before training")

        model = train_convlstm_embed(
            model=model,
            train_df=train_df,
            val_df=val_df,
            seq_len=SEQ_LEN,
            patch_radius=cfg.get("patch_radius", 4),
            epochs=20,
            lr=1e-3,
            batch_size=256,
            use_residual=True,
            lookup_df=df_true
        )

        gpu_status("after training")

        save_convlstm_pkl(
            model,
            CONVLSTM_PATH,
            cfg
        )

    # ==============================
    # 預測
    # ==============================

    print("產出 ConvLSTM 預測...")

    df_pred = make_df_pred_convlstm_embed(
        df_feat,
        model,
        cfg,
        use_residual=True,
        lookup_df=df_true
    )

    # ==============================
    # Anchors
    # ==============================

    anchors = [

        {"x": 24, "y": 151, "lat": 43.06918333153887, "lng": 141.35147072116592},

        {"x": 24, "y": 148, "lat": 43.07940372979633, "lng": 141.34225589803765},

        {"x": 26, "y": 153, "lat": 43.05798589528942, "lng": 141.35402112326315},

        {"x": 52, "y": 81, "lat": 43.1982317547878, "lng": 140.99403634015297},

        {"x": 50, "y": 41, "lat": 43.188064114901195, "lng": 140.79455411455163},

        {"x": 182, "y": 186, "lat": 43.85360951281324, "lng": 141.52348814480132},
    ]

    mapper = GridLatLngMapper(anchors)

    engine = ConfidenceEngine(
        df_pred,
        mapper,
        client
    )

    city_bounds = (42.9, 140.7, 43.9, 141.6)

    user_input = "札幌車站 週六 12:00 附近"

    print("解析並運算中:", user_input)

    try:

        q_data = engine.parse_query_llm(user_input)

        xy = engine.resolve_location(
            q_data,
            city_bounds
        )

        if xy:

            resp = engine.calculate_circles(
                q_data,
                xy[0],
                xy[1]
            )

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

            print("✓ 地圖已輸出 predict_confidence.html")

        else:

            print("找不到地點")

    except Exception as e:

        print("發生錯誤:", e)


# ==============================

if __name__ == "__main__":
    main()