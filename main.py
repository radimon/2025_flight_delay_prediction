import os
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
# 路徑設定ㄒ
# ==============================

CONVLSTM_PATH = "models/convlstm_sapporo.pkl"
PARQUET_PATH = "data/processed/sapporo_density.parquet"
SAVED_FILLED_ZERO_DATA_PATH = "data/processed/sapporo_density_filled_zero.parquet"

# ==============================
# 主要開關
# ==============================

NEED_TRAIN = True     # True: 訓練 / False: 直接載入模型並做推論
REBUILD_DATA = False   # True: 重建補零資料 / False: 讀取已補零資料

# ==============================
# 速度調參區（你主要改這裡）
# ==============================

DL_NUM_WORKERS = 0         # Windows 建議 0（避免 spawn pickle 炸裂）
TRAIN_BATCH_SIZE = 2048    # 訓練 batch（顯存夠可 1024）
PRED_BATCH_SIZE = 8192     # 推論 batch（通常可比訓練更大）
EPOCHS = 20                # 訓練最大 epoch（反正有 early stop）
USE_AMP = True             # 混合精度（建議開）
USE_COMPILE = False        # torch.compile（PyTorch 2.x 可開；第一次會 compile 慢）
PIN_MEMORY = True

# 動態裁切：只預測該點附近 R 格（越大越慢但越完整）
PRED_WINDOW_R = 40

# ==============================
# GPU 強制檢查 + 加速開關
# ==============================

def setup_torch_accel():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def force_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("❌ CUDA 不可用，但你要求 GPU")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    print("🔥 使用 GPU:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)
    return device


def gpu_status(tag=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"[GPU {tag}] allocated={allocated:.1f} MB | reserved={reserved:.1f} MB")


def _make_dummy_df_pred(df_feat: pd.DataFrame) -> pd.DataFrame:
    """
    給 ConfidenceEngine 初始化用的最小 df_pred（因為它 __init__ 會立刻建 KDTree）
    只要有 x,y 且不為空即可。
    """
    dummy = df_feat[["x", "y"]].drop_duplicates().head(5000).copy()
    # ConfidenceEngine 內部會用 df_pred[["d"]] 找 weekday，因此補最小 d/t/score
    dummy["d"] = 0
    dummy["t"] = 0
    dummy["score"] = 0.0
    return dummy[["d", "t", "x", "y", "score"]]


# ==============================
# 主程式
# ==============================

def main():
    load_dotenv()
    client = OpenAI()

    setup_torch_accel()
    device = force_cuda()
    gpu_status("startup")

    # ==============================
    # 載入資料
    # ==============================

    print("正在載入原始資料...")
    df_raw = prepare_base_df(PARQUET_PATH, remove_sentinel=True)

    if REBUILD_DATA:
        print("重新補零資料中...")
        df_true = densify_topk_series(df_raw, top_k=20000)
        save_filled_zero_data(df_true, SAVED_FILLED_ZERO_DATA_PATH)
    else:
        df_true = pd.read_parquet(SAVED_FILLED_ZERO_DATA_PATH).copy()
    
    # ==============================
    # 載入 ConvLSTM
    # ==============================

    if not os.path.exists(CONVLSTM_PATH) and not NEED_TRAIN:
        raise FileNotFoundError(
            f"找不到模型 {CONVLSTM_PATH}，但 NEED_TRAIN=False。請先訓練一次或把 NEED_TRAIN=True"
        )

    model, cfg = None, None
    if os.path.exists(CONVLSTM_PATH):
        print("載入 ConvLSTM 模型...")
        model, cfg = load_convlstm_embed(CONVLSTM_PATH, map_location="cpu")
        model = model.to(device)
        print("模型所在裝置:", next(model.parameters()).device)
        gpu_status("after model load")

    SEQ_LEN = int(cfg["seq_len"]) if cfg and "seq_len" in cfg else 7

    # ==============================
    # 特徵工程（全量一次；後面依使用者輸入裁切）
    # ==============================

    print("執行特徵工程...")
    df_feat = add_lags_and_rollings(
        df_true.copy(),
        seq_len=SEQ_LEN,
        start_date="2019-09-15"
    )

    need_cols = [f"lag_{k}" for k in range(1, SEQ_LEN + 1)]
    df_feat = df_feat.dropna(subset=need_cols).copy()

    train_df, val_df, test_df = split_by_day(df_feat, test_days=7, val_days=7)
    print("Train / Val / Test size:")
    print(len(train_df), len(val_df), len(test_df))

    # ==============================
    # 訓練（可選）
    # ==============================

    if NEED_TRAIN:
        print("重新訓練 ConvLSTM...")

        cfg = cfg or {}
        cfg.update({
            "seq_len": SEQ_LEN,
            "hid_ch": cfg.get("hid_ch", 32),
            "kernel_size": cfg.get("kernel_size", 3),
            "patch_radius": cfg.get("patch_radius", 4),
            "n_weekday": 7,
            "n_t": 48,
            "n_x": int(df_feat["x"].max()) + 1,
            "n_y": int(df_feat["y"].max()) + 1,
            "emb_wd": cfg.get("emb_wd", 2),
            "emb_t": cfg.get("emb_t", 8),
            "emb_x": cfg.get("emb_x", 16),
            "emb_y": cfg.get("emb_y", 16),
            "mlp": cfg.get("mlp", 128),
            "cube_dtype": cfg.get("cube_dtype", "float16"),
        })

        model = ConvLSTMRegEmbed(
            hid_ch=cfg["hid_ch"],
            kernel_size=cfg["kernel_size"],
            n_weekday=cfg["n_weekday"],
            n_t=cfg["n_t"],
            n_x=cfg["n_x"],
            n_y=cfg["n_y"],
            emb_wd=cfg["emb_wd"],
            emb_t=cfg["emb_t"],
            emb_x=cfg["emb_x"],
            emb_y=cfg["emb_y"],
            mlp=cfg["mlp"],
        ).to(device)

        if USE_COMPILE:
            model = torch.compile(model)

        gpu_status("before training")

        model = train_convlstm_embed(
            model=model,
            train_df=train_df,
            val_df=val_df,
            seq_len=SEQ_LEN,
            patch_radius=cfg.get("patch_radius", 4),
            epochs=EPOCHS,
            lr=1e-3,
            batch_size=TRAIN_BATCH_SIZE,
            sample_n=100_000,
            use_residual=True,
            lookup_df=df_true,
            amp=USE_AMP,
            num_workers=DL_NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            prefetch_factor=4,
            persistent_workers=True,
        )

        gpu_status("after training")
        save_convlstm_pkl(model, CONVLSTM_PATH, cfg)

    # ==============================
    # Anchors / Mapper
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
    city_bounds = (42.9, 140.7, 43.9, 141.6)

    # ==============================
    # 使用者輸入（CLI 雛形）
    # ==============================

    user_input = input("請輸入查詢（例：小樽車站 週六 18:00 附近）：").strip()
    if not user_input:
        user_input = "小樽車站 週六 18:00 附近"

    print("解析並運算中:", user_input)

    try:
        # 先用 dummy df_pred 建 engine（因為 __init__ 會建 KDTree，不能 df_pred=None）
        dummy_pred = _make_dummy_df_pred(df_feat)
        engine_tmp = ConfidenceEngine(dummy_pred, mapper, client)

        q = engine_tmp.parse_query_llm(user_input)
        print("DEBUG q =", q)

        # resolve_location 需要 KDTree，所以也用 engine_tmp
        xy = engine_tmp.resolve_location(q, city_bounds)

        # ---- 動態裁切 df_feat（依 q + xy）----
        df_feat_small = df_feat

        # time slot
        t_slot = engine_tmp.time_to_slot(q.hhmm) if q.hhmm else 24
        if "t" in df_feat_small.columns:
            df_feat_small = df_feat_small[df_feat_small["t"] == int(t_slot)].copy()

        # weekday
        if q.weekday is not None and "weekday" in df_feat_small.columns:
            df_feat_small = df_feat_small[df_feat_small["weekday"] == int(q.weekday)].copy()

        # window around xy
        if xy is not None:
            x0, y0 = int(xy[0]), int(xy[1])
            R = int(PRED_WINDOW_R)
            df_feat_small = df_feat_small[
                (df_feat_small["x"].between(x0 - R, x0 + R)) &
                (df_feat_small["y"].between(y0 - R, y0 + R))
            ].copy()

        print("✅ 這次需要預測的資料量:", len(df_feat_small), "rows")

        # ==============================
        # 只對裁切後資料做預測
        # ==============================

        print("產出 ConvLSTM 預測（裁切後）...")

        df_pred_small = make_df_pred_convlstm_embed(
            df_feat_small,
            model,
            cfg,
            use_residual=True,
            lookup_df=df_true,
            batch_size=PRED_BATCH_SIZE,
            num_workers=DL_NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            amp=USE_AMP
        )

        # 用真正 df_pred_small 建「正式 engine」（KDTree 對齊）
        engine = ConfidenceEngine(df_pred_small, mapper, client)

        if xy:
            resp = engine.calculate_circles(q, xy[0], xy[1])
            plot_query_on_map(
                resp,
                mapper,
                df_pred_small,
                {
                    "lat_min": city_bounds[0],
                    "lng_min": city_bounds[1],
                    "lat_max": city_bounds[2],
                    "lng_max": city_bounds[3],
                }
            )
            print("✓ 地圖已輸出 predict_confidence.html")
        else:
            print("找不到地點（resolve_location 回傳 None）")

    except Exception as e:
        print("發生錯誤:", e)


if __name__ == "__main__":
    main()