import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from src.LSTM import LSTMRegEmbed, train_lstm_embed

#切資料
def split_by_day(df, test_days=7, val_days=7):
    max_day = int(df["d"].max())
    test_start = max_day - test_days + 1
    val_start  = test_start - val_days

    train_df = df[df["d"] < val_start].copy()
    val_df   = df[(df["d"] >= val_start) & (df["d"] < test_start)].copy()
    test_df  = df[df["d"] >= test_start].copy()

    return train_df, val_df, test_df

def densify_topk_series(df_raw: pd.DataFrame, top_k=20000):
    """補齊時間序列缺失值 (用於訓練前處理)"""
    df = df_raw[["d","t","x","y","count"]].copy()
    key_sum = df.groupby(["x","y","t"])["count"].sum().sort_values(ascending=False)
    top_keys = key_sum.head(top_k).index
    df = df.set_index(["x","y","t"]).loc[top_keys].reset_index()
    dmin, dmax = int(df["d"].min()), int(df["d"].max())
    all_d = np.arange(dmin, dmax + 1, dtype=int)
    out = []
    for (x,y,t), g in df.groupby(["x","y","t"], sort=False):
        g2 = g.set_index("d").reindex(all_d)
        g2["count"] = g2["count"].fillna(0.0); g2["d"] = all_d
        g2["x"] = x; g2["y"] = y; g2["t"] = t
        out.append(g2[["d","t","x","y","count"]])
    return pd.concat(out, ignore_index=True)

def save_filled_zero_data(df_true: pd.DataFrame, saved_path):
    df_true.to_parquet(saved_path, index=False)

def prepare_base_df(parquet_path, remove_sentinel=True):
    df = pd.read_parquet(parquet_path).copy()
    if remove_sentinel:
        df = df[~((df["x"]==999) & (df["y"]==999))].copy()

    return df

def add_lags_and_rollings(df, seq_len=8, start_date="2023-01-01"):
    """
    Make lag features consistent with ConvLSTM / PredRNN offsets:
        lag_1 ~ lag_6 : same day previous 1~6 time slots
        lag_7         : previous day same time slot
        lag_8         : previous week same time slot

    Notes
    -----
    - Assumes df has columns: d, t, x, y, count
    - Assumes time slots t are ordered discrete indices (e.g. 0~47)
    - Keeps output column names lag_1 ... lag_8 unchanged,
      so downstream notebook code does not need modification.
    """

    if seq_len != 8:
        raise ValueError(
            f"This version is designed for seq_len=8 to match ConvLSTM offsets, got seq_len={seq_len}."
        )

    df = df.copy()

    # calendar features
    df["date"] = pd.to_datetime(start_date) + pd.to_timedelta(df["d"], unit="D")
    df["weekday"] = df["date"].dt.weekday
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)

    # base index for self-merge
    base = df[["d", "t", "x", "y", "count"]].copy()

    # offsets aligned with ConvLSTM / PredRNN
    offsets = [
        (0, -1),
        (0, -2),
        (0, -3),
        (0, -4),
        (0, -5),
        (0, -6),
        (-1, 0),
        (-7, 0),
    ]

    for i, (d_shift, t_shift) in enumerate(offsets, start=1):
        tmp = base.copy()
        tmp["d"] = tmp["d"] - d_shift
        tmp["t"] = tmp["t"] - t_shift
        tmp = tmp.rename(columns={"count": f"lag_{i}"})
        df = df.merge(
            tmp[["d", "t", "x", "y", f"lag_{i}"]],
            on=["d", "t", "x", "y"],
            how="left",
        )

    return df


