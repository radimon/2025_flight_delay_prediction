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

def add_lags_and_rollings(df, seq_len, start_date="2023-01-01"):
    # weekday/is_weekend（你如果有更準的 d->date 對照，就改這裡）
    df["date"] = pd.to_datetime(start_date) + pd.to_timedelta(df["d"], unit="D")
    df["weekday"] = df["date"].dt.weekday
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)

    df = df.sort_values(["x","y","t","d"]).reset_index(drop=True)
    g = df.groupby(["x","y","t"])["count"]

    # LSTM 用：lag_1..lag_seq_len
    for k in range(1, seq_len+1):
        df[f"lag_{k}"] = g.shift(k)

    return df


