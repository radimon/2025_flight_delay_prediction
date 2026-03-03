import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class SeqDatasetEmbed(Dataset):
    def __init__(self, df, seq_len):
        self.seq_len = seq_len
        # 序列：lag_seq_len ... lag_1（越後面越接近現在）
        lag_cols = [f"lag_{k}" for k in range(seq_len, 0, -1)]
        self.X_seq = df[lag_cols].to_numpy(dtype=np.float32)[:, :, None]  # (N,L,1)

        # ids
        self.weekday = df['weekday'].to_numpy(dtype=np.int64)
        self.t_id = df['t'].to_numpy(dtype=np.int64)
        self.x_id = df['x'].to_numpy(dtype=np.int64)
        self.y_id = df['y'].to_numpy(dtype=np.int64)
        self.is_weekend = df["is_weekend"].to_numpy(dtype=np.float32)

         # baseline log（可選）
        self.base_log = df["base_log"].to_numpy(dtype=np.float32) if "base_log" in df.columns else None

        # target：若有 y_target 就用（residual learning），否則維持原本 log1p(count)
        if "y_target" in df.columns:
            self.y = df["y_target"].to_numpy(dtype=np.float32)
        else:
            self.y = np.log1p(df["count"].to_numpy(dtype=np.float32))

    def __len__(self): return len(self.y)

    def __getitem__(self, i):
        base_log_i = 0.0 if self.base_log is None else float(self.base_log[i])
        return (
            torch.from_numpy(self.X_seq[i]),
            torch.tensor(self.weekday[i], dtype=torch.long),
            torch.tensor(self.t_id[i], dtype=torch.long),
            torch.tensor(self.x_id[i], dtype=torch.long),
            torch.tensor(self.y_id[i], dtype=torch.long),
            torch.tensor([self.is_weekend[i]], dtype=torch.float32),  # shape (1,)
            torch.tensor([base_log_i], dtype=torch.float32),
            torch.tensor(self.y[i], dtype=torch.float32),
        )

class LSTMRegEmbed(nn.Module):
    def __init__(self, hidden=64, layers=1, n_weekday=7, n_t=48, n_x=2000, n_y=2000,
                 emb_wd=2, emb_t=8, emb_x=16, emb_y=16):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, num_layers=layers,
                            batch_first=True, dropout=0.1 if layers > 1 else 0.0)
        
        # embeddings
        self.emb_weekday = nn.Embedding(n_weekday, emb_wd)
        self.emb_t = nn.Embedding(n_t, emb_t)
        self.emb_x = nn.Embedding(n_x, emb_x)
        self.emb_y = nn.Embedding(n_y, emb_y)

        static_dim = emb_wd + emb_t + emb_x + emb_y + 1  # + is_weekend(1)

        self.head = nn.Sequential(
            nn.Linear(hidden + static_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x_seq, weekday, t_id, x_id, y_id, is_weekend):
        out, _ = self.lstm(x_seq)          # (B,L,H)
        h = out[:, -1, :]                  # last hidden
        e = torch.cat([
            self.emb_weekday(weekday),
            self.emb_t(t_id),
            self.emb_x(x_id),
            self.emb_y(y_id),
            is_weekend,  # already (B,1)
        ], dim=1)

        z = torch.cat([h, e], dim=1)
        return self.head(z).squeeze(1)     # log1p(count)
    

def train_lstm_embed(model, train_df, val_df, seq_len, 
                     loss="mse", huber_beta=1.0, use_residual=False,
                     sample_n=800_000, batch_size=1024, epochs=20, lr=1e-3, seed=42):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr = train_df.sample(n=min(sample_n, len(train_df)), random_state=seed)
    ds_tr = SeqDatasetEmbed(tr, seq_len)
    ds_va = SeqDatasetEmbed(val_df, seq_len)
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=0)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=0)

    n_x = int(pd.concat([train_df["x"], val_df["x"]]).max()) + 1
    n_y = int(pd.concat([train_df["y"], val_df["y"]]).max()) + 1
    n_t = int(pd.concat([train_df["t"], val_df["t"]]).max()) + 1

    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    if loss == "huber":
        loss_fn = nn.SmoothL1Loss(beta=huber_beta)   # Huber
    else:
        loss_fn = nn.MSELoss()

    best_rmse = float("inf")
    best_state = None
    patience, bad = 3, 0

    for ep in range(1, epochs+1):
        model.train()
        for xseq, wd, tid, xid, yid, isw, base_log, y in dl_tr:
            xseq, wd, tid, xid, yid, isw, base_log, y = (
                xseq.to(device), wd.to(device), tid.to(device), xid.to(device),
                yid.to(device), isw.to(device), base_log.to(device), y.to(device)
            )
            pred = model(xseq, wd, tid, xid, yid, isw)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

        # val RMSE（在原始 count 尺度）
        model.eval()
        preds, ys = [], []
        with torch.no_grad():
            for xseq, wd, tid, xid, yid, isw, base_log, y in dl_va:
                xseq, wd, tid, xid, yid, isw, base_log = (
                    xseq.to(device), wd.to(device), tid.to(device), xid.to(device),
                    yid.to(device), isw.to(device), base_log.to(device)
                )
                pred_log = model(xseq, wd, tid, xid, yid, isw)

                if use_residual:
                    # pred_log 是 residual(log space)，加回 baseline log
                    pred_log_full = pred_log + base_log.squeeze(1)
                    y_log_full = y + base_log.squeeze(1)
                else:
                    pred_log_full = pred_log
                    y_log_full = y

                preds.append(np.expm1(pred_log_full.cpu().numpy()))
                ys.append(np.expm1(y_log_full.cpu().numpy()))

        yhat = np.concatenate(preds)
        ytrue = np.concatenate(ys)
        rmse = float(root_mean_squared_error(ytrue, yhat))
        print(f"epoch {ep}: val RMSE={rmse:.4f}")

        if rmse < best_rmse - 1e-4:
            best_rmse = rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print("Early stop.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def predict_lstm_embed(model, df_split, seq_len, batch_size=2048, use_residual=False):
    device = next(model.parameters()).device
    ds = SeqDatasetEmbed(df_split, seq_len)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    preds = []
    with torch.no_grad():
        for xseq, wd, tid, xid, yid, isw, base_log, _y in dl:
            xseq, wd, tid, xid, yid, isw, base_log = (
                xseq.to(device), wd.to(device), tid.to(device), xid.to(device),
                yid.to(device), isw.to(device), base_log.to(device)
            )
            pred_log = model(xseq, wd, tid, xid, yid, isw)
            
            if use_residual:
                pred_log = pred_log + base_log.squeeze(1)

            preds.append(np.expm1(pred_log.cpu().numpy()))

    yhat = np.concatenate(preds)
    return np.clip(yhat, 0, None)

def make_df_pred_lstm_embed(df_feat, lstm_model, cfg, batch_size=2048, use_residual=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lstm_model.to(device)
    lstm_model.eval()

    seq_len = cfg["seq_len"]
    lag_cols = [f"lag_{k}" for k in range(seq_len, 0, -1)]  # 舊→新

    X_seq = df_feat[lag_cols].to_numpy(np.float32)[:, :, None]
    weekday = df_feat["weekday"].to_numpy(np.int64)
    t_id    = df_feat["t"].to_numpy(np.int64)
    x_id    = df_feat["x"].to_numpy(np.int64)
    y_id    = df_feat["y"].to_numpy(np.int64)
    is_wknd = df_feat["is_weekend"].to_numpy(np.float32)[:, None]

    if use_residual:
        if "base_log" not in df_feat.columns:
            raise ValueError("use_residual=True 需要 df_feat 內有 'base_log' 欄位（log1p(y_hist)）。")
        base_log_all = df_feat["base_log"].to_numpy(np.float32)

    preds = []
    n = len(df_feat)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            xs = torch.from_numpy(X_seq[i:i+batch_size]).to(device)
            wd = torch.from_numpy(weekday[i:i+batch_size]).to(device)
            tt = torch.from_numpy(t_id[i:i+batch_size]).to(device)
            xx = torch.from_numpy(x_id[i:i+batch_size]).to(device)
            yy = torch.from_numpy(y_id[i:i+batch_size]).to(device)
            wk = torch.from_numpy(is_wknd[i:i+batch_size]).to(device)

            pred_log = lstm_model(xs, wd, tt, xx, yy, wk).cpu().numpy()  # log1p(count)

            if use_residual:
                base = torch.from_numpy(base_log_all[i:i+batch_size]).to(device)  # (B,)
                pred_log = pred_log + base

            pred = np.expm1(pred_log)                                    # 回到 count
            preds.append(pred)

    score = np.clip(np.concatenate(preds), 0, None)

    df_pred = df_feat[["d","t","x","y"]].copy()
    df_pred["score"] = score
    return df_pred

def save_lstm_pkl(model, path_pkl, config: dict):
    payload = {
        "state_dict": model.state_dict(),
        "config": config
    }
    torch.save(payload, path_pkl)   # 用 torch 的序列化最穩（副檔名可叫 .pkl）
    