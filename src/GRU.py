import torch
import torch.nn as nn
from .LSTM import SeqDatasetEmbed  # 如果相對匯入有問題，可改成 from src.LSTM import SeqDatasetEmbed


class GRURegEmbed(nn.Module):
    def __init__(
        self,
        hidden=64,
        layers=1,
        n_weekday=7,
        n_t=48,
        n_x=2000,
        n_y=2000,
        emb_wd=2,
        emb_t=8,
        emb_x=16,
        emb_y=16
    ):
        super().__init__()

        # 序列模型：輸入是過去 seq_len 個 count，shape 通常為 (batch, seq_len, 1)
        self.gru = nn.GRU(
            input_size=1,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=0.1 if layers > 1 else 0.0
        )

        # 類別特徵 embedding
        self.emb_weekday = nn.Embedding(n_weekday, emb_wd)
        self.emb_t = nn.Embedding(n_t, emb_t)
        self.emb_x = nn.Embedding(n_x, emb_x)
        self.emb_y = nn.Embedding(n_y, emb_y)

        # 靜態特徵維度：
        # weekday embedding + t embedding + x embedding + y embedding + is_weekend(1維)
        static_dim = emb_wd + emb_t + emb_x + emb_y + 1

        # 預測 head
        self.head = nn.Sequential(
            nn.Linear(hidden + static_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x_seq, weekday, t_id, x_id, y_id, is_weekend):
        """
        x_seq:      (batch, seq_len, 1)
        weekday:    (batch,)
        t_id:       (batch,)
        x_id:       (batch,)
        y_id:       (batch,)
        is_weekend: (batch,) 或 (batch, 1)
        """

        # 確保型別正確
        x_seq = x_seq.float()
        weekday = weekday.long()
        t_id = t_id.long()
        x_id = x_id.long()
        y_id = y_id.long()
        is_weekend = is_weekend.float()

        # 如果 is_weekend 是 (batch,) -> 轉成 (batch, 1)
        if is_weekend.dim() == 1:
            is_weekend = is_weekend.unsqueeze(1)

        # GRU 輸出
        out, _ = self.gru(x_seq)          # out: (batch, seq_len, hidden)
        h = out[:, -1, :]                 # 取最後一個時間步 -> (batch, hidden)

        # embedding 後拼接靜態特徵
        e = torch.cat([
            self.emb_weekday(weekday),    # (batch, emb_wd)
            self.emb_t(t_id),             # (batch, emb_t)
            self.emb_x(x_id),             # (batch, emb_x)
            self.emb_y(y_id),             # (batch, emb_y)
            is_weekend,                   # (batch, 1)
        ], dim=1)

        # 合併序列表示與靜態特徵
        z = torch.cat([h, e], dim=1)

        # 輸出 shape: (batch,)
        return self.head(z).squeeze(1)