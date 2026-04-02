import torch
import torch.nn as nn
from LSTM import SeqDatasetEmbed # 複用你的 Dataset

class GRURegEmbed(nn.Module):
    def __init__(self, hidden=64, layers=1, n_weekday=7, n_t=48, n_x=2000, n_y=2000,
                 emb_wd=2, emb_t=8, emb_x=16, emb_y=16):
        super().__init__()
        # 將 LSTM 換成 GRU
        self.gru = nn.GRU(input_size=1, hidden_size=hidden, num_layers=layers,
                          batch_first=True, dropout=0.1 if layers > 1 else 0.0)
        
        self.emb_weekday = nn.Embedding(n_weekday, emb_wd)
        self.emb_t = nn.Embedding(n_t, emb_t)
        self.emb_x = nn.Embedding(n_x, emb_x)
        self.emb_y = nn.Embedding(n_y, emb_y)

        static_dim = emb_wd + emb_t + emb_x + emb_y + 1
        self.head = nn.Sequential(
            nn.Linear(hidden + static_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x_seq, weekday, t_id, x_id, y_id, is_weekend):
        out, _ = self.gru(x_seq)
        h = out[:, -1, :]
        e = torch.cat([
            self.emb_weekday(weekday),
            self.emb_t(tid),
            self.emb_x(xid),
            self.emb_y(yid),
            is_weekend,
        ], dim=1)
        z = torch.cat([h, e], dim=1)
        return self.head(z).squeeze(1)