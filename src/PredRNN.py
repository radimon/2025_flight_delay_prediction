import torch
import torch.nn as nn

class ST_LSTMCell(nn.Module):
    def __init__(self, in_ch, hid_ch, kernel_size=3):
        super().__init__()
        self.hid_ch = hid_ch
        padding = kernel_size // 2
        # PredRNN 的核心：同時處理時間狀態 C 與時空狀態 M
        self.conv = nn.Conv2d(in_ch + hid_ch * 2, hid_ch * 7, kernel_size, padding=padding)

    def forward(self, x, h, c, m):
        # x: (B, in_ch, H, W)
        # h, c, m: (B, hid_ch, H, W)
        combined = torch.cat([x, h, m], dim=1)
        gates = self.conv(combined)
        
        # 拆分為 7 個 Gate: i, f, g (時間) / i', f', g' (時空) / o (輸出)
        i, f, g, i_p, f_p, g_p, o = torch.chunk(gates, 7, dim=1)

        i, f, g = torch.sigmoid(i), torch.sigmoid(f), torch.tanh(g)
        c_next = f * c + i * g

        i_p, f_p, g_p = torch.sigmoid(i_p), torch.sigmoid(f_p), torch.tanh(g_p)
        m_next = f_p * m + i_p * g_p

        o = torch.sigmoid(o)
        h_next = o * torch.tanh(torch.cat([c_next, m_next], dim=1)[:, :self.hid_ch]) # 簡化拼接
        
        return h_next, c_next, m_next

class PredRNNRegEmbed(nn.Module):
    def __init__(self, in_ch=1, hid_ch=64, n_wd=7, n_t=48, n_x=2000, n_y=2000):
        super().__init__()
        self.hid_ch = hid_ch
        self.cell = ST_LSTMCell(in_ch, hid_ch)
        
        # 嵌入層 (與你的 ConvLSTM 一致)
        self.emb_wd = nn.Embedding(n_wd, 2)
        self.emb_t = nn.Embedding(n_t, 8)
        self.emb_x = nn.Embedding(n_x, 16)
        self.emb_y = nn.Embedding(n_y, 16)

        self.head = nn.Sequential(
            nn.Linear(hid_ch + 43, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x_patch, wd, tid, xid, yid, wknd):
        # x_patch: (B, Seq, 1, H, W)
        batch, seq, _, h, w = x_patch.shape
        device = x_patch.device
        
        # 初始化狀態
        h_t = torch.zeros(batch, self.hid_ch, h, w).to(device)
        c_t = torch.zeros(batch, self.hid_ch, h, w).to(device)
        m_t = torch.zeros(batch, self.hid_ch, h, w).to(device)

        for t in range(seq):
            h_t, c_t, m_t = self.cell(x_patch[:, t], h_t, c_t, m_t)
        
        # 取最後一個時間步的空間特徵並攤平
        spatial_feat = h_t[:, :, h//2, w//2] # 取中心點特徵
        
        e = torch.cat([
            self.emb_wd(wd), self.emb_t(tid),
            self.emb_x(xid), self.emb_y(yid),
            wknd.unsqueeze(1)
        ], dim=1)
        
        return self.head(torch.cat([spatial_feat, e], dim=1)).squeeze(1)