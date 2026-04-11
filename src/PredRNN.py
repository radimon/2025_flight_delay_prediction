import torch
import torch.nn as nn

class ST_LSTMCell(nn.Module):
    def __init__(self, in_ch, hid_ch, kernel_size=3):
        super().__init__()
        self.hid_ch = hid_ch
        padding = kernel_size // 2

        # input = x + h + m
        self.conv = nn.Conv2d(
            in_ch + hid_ch * 2,
            hid_ch * 7,
            kernel_size,
            padding=padding
        )

    def forward(self, x, h, c, m):
        combined = torch.cat([x, h, m], dim=1)
        gates = self.conv(combined)

        i, f, g, i_p, f_p, g_p, o = torch.chunk(gates, 7, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        c_next = f * c + i * g

        i_p = torch.sigmoid(i_p)
        f_p = torch.sigmoid(f_p)
        g_p = torch.tanh(g_p)
        m_next = f_p * m + i_p * g_p

        o = torch.sigmoid(o)
        cm_cat = torch.cat([c_next, m_next], dim=1)
        h_next = o * torch.tanh(cm_cat[:, :self.hid_ch])

        return h_next, c_next, m_next


class PredRNNRegEmbed(nn.Module):
    def __init__(self, in_ch=2, hid_ch=64, n_wd=7, n_t=48, n_x=2000, n_y=2000):
        super().__init__()
        self.hid_ch = hid_ch
        self.in_ch = in_ch   # ⭐ 這行不能少

        self.cell = ST_LSTMCell(in_ch, hid_ch)

        self.emb_wd = nn.Embedding(n_wd, 2)
        self.emb_t = nn.Embedding(n_t, 8)
        self.emb_x = nn.Embedding(n_x, 16)
        self.emb_y = nn.Embedding(n_y, 16)

        # 2 + 8 + 16 + 16 + 1 = 43
        self.head = nn.Sequential(
            nn.Linear(hid_ch + 43, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x_patch, wd, tid, xid, yid, wknd):
        batch, seq, ch, h, w = x_patch.shape
        device = x_patch.device

        if ch != self.in_ch:
            raise ValueError(f"x_patch channel mismatch: expected {self.in_ch}, got {ch}")

        h_t = torch.zeros(batch, self.hid_ch, h, w, device=device)
        c_t = torch.zeros(batch, self.hid_ch, h, w, device=device)
        m_t = torch.zeros(batch, self.hid_ch, h, w, device=device)

        for t in range(seq):
            h_t, c_t, m_t = self.cell(x_patch[:, t], h_t, c_t, m_t)

        spatial_feat = h_t[:, :, h // 2, w // 2]

        wd = wd.long()
        tid = tid.long()
        xid = xid.long()
        yid = yid.long()

        wknd = wknd.float()
        if wknd.dim() == 1:
            wknd = wknd.unsqueeze(1)
        elif wknd.dim() > 2:
            wknd = wknd.view(wknd.size(0), -1)

        e = torch.cat([
            self.emb_wd(wd),
            self.emb_t(tid),
            self.emb_x(xid),
            self.emb_y(yid),
            wknd
        ], dim=1)

        out = torch.cat([spatial_feat, e], dim=1)
        return self.head(out).squeeze(1)