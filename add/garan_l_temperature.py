import torch
import torch.nn.functional as F
from torch import nn


class CollectDiffuseAttention(nn.Module):
    ''' CollectDiffuseAttention '''

    def __init__(self, attn_dropout=0.1):
        super().__init__()
        self.dropout_c = nn.Dropout(attn_dropout)
        self.dropout_d = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)
        self.temperature_c = nn.Parameter(torch.tensor(1.0))
        self.temperature_d = nn.Parameter(torch.tensor(1.0))

    def forward(self, q, kc, kd, v, mask=None):
        '''
        q: n*b,1,d_o
        kc: n*b,h*w,d_o
        kd: n*b,h*w,d_o
        v: n*b,h*w,d_o
        '''

        attn_col_logit = torch.bmm(q, kc.transpose(1, 2)) / self.temperature_c.unsqueeze(0).unsqueeze(0)
        attn_col = self.softmax(attn_col_logit)
        attn_col = self.dropout_c(attn_col)
        attn = torch.bmm(attn_col, v)

        attn_dif_logit = torch.bmm(kd, q.transpose(1, 2)) / self.temperature_d.unsqueeze(0).unsqueeze(0)
        attn_dif = F.sigmoid(attn_dif_logit)
        attn_dif = self.dropout_d(attn_dif)
        output = torch.bmm(attn_dif, attn)
        return output, attn_col_logit.squeeze(1)

class GaranAttention(nn.Module):
    ''' GaranAttention module '''

    def __init__(self, d_q, d_v, n_head=1, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_q = d_q
        self.d_v = d_v
        self.d_k = d_v
        self.d_o = d_v
        d_o = d_v

        self.w_qs = nn.Linear(d_q, d_o, bias=True)
        self.w_kc = nn.Conv2d(d_v, d_o, 1, bias=True)
        self.w_kd = nn.Conv2d(d_v, d_o, 1, bias=True)
        self.w_vs = nn.Conv2d(d_v, d_o, 1, bias=True)
        self.attention = CollectDiffuseAttention(attn_dropout=dropout)
        self.layer_norm = nn.BatchNorm2d(d_o)
        self.layer_acti = nn.LeakyReLU(0.1, inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, v, mask=None):
        # q: [B,C]
        # v: [B,C,H,W]

        d_k, d_v, n_head, d_o = self.d_k, self.d_v, self.n_head, self.d_o

        sz_b, c_q = q.size()
        sz_b, c_v, h_v, w_v = v.size()
        residual = v

        q = self.w_qs(q).view(sz_b, n_head, 1, d_o // n_head)  # [B, n_head, 1, d_o // n_head]
        kc = self.w_kc(v).view(sz_b, n_head, d_o // n_head, h_v * w_v)  # [B, n_head, d_o // n_head, H*W]
        kd = self.w_kd(v).view(sz_b, n_head, d_o // n_head, h_v * w_v)  # [B, n_head, d_o // n_head, H*W]
        v = self.w_vs(v).view(sz_b, n_head, d_o // n_head, h_v * w_v)  # [B, n_head, d_o // n_head, H*W]

        q = q.view(-1, 1, d_o // n_head)  # [(n*B) x 1 x (d_o // n_head)]
        kc = kc.permute(0, 1, 3, 2).contiguous().view(-1, h_v * w_v, d_o // n_head)  # [(n*B) x (H*W) x (d_o // n_head)]
        kd = kd.permute(0, 1, 3, 2).contiguous().view(-1, h_v * w_v, d_o // n_head)  # [(n*B) x (H*W) x (d_o // n_head)]
        v = v.permute(0, 1, 3, 2).contiguous().view(-1, h_v * w_v, d_o // n_head)  # [(n*B) x (H*W) x (d_o // n_head)]

        output, attn = self.attention(q, kc, kd, v)
        output = output.view(sz_b, n_head, h_v, w_v, d_o // n_head)
        output = output.permute(0, 1, 4, 3, 2).contiguous().view(sz_b, -1, h_v, w_v)  # [B, lq, (n*dv), H, W]
        attn = attn.view(sz_b, n_head, h_v, w_v)
        attn = attn.mean(1)

        output = output + residual
        output = self.layer_norm(output)
        output = self.layer_acti(output)
        output = self.dropout(output)

        return output
