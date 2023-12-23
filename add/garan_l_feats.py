import torch.nn as nn
import torch
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import numpy as np


# class DynamicLinear(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(DynamicLinear, self).__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
#         self.bias = nn.Parameter(torch.Tensor(output_dim))
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         nn.init.xavier_uniform_(self.weight)
#         nn.init.zeros_(self.bias)
#
#     def forward(self, x):
#         batch_size, _, _ = x.size()
#         weight = self.weight.unsqueeze(0).unsqueeze(1).expand(batch_size, -1, -1, -1)
#         bias = self.bias.unsqueeze(0).expand(batch_size, -1, -1)
#         out = torch.matmul(x.unsqueeze(2), weight).squeeze(2) + bias
#         return out






class CollectDiffuseAttention(nn.Module):
    ''' CollectDiffuseAttention '''

    '''
    计算Acol和Adif
    Acol:获取注意力特征
    Adif:用来信息扩散

    输入文本特征ft和视觉特征Fv,融合，得到Acol和Adif
    '''

    def __init__(self, d_o, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout_c = nn.Dropout(attn_dropout)  # collect
        self.dropout_d = nn.Dropout(attn_dropout)  # diffuse
        self.softmax = nn.Softmax(dim=2)

        # self.dynamic_lin_c = DynamicLinear(d_o, d_o)
        # self.dynamic_lin_d = DynamicLinear(d_o, d_o)

    def forward(self, q, kc, kd, v, mask=None):
        """
        q: n*b,1,d_o       -->B,1,C      文本特征  B，1，384
        kc: n*b,h*w,d_o    -->B,h*W,C  B，400，384
        kd: n*b,h*w,d_o    -->B,h*w,C  B，400，384
        v: n*b,h*w,d_o     -->B,h*w,C    视觉特征  B，400，384
        n = num_head,设为1
        """

        """
        在这里，q, kc, kd的维度都是n*b,h*w,d_o。现在我们先通过动态线性层计算新的kc和kd。
        """
        # kc = self.dynamic_lin_c(kc)
        # kd = self.dynamic_lin_d(kd)

        attn_col = torch.bmm(q, kc.transpose(1, 2))  # n*b,1,h*w
        attn_col_logit = attn_col / self.temperature
        attn_col = self.softmax(attn_col_logit)
        attn_col = self.dropout_c(attn_col)
        attn = torch.bmm(attn_col, v)  # n*b,1,d_o    Fv加权求和得fatt，即attn

        attn_dif = torch.bmm(kd, q.transpose(1, 2))  # n*b,h*w,1
        attn_dif_logit = attn_dif / self.temperature
        attn_dif = F.sigmoid(attn_dif_logit)
        attn_dif = self.dropout_d(attn_dif)
        output = torch.bmm(attn_dif, attn)  # 基于Adif,将fatt-->Fatt(与Fv相同纬度的特征矩阵)。Fatt即output
        return output, attn_col_logit.squeeze(1)


class GaranAttention(nn.Module):
    ''' GaranAttention module '''
    # num_head=1 单头
    """
    d_q:language_feature_channel 768
    d_v:visual_feature_channel
    """

    def __init__(self, d_q, d_v, n_head=1, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_q = d_q  # 768
        self.d_v = d_v
        self.d_k = d_v
        self.d_o = d_v
        d_o = d_v

        self.w_qs = nn.Linear(d_q, d_o, 1)  # (768,384,True)
        self.w_kc = nn.Conv2d(d_v, d_o, 1)
        self.w_kd = nn.Conv2d(d_v, d_o, 1)
        self.w_vs = nn.Conv2d(d_v, d_o, 1)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_q + d_o)))
        nn.init.normal_(self.w_kc.weight, mean=0, std=np.sqrt(2.0 / (d_v + d_o // n_head)))
        nn.init.normal_(self.w_kd.weight, mean=0, std=np.sqrt(2.0 / (d_v + d_o // n_head)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_v + d_o // n_head)))

        self.attention = CollectDiffuseAttention(d_o, temperature=np.power(d_o // n_head, 0.5))
        self.layer_norm = nn.BatchNorm2d(d_o)
        self.layer_acti = nn.LeakyReLU(0.1, inplace=True)

        # nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, v, mask=None):
        # q
        # v :[B,C,H,W][20,384,20,20]

        d_k, d_v, n_head, d_o = self.d_k, self.d_v, self.n_head, self.d_o

        sz_b, c_q = q.size()
        sz_b, c_v, h_v, w_v = v.size()
        # print(v.size())
        residual = v

        q = self.w_qs(q)
        kc = self.w_kc(v).view(sz_b, n_head, d_o // n_head, h_v * w_v)
        kd = self.w_kd(v).view(sz_b, n_head, d_o // n_head, h_v * w_v)
        v = self.w_vs(v).view(sz_b, n_head, d_o // n_head, h_v * w_v)
        q = q.view(sz_b, n_head, 1, d_o // n_head)
        # v=v.view(sz_b,h_v*w_v,n_head,c_v//n_head)

        q = q.view(-1, 1, d_o // n_head)  # (n*b) x lq x dk    # l_pooler时:[20,1,384]
        kc = kc.permute(0, 1, 3, 2).contiguous().view(-1, h_v * w_v, d_o // n_head)  # (n*b) x lk x dk
        kd = kd.permute(0, 1, 3, 2).contiguous().view(-1, h_v * w_v, d_o // n_head)  # (n*b) x lk x dk
        v = v.permute(0, 1, 3, 2).contiguous().view(-1, h_v * w_v, d_o // n_head)  # (n*b) x lv x dv

        output, attn = self.attention(q, kc, kd, v)
        # n * b, h * w, d_o
        output = output.view(sz_b, n_head, h_v, w_v, d_o // n_head)
        output = output.permute(0, 1, 4, 3, 2).contiguous().view(sz_b, -1, h_v, w_v)  # b x lq x (n*dv)
        attn = attn.view(sz_b, n_head, h_v, w_v)
        attn = attn.mean(1)
        # residual connect
        output = output + residual
        output = self.layer_norm(output)
        output = self.layer_acti(output)

        # output = self.dropout(self.fc(output))

        return output
