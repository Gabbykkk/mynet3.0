import torch.nn as nn
import torch
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt

from add.visualize import visualize_collect_diffuse, plot_attention_heatmaps


class CascadeModel(nn.Module):
    def __init__(self,d_q, n_head, dropout):
        super(CascadeModel, self).__init__()
        self.linear = nn.Linear(d_q, d_q)  # 更新线性层的输入维度

    def forward(self, l_feats, l_pooler):
        # l_feats: (B, C, T)
        # l_pooler: (B, C)
        B, C, T = l_feats.size()

        # 将 l_feats 和 l_pooler 融合
        fused_features = torch.cat([l_feats, l_pooler.unsqueeze(2).expand(B, C, T)], dim=2)  # [B,C,T*2]
        fused_features = fused_features.permute(0,2,1)  # (B, T*2,C)
        fused_q = self.linear(fused_features)  # 融合线性变换，得到融合特征 (B, T*2, d_o)
        fused_q = fused_q.permute(0,2,1)
        fused_q = torch.mean(fused_q, dim=2)
        return fused_q


class CollectDiffuseAttention(nn.Module):
    ''' CollectDiffuseAttention '''

    '''
    计算Acol和Adif
    Acol:获取注意力特征
    Adif:用来信息扩散
    
    输入文本特征ft和视觉特征Fv,融合，得到Acol和Adif
    '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout_c = nn.Dropout(attn_dropout)  # collect
        self.dropout_d = nn.Dropout(attn_dropout)  # diffuse
        self.softmax = nn.Softmax(dim=2)
        # self.batch_counter = 0

    def forward(self, l_fusion, kc, kd, v, mask=None):
        """
        l_fusion: n*b,1,d_o       -->B,1,C      文本特征   [20,1,384]
        kc: n*b,h*w,d_o    -->B,h*W,C             [20,400,384]
        kd: n*b,h*w,d_o    -->B,h*w,C             [20,400,384]
        v: n*b,h*w,d_o     -->B,h*w,C    视觉特征   [20,400,384]
        n = num_head,设为1
        """

        attn_col = torch.bmm(l_fusion,
                             kc.transpose(1, 2))  # n*b,1,h*w  注意力特征Acol,attention map  通过将查询向量 l_fusion 与键值对 kc 进行点乘得到的
        attn_col_logit = attn_col / self.temperature  # attn_col_logit:[20,1,400]  attn_col_logit 表示注意力图，是通过将 attn_col 除以温度值进行标准化得到的
        attn_col = self.softmax(
            attn_col_logit)  # 使用 softmax 函数对 attn_col_logit 进行操作，将其转换为概率分布。softmax 函数将每个元素的值压缩到 0 到 1 之间，并且所有元素的总和等于 1。这样做可以使得每个位置的注意力权重相对于其他位置的权重更加明确和显著。
        attn_col = self.dropout_c(
            attn_col)  # attn_col : [20,1,400]  使用 dropout 操作对 attn_col 进行随机失活。dropout 是一种正则化技术，它在训练过程中随机地将部分元素置为零，从而减少模型的过拟合。通过对 attn_col 进行 dropout，可以减少注意力权重之间的依赖关系，增加模型的鲁棒性。
        attn = torch.bmm(attn_col, v)  # n*b,1,d_o    Fv加权求和得fatt，即attn  attn:[20,1,384]

        attn_dif = torch.bmm(kd, l_fusion.transpose(1,
                                                 2))  # n*b,h*w,1  注意力特征Adif,attention map  过将键值对 kd 与查询向量 l_fusion 的转置进行点乘得到的
        attn_dif_logit = attn_dif / self.temperature  # attn_dif_logit : [20,400,1]  attn_dif_logit 也表示注意力图，是通过将 attn_dif 除以温度值进行标准化得到的
        attn_dif = F.sigmoid(attn_dif_logit)
        attn_dif = self.dropout_d(attn_dif)  # attn_dif:[20,400,1]
        output = torch.bmm(attn_dif, attn)  # 基于Adif,将fatt-->Fatt(与Fv相同纬度的特征矩阵)。Fatt即output  output:[20,400,384]

        # self.batch_counter += 1
        # if self.batch_counter == 2120:
        #     self.batch_counter = 1

        # plot_attention_heatmaps(attn_col,attn_dif)

        # return output, attn_col_logit.squeeze(1),attn_col,attn_dif
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

        self.attention = CollectDiffuseAttention(temperature=np.power(d_o // n_head, 0.5))
        self.layer_norm = nn.BatchNorm2d(d_o)
        self.layer_acti = nn.LeakyReLU(0.1, inplace=True)

        # nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)
        self.cascade = CascadeModel(d_q, 1, 0.1)

    def forward(self, l_feats, l_pooler, v, mask=None):
        # v :[B,C,H,W][20,384,20,20]
        l_fusion = self.cascade(l_feats, l_pooler)

        d_k, d_v, n_head, d_o = self.d_k, self.d_v, self.n_head, self.d_o

        sz_b, c_q = l_fusion.size()
        sz_b, c_v, h_v, w_v = v.size()
        # print(v.size())
        residual = v
        l_fusion = l_fusion.to(self.w_qs.weight.dtype)
        l_fusion = self.w_qs(l_fusion)  # l_pooler时:[20,384]
        kc = self.w_kc(v).view(sz_b, n_head, d_o // n_head, h_v * w_v)  # l_pooler时:[20,1,384,400]
        kd = self.w_kd(v).view(sz_b, n_head, d_o // n_head, h_v * w_v)  # l_pooler时:[20,1,384,400]
        v = self.w_vs(v).view(sz_b, n_head, d_o // n_head, h_v * w_v)  # l_pooler时:[20,1,384,400]
        l_fusion = l_fusion.view(sz_b, n_head, 1, d_o // n_head)  # l_pooler时:[20,1,1,384]
        # v=v.view(sz_b,h_v*w_v,n_head,c_v//n_head)

        l_fusion = l_fusion.view(-1, 1, d_o // n_head)  # (n*b) x lq x dk    # l_pooler时:[20,1,384]
        kc = kc.permute(0, 1, 3, 2).contiguous().view(-1, h_v * w_v, d_o // n_head)  # (n*b) x lk x dk
        kd = kd.permute(0, 1, 3, 2).contiguous().view(-1, h_v * w_v, d_o // n_head)  # (n*b) x lk x dk
        v = v.permute(0, 1, 3, 2).contiguous().view(-1, h_v * w_v, d_o // n_head)  # (n*b) x lv x dv

        # output, attn, attn_col, attn_dif = self.attention(l_fusion, kc, kd, v)
        output, attn = self.attention(l_fusion, kc, kd, v)

        # n * b, h * w, d_o
        output = output.view(sz_b, n_head, h_v, w_v, d_o // n_head)
        output = output.permute(0, 1, 4, 3, 2).contiguous().view(sz_b, -1, h_v, w_v)  # b x lq x (n*dv)
        attn = attn.view(sz_b, n_head, h_v, w_v)
        attn = attn.mean(1)  # attn:[20,20,20]
        # residual connect
        output = output + residual
        output = self.layer_norm(output)
        output = self.layer_acti(output)  # output:[20，384，20，20]

        # output = self.dropout(self.fc(output))

        # return output, attn_col, attn_dif
        return output
