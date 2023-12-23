import torch.nn as nn
import torch
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt

from add.visualize import visualize_collect_diffuse, plot_attention_heatmaps
# 消融实验,仅用Acol

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
        self.softmax = nn.Softmax(dim=2)
        # self.batch_counter = 0

    def forward(self, q, kc, v, mask=None):
        """
        q: n*b,1,d_o       -->B,1,C      文本特征   [20,1,384]
        kc: n*b,h*w,d_o    -->B,h*W,C             [20,400,384]
        kd: n*b,h*w,d_o    -->B,h*w,C             [20,400,384]
        v: n*b,h*w,d_o     -->B,h*w,C    视觉特征   [20,400,384]
        n = num_head,设为1
        """

        attn_col = torch.bmm(q, kc.transpose(1, 2))  # n*b,1,h*w  注意力特征Acol,attention map  通过将查询向量 q 与键值对 kc 进行点乘得到的
        attn_col_logit = attn_col / self.temperature  # attn_col_logit:[20,1,400]  attn_col_logit 表示注意力图，是通过将 attn_col 除以温度值进行标准化得到的
        attn_col = self.softmax(attn_col_logit)  # 使用 softmax 函数对 attn_col_logit 进行操作，将其转换为概率分布。softmax 函数将每个元素的值压缩到 0 到 1 之间，并且所有元素的总和等于 1。这样做可以使得每个位置的注意力权重相对于其他位置的权重更加明确和显著。
        attn_col = self.dropout_c(attn_col)  # attn_col : [20,1,400]  使用 dropout 操作对 attn_col 进行随机失活。dropout 是一种正则化技术，它在训练过程中随机地将部分元素置为零，从而减少模型的过拟合。通过对 attn_col 进行 dropout，可以减少注意力权重之间的依赖关系，增加模型的鲁棒性。
        attn = torch.bmm(attn_col, v)  # n*b,1,d_o    Fv加权求和得fatt，即attn  attn:[20,1,384]

        output = attn

        return output


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

    def forward(self, q, v, mask=None):
        # q=l_pooler [B,C] [20,768]
        # v :[B,C,H,W][20,384,20,20]

        d_k, d_v, n_head, d_o = self.d_k, self.d_v, self.n_head, self.d_o

        sz_b, c_q = q.size()
        sz_b, c_v, h_v, w_v = v.size()
        # print(v.size())
        residual = v

        q = self.w_qs(q)  # l_pooler时:[20,384]
        kc = self.w_kc(v).view(sz_b, n_head, d_o // n_head, h_v * w_v)  # l_pooler时:[20,1,384,400]
        kd = self.w_kd(v).view(sz_b, n_head, d_o // n_head, h_v * w_v)  # l_pooler时:[20,1,384,400]
        v = self.w_vs(v).view(sz_b, n_head, d_o // n_head, h_v * w_v)  # l_pooler时:[20,1,384,400]
        q = q.view(sz_b, n_head, 1, d_o // n_head)  # l_pooler时:[20,1,1,384]
        # v=v.view(sz_b,h_v*w_v,n_head,c_v//n_head)

        q = q.view(-1, 1, d_o // n_head)  # (n*b) x lq x dk    # l_pooler时:[20,1,384]
        kc = kc.permute(0, 1, 3, 2).contiguous().view(-1, h_v * w_v, d_o // n_head)  # (n*b) x lk x dk
        kd = kd.permute(0, 1, 3, 2).contiguous().view(-1, h_v * w_v, d_o // n_head)  # (n*b) x lk x dk
        v = v.permute(0, 1, 3, 2).contiguous().view(-1, h_v * w_v, d_o // n_head)  # (n*b) x lv x dv

        # output, attn, attn_col, attn_dif = self.attention(q, kc, kd, v)
        output = self.attention(q, kc, kd, v)   # [B,1,C]

        output = torch.squeeze(output, dim=1)
        # 使用 torch.unsqueeze() 在第二维度上插入一个维度
        output = torch.unsqueeze(output, dim=1)

        # 使用 expand() 方法扩展第二维度，使其与原始输出的尺度相匹配
        output = output.expand(sz_b, h_v*w_v, c_v)


        # n * b, h * w, d_o
        output = output.view(sz_b, n_head, h_v, w_v, d_o // n_head)
        output = output.permute(0, 1, 4, 3, 2).contiguous().view(sz_b, -1, h_v, w_v)  # b x lq x (n*dv)

        # residual connect
        output = output + residual
        output = self.layer_norm(output)
        output = self.layer_acti(output)  # output:[20，384，20，20]

        return output
