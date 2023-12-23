import math

import torch
from matplotlib import pyplot as plt
from torch import nn, Tensor
from torchvision import transforms


class SimpleAttention(nn.Module):
    def __init__(
            self,
            visual_channel,  # input visual features' channel
            language_channel,  # input language features,
            inter_channel
    ) -> None:
        super().__init__()

        # instance normalization
        self.ins_q = nn.InstanceNorm2d(inter_channel, affine=True)
        self.ins_w = nn.InstanceNorm2d(visual_channel, affine=True)

        self.vis_project = nn.Conv2d(visual_channel, inter_channel, 1)
        self.lan_project = nn.Conv1d(language_channel, inter_channel, 1)

        # self.self_attn = Attention(inter_channel, 8, proj_drop=0.1)
        self.self_attn = Attention(inter_channel)

        self.output_layer = nn.Conv2d(inter_channel, visual_channel, 1)

    def forward(self, vis_feat: Tensor, lan_feat: Tensor, H, W):
        # vis_feat:[B,H*W,Ci]
        # lan_feat:[B,Ct,T]
        B, L, Ci = vis_feat.size()
        B, Ct, T = lan_feat.size()
        vis_feat = vis_feat.permute(0,2,1).reshape(B, Ci, H, W)

        vis_p = self.vis_project(vis_feat)  # [B,Ci,H,W]
        vis_p = self.ins_q(vis_p)  # [B,Ci,H,W]
        lan_p = self.lan_project(lan_feat)  # [B,C,T]

        cat_res = torch.cat([vis_p.flatten(2), lan_p], dim=2)  # [B,C,H*W+T]
        x = q = k = v = cat_res.transpose(1, 2)  #[B,H*W+T,C]
        attn = self.self_attn(x)  # [B,H*W+T,C]

        out_feats = attn[:, :-T, :].permute(0, 2, 1).reshape(B, -1, H, W)  # [20,96,80,80] [B,C,H,W]

        output = self.output_layer(out_feats)  # [20,96,80,80] [B,C,H,W]
        output = self.ins_w(output)  # [B,C,H,W]
        output = output.flatten(2).permute(0,2,1)  # [B,H*W,C]

        return output


class Attention(nn.Module):
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.dim_q = dim
        self.dim_k = dim
        self.dim_v = dim

        # 定义线性变换函数
        self.linear_q = nn.Linear(dim, dim, bias=False)
        self.linear_k = nn.Linear(dim, dim, bias=False)
        self.linear_v = nn.Linear(dim, dim, bias=False)
        self._norm_fact = 1 / math.sqrt(dim)  # {float} 0.08838834764831843

    def forward(self, x):
        # x: batch, n, dim_q
        # x:[B,H*W+T,C]
        # 根据文本获得相应的维度
        batch, n, dim_q = x.shape  # batch = 20,n=6420,dim_q = 128
        assert dim_q == self.dim_q

        q = self.linear_q(x)  # batch, n, dim_k [20,6420,128]
        k = self.linear_k(x)  # batch, n, dim_k
        v = self.linear_v(x)  # batch, n, dim_v
        # q*k的转置 并*开根号后的dk
        dist = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact  # batch, n, n [20,6420,6420]
        # dist2 = torch.matmul(q, k.transpose(1, 2)) * self._norm_fact  # batch, n, n [20,6420,6420] 与bmm结果一样
        # 归一化获得attention的相关系数
        dist = torch.softmax(dist, dim=-1)  # batch, n, n [20,6420,6420]
        # attention系数和v相乘，获得最终的得分
        att = torch.bmm(dist, v)  # batch,n,c[20,6420,96]
        return att

#
# x = torch.randn([20, 6400, 96])
# # image = torch.randn([1,80,80])
# # plt.imshow(transforms.ToPILImage()(image),interpolation="bicubic")
# # transforms.ToPILImage()(image).show()
# l = torch.randn([20, 768, 20])
# model = SimpleAttention(96, 768, 96)
# output = model(x, l, 80, 80)
