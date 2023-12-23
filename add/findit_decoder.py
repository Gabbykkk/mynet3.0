import math

import torch
from matplotlib import pyplot as plt
from torch import nn, Tensor
from torchvision import transforms

# decoder的输入：
# [B,8C,H/16,W/16]
# [B,4C,H/16,W/16]
# 输出：
# [B,4C,H/16,W/16]
class SimpleAttention(nn.Module):
    def __init__(
            self,
            channel1,  #
            channel2,  #
            inter_channel  # hidden_stat
    ) -> None:
        super().__init__()

        # instance normalization
        self.ins_q = nn.InstanceNorm2d(inter_channel, affine=True)
        self.ins_w = nn.InstanceNorm2d(inter_channel, affine=True)

        self.vis_project = nn.Conv2d(channel1, inter_channel, 1)
        self.lan_project = nn.Conv2d(channel2, inter_channel, 1)

        # self.self_attn = Attention(inter_channel, 8, proj_drop=0.1)
        self.self_attn = Attention(inter_channel)

    def forward(self, feat1: Tensor, feat2: Tensor):
        B, Ci ,H,W= feat1.size()  # [B,8C,H/16,W/16]
        B, Ct, H,W = feat2.size()  # [B,4C,H/16,W/16]

        vis_p = self.vis_project(feat1)  # [B,8C,H/16,W/16]-->[B,4C,H/16,W/16]
        vis_p = self.ins_q(vis_p)  # [B,4C,H/16,W/16]
        lan_p = self.lan_project(feat2)  # [B,4C,H/16,W/16]-->[B,4C,H/16,W/16]
        T=H*W
        cat_res = torch.cat([vis_p.flatten(2), lan_p.flatten(2)], dim=2)  # [B,4C,H/16*W/16+T]
        x = q = k = v = cat_res.transpose(1, 2)  #[B,H*W+T,4C]
        attn = self.self_attn(x)  # [B,H*W+T,4C]

        out_feats = attn[:, :-T, :].permute(0, 2, 1).reshape(B, -1, H, W)  # [20,96,80,80] [B,4C,H/16,W/16]

        output = self.ins_w(out_feats)  # [B,C,H,W]
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

        # if q.shape[1] == 12800 :
        #     chunks = torch.split(q,split_size_or_sections = 128 , dim=1)
        #     results = []
        #     for chunk in chunks:
        #         result = torch.bmm(chunk,k.transpose(1,2))
        #         results.append(result)
        #     final_result = torch.cat(results,dim = 1)
        #     dist = final_result*self._norm_fact
        # q*k的转置 并*开根号后的dk
        dist = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact  # batch, n, n [20,6420,6420]
        # dist2 = torch.matmul(q, k.transpose(1, 2)) * self._norm_fact  # batch, n, n [20,6420,6420] 与bmm结果一样
        # 归一化获得attention的相关系数
        dist = torch.softmax(dist, dim=-1)  # batch, n, n [20,6420,6420]
        # attention系数和v相乘，获得最终的得分
        att = torch.bmm(dist, v)  # batch,n,c[20,6420,96]
        return att

#
# x = torch.randn(2, 8, 3, 3)
# # image = torch.randn([1,80,80])
# # plt.imshow(transforms.ToPILImage()(image),interpolation="bicubic")
# # transforms.ToPILImage()(image).show()
# l = torch.randn(2, 4, 3, 3)
# model = SimpleAttention(8, 4, 5)
# output = model(x, l)
# print(output.shape)
