from torch import nn
import torch
import torch.nn.functional as F
# from .Linear import DynamicLinear, MuModuleList
from add.dynamic_attention.Linear import MuModuleList, DynamicLinear



# x : Input feature, tensor size (B, H*W, C).
# H, W: Spatial resolution of the input feature.
# text: tensor size (B, D)


def logsumexp(tensor):
    # tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor, dim=2, keepdim=True)
    outputs = s + (tensor - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, text_dim, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels  # Ci
        self.mlp = MuModuleList([
            DynamicLinear(gate_channels, gate_channels // reduction_ratio, text_dim),
            nn.ReLU(),
            DynamicLinear(gate_channels // reduction_ratio, gate_channels, text_dim)
        ])
        self.pool_types = pool_types

    def forward(self, x, mu):
        # x: [B,HW,Ci]
        # mu : [B,Ct]  []
        B = x.shape[0]  # batchsize 20
        D = x.shape[-1]  # dimension ,Ci 96
        channel_att_sum = None
        for pool_type in self.pool_types:
            pre_pool = x.view(B, -1, D)  # [B,HW,Ci] [20,6400,96]
            if pool_type == 'avg':
                # 一个三维向量，[2,3,2] 可以理解为有两行字符串，每行3个文字，每个文字的字向量维度为3的数字表示。
                # b.mean(dim=1) 的意义是将每行字符串中、每个字的相同维度的字向量求平均。
                avg_pool = torch.mean(pre_pool, dim=1)  # [B,Ci] [20,96]
                channel_att_raw = self.mlp(avg_pool, mu)
            elif pool_type == 'max':
                max_pool = torch.max(pre_pool, dim=1).values
                channel_att_raw = self.mlp(max_pool, mu)
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum)
        scale = scale.view(((scale.shape[0],) + (1,) * (len(x.size()) - 2) + (scale.shape[-1],)))

        return x * scale


class SpatialGate(nn.Module):
    def __init__(self, gate_channels, mu_dim):
        super(SpatialGate, self).__init__()
        self.spatial = DynamicLinear(gate_channels, 1, mu_dim)

    def forward(self, x, mu):
        assert len(x.size()) > 2  # B spatial D

        x_out = self.spatial(x, mu)
        scale = torch.sigmoid(x_out)  # broadcasting
        res = x * scale
        return res


class QueryDynamicAttention(nn.Module):
    def __init__(self, gate_channels, mu_dim, reduction_ratio, pool_types, use_spatial=True, use_channel=True):
        super(QueryDynamicAttention, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, mu_dim, reduction_ratio, pool_types)
        self.SpatialGate = SpatialGate(gate_channels, mu_dim)
        self.use_spatial = use_spatial
        self.use_channel = use_channel

    def forward(self, x, mu):
        # x:[B,H*W,Ci] [20,6400,96]
        # mu : [B,Ct]  [20,768]

        if self.use_channel:
            x = self.ChannelGate(x, mu)
        if len(x.size()) <= 2:
            return x
        if self.use_spatial:
            x = self.SpatialGate(x, mu)
        return x


# if __name__ == '__main__':
#     pass
# qdatt = QueryDynamicAttention(gate_channels=96, mu_dim=768, reduction_ratio=16, pool_types=['avg', 'max'], use_spatial=True, use_channel=True)
# x = torch.randn(20,6400,96)
# text = torch.randn(20,768,20)
# output = qdatt(x, text)
# print(output.shape)
