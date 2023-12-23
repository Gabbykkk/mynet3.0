import torch
import torch.nn as nn

"""
input:

output: FrM:[2C,H,W]

FrM : output
FcM : lan_feat进行一系列计算，再与vis_feat点乘，得到FcM,命名为lan_feat_vis
    对lan_feat全局平均池化，(去除concate),得到finM
    finM --公式1--> tcM
    tcM 通道乘法 FM --> FcM
FsM : vis_feat进行一系列运算，再与vis_feat点乘，得到FsM,命名为vis_feat_vis


MC替换LG
两个输入，分别是:
    swin transformer stage的输出，即为vis_feat:[B,H*W,Ci]
    pwam模块的输出，即为mm_feat :[B,H*W,C]
    C=Ci
"""


class MutualComplementarity(nn.Module):
    def __init__(self, vis_channel, mm_channel):
        super(MutualComplementarity, self).__init__()

        self.vis_channel = vis_channel
        self.mm_channel = mm_channel

        # se attention
        self.se_vis = SELayer(vis_channel)
        self.se_mm = SELayer(mm_channel)

    def forward(self, vis_feat, mm_feat, H, W):
        vis_feat_vis = self.se_vis(vis_feat, vis_feat,H,W)  # [B,C,H,W]
        mm_feat_vis = self.se_mm(mm_feat, vis_feat,H,W)  # [B,C,H,W]
        # output = torch.cat((vis_feat_vis, mm_feat_vis),dim=1)  # [B,2C,H,W]
        output = (vis_feat_vis + mm_feat_vis).flatten(2).permute(0,2,1)  # [B,H*W,C]
        return output


class SELayer(nn.Module):
    def __init__(self, in_channel, ratio=16):
        '''
        channel-wise attention 关注每一个通道的比重
        实现过程：
        1、对输入进来的特征层进行全局平均池化（在H，W上做全局平均池化，完成之后高和宽都是1）
        2、然后进行两次全连接层：第一次全连接神经元个数较少，第二次全连接神经元个数和输入特征层相同
        3、完成两次全连接层后，使用一次sigmoid将值固定到0-1之间，此时就获得了输入特征层每一个通道的权值（0-1）
        4、获得权值后，将权值乘上原来的输入特征层即可
        Args:
            in_channel:输入特征的通道数
            ratio: 缩放的比例，在第一个神经元个数较少的全连接层会对输入进来的特征长条进行一个缩放
        '''
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.fc = nn.Sequential(
            nn.Linear(in_channel, in_channel // ratio, bias=False),
            # 第一次全连接，输入进来的神经元个数是通道数，输出出来的神经元个数是通道数除以缩放的比例，不使用偏置
            nn.ReLU(inplace=True),  # 激活函数
            nn.Linear(in_channel // ratio, in_channel, bias=False),  # 第二次全连接
            nn.Sigmoid()  # 激活函数,把值固定到0-1之间,获得每一个通道的权值，每一个神经元对应每一个通道
        )

    def forward(self, feat1, feat2,H,W):
        B, L, C = feat1.size()
        feat1 = feat1.reshape(B,C,H,W)
        feat2 = feat2.reshape(B,C,H,W)
        avg = self.avg_pool(feat1)  # [b,c,h,w] -->[B,C,1,1]
        avg = avg.view([B, C])  # [20,128]  [B,C]
        # b,c --> b,c//ratio --> b,c
        fc = self.fc(avg)  # [20,128]  [B,C]
        fc = fc.view([B, C, 1, 1])  # [20,128,1,1]  [B,C,1,1]
        # expand_as(tensor)将张量扩展成参数tensor的大小
        output = fc.expand_as(feat1)  # [20,128,80,80]
        output = feat2 * output  # [20,128,80,80]两次全连接的结果乘上最开始的输入
        return output


# x = torch.randn(2,16,5)
# y = torch.randn(2,16,5)
# mc = MutualComplementarity(5,5)
# output = mc(x,y,4,4)
# print(output.shape)