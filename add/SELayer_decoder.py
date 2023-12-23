import torch
import torch.nn as nn

# 替换decoder的concate
# 输入 [B,C,H,W]
# 输入为pwam的输出，x_residual : [B,H*W,C]
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

    def forward(self, x):
        B,C,H,W = x.size()
        avg = self.avg_pool(x)  # [b,c,h,w] -->[B,C,1,1]
        avg = avg.view([B, C])  # [20,96]  [B,C]
        # b,c --> b,c//ratio --> b,c
        fc = self.fc(avg)  # [20,96]  [B,C]
        fc = fc.view([B, C, 1, 1])  # [20,96,1,1]  [B,C,1,1]
        # expand_as(tensor)将张量扩展成参数tensor的大小
        output = fc.expand_as(x)  # [20,128,80,80]
        output = x * output  # [20,128,80,80]两次全连接的结果乘上最开始的输入
        return output