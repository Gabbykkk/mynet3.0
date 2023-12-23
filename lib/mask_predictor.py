import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict

# concate换成
# from add.SELayer_decoder import SELayer
# from add.findit_decoder import SimpleAttention
from add.HLF import hlf
from add.garan_l_pooler import GaranAttention


class SimpleDecoding(nn.Module):
    def __init__(self, c4_dims, l_dim, factor=2):  # c4_dims = 8*embed_dim=8C ,small_embed_dim =96   l_pooler_dim=768
        super(SimpleDecoding, self).__init__()

        hidden_size = c4_dims // factor  # 4C
        c4_size = c4_dims  # 8C
        c3_size = c4_dims // (factor ** 1)  # 4C  F3_size_dim_channel
        c2_size = c4_dims // (factor ** 2)  # 2C  F2_size_dim_channel
        c1_size = c4_dims // (factor ** 3)  # C   F1_size_dim_channel

        self.conv1_4 = nn.Conv2d(c4_size + c3_size, hidden_size, 3, padding=1, bias=False)  # channels:8C+4C-->4C
        self.bn1_4 = nn.BatchNorm2d(hidden_size)
        self.relu1_4 = nn.ReLU()

        self.conv2_4 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1, bias=False)  # channels:4C-->4C
        self.bn2_4 = nn.BatchNorm2d(hidden_size)
        self.relu2_4 = nn.ReLU()

        self.conv1_3 = nn.Conv2d(hidden_size + c2_size, hidden_size, 3, padding=1, bias=False)  # channels:4C+2C-->4C
        self.bn1_3 = nn.BatchNorm2d(hidden_size)
        self.relu1_3 = nn.ReLU()

        self.conv2_3 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1, bias=False)  # channels:4C-->4C
        self.bn2_3 = nn.BatchNorm2d(hidden_size)
        self.relu2_3 = nn.ReLU()

        self.conv1_2 = nn.Conv2d(hidden_size + c1_size, hidden_size, 3, padding=1, bias=False)  # channels:4C+C-->4C
        self.bn1_2 = nn.BatchNorm2d(hidden_size)
        self.relu1_2 = nn.ReLU()

        self.conv2_2 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1, bias=False)  # channels:4C-->4C
        self.bn2_2 = nn.BatchNorm2d(hidden_size)
        self.relu2_2 = nn.ReLU()

        self.conv1_1 = nn.Conv2d(hidden_size, 2, 1)  # channels:8C-->2
        self.conv1 = nn.Conv2d(c1_size, 2, 1)  # channels:8C-->2

        # self.findit4 = SimpleAttention(c4_size,c3_size,hidden_size)
        # self.findit3 = SimpleAttention(hidden_size,c2_size,hidden_size)
        # self.findit2 = SimpleAttention(hidden_size,c1_size,hidden_size)

        # self.se4 = SELayer(c3_size)  # 4C
        # self.se3 = SELayer(c2_size)  # 2C
        # self.se2 = SELayer(c1_size)  # C

        self.garan4 = GaranAttention(l_dim, c3_size)  # 4C  传入的参数是语言特征channel768和视觉特征（即pwam输出的融合特征F1，2，3）channle
        self.garan3 = GaranAttention(l_dim, c2_size)  # 2C
        self.garan2 = GaranAttention(l_dim, c1_size)  # C


    def forward(self, l_pooler, x_c4, x_c3, x_c2, x_c1):
        """
        small：
        x_c1={Tensor:(20,96,80,80)} [B,C,H/4,W/4] F1
        x_c2={Tensor:(20,192,40,40)} [B,2C,H/8,W/8] F2
        x_c3={Tensor:(20,384,20,20)} [B,4C,H/16,W/16] F3
        x_c4={Tensor:(20,768,10,10)} [B,8C,H/32,W/32] V4=Y4=F4
        l_pooler:[20,768]
        l_feats:[20,768,20]  [B,dim_C,N_l]
        l_fusion:
        """

        # fuse Y4 and Y3
        # fuse Y4 and F3 -->Y3
        if x_c4.size(-2) < x_c3.size(-2) or x_c4.size(-1) < x_c3.size(-1):
            x_c4 = F.interpolate(input=x_c4, size=(x_c3.size(-2), x_c3.size(-1)), mode='bilinear',
                                 align_corners=True)  # x_c4 [20,8C,H/16,W/16]
        l_garan = l_pooler  # [B,C] [20,768]
        x_c3_garan = x_c3  # [B,C,H,W] [20,384,20,20]
        x3 = self.garan4(l_garan, x_c3_garan)
        x = torch.cat([x_c4, x3], dim=1)  # concat(dim=1),dim1的数值相加 ,x:[B,8C+4C,H/16,W/16]
        x = self.conv1_4(x)  # [B,8C+4C,H/16,W/16]-->[B,4C,H/16,W/16]
        x = self.bn1_4(x)
        x = self.relu1_4(x)

        x = self.conv2_4(x)  # [B,4C,H/16,W/16]-->[B,4C,H/16,W/16]
        x = self.bn2_4(x)
        x = self.relu2_4(x)  # Y3:[B,4C,H/16,W/16]

        # fuse top-down features and Y2 features
        # fuse Y3 and F2 --> Y2
        if x.size(-2) < x_c2.size(-2) or x.size(-1) < x_c2.size(-1):
            x = F.interpolate(input=x, size=(x_c2.size(-2), x_c2.size(-1)), mode='bilinear',
                              align_corners=True)  # [B,4C,H/16,W/16]-->[B,4C,H/8,W/8]
        x_c2_garan = x_c2
        x2 = self.garan3(l_garan, x_c2_garan)
        x = torch.cat([x, x2], dim=1)  # -->[B,4C+2C,H/8,W/8]
        x = self.conv1_3(x)  # [B,4C+2C,H/8,W/8]-->[B,4C,H/8,W/8]
        x = self.bn1_3(x)
        x = self.relu1_3(x)

        x = self.conv2_3(x)  # [B,4C,H/8,W/8]-->[B,4C,H/8,W/8]
        x = self.bn2_3(x)
        x = self.relu2_3(x)  # Y2:[B,4C,H/8,W/8]

        # fuse top-down features and Y1 features
        # fuse Y2 and F1 --> Y1
        if x.size(-2) < x_c1.size(-2) or x.size(-1) < x_c1.size(-1):
            x = F.interpolate(input=x, size=(x_c1.size(-2), x_c1.size(-1)), mode='bilinear',
                              align_corners=True)  # [B,4C,H/8,W/8]-->[B,4C,H/4,W/4]
        x_c1_garan = x_c1
        x1 = self.garan2(l_garan,  x_c1_garan)
        x_garan = x1  # [B,C,H/4,H/4]
        x = torch.cat([x, x1], dim=1)  # -->[B,4C+C,H/4,W/4]
        x = self.conv1_2(x)  # [B,4C+C,H/4,W/4]-->[B,4C,H/4,W/4]
        # x = self.findit2(x,x_c1)
        x = self.bn1_2(x)
        x = self.relu1_2(x)

        x = self.conv2_2(x)  # [B,4C,H/4,W/4]-->[B,4C,H/4,W/4]
        x = self.bn2_2(x)
        x = self.relu2_2(x)  # Y1:[B,4C,H/4,W/4]

        return self.conv1_1(x), self.conv1(x_garan)  # [B,4C,H/4,W/4]-->[B,2,H/4,W/4]
