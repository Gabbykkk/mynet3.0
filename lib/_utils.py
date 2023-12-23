import pdb
from collections import OrderedDict
import sys
import torch
from torch import nn
from torch.nn import functional as F

# from add.HLF import hlf
# from add.PharseAtt import PhraseAttention
# from add.l_visualize import plot_heatmaps
from bert.modeling_bert import BertModel


############用的模型lavt
class _LAVTSimpleDecode(nn.Module):
    def __init__(self, backbone, classifier):
        super(_LAVTSimpleDecode, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.conv_l = nn.Conv1d(in_channels=768, out_channels=768, kernel_size=20, stride=20)

    def forward(self, x, l_feats, l_pooler, hlf_model, all_hidden_states, l_mask):
        # x : 图像数据  [B，C,H,W] [20,3,320,320]
        # l_feats : 位置特征 [B,Ct,T] [20,768,20]
        # l_pooler : 池化特征 [B,Ct]  [20,768]
        # l_mask : 位置掩码 [B,T,1] [20,20,1]
        input_shape = x.shape[-2:]  # [320,320]

        features = self.backbone(x, l_feats, l_mask)  # 首先通过主干网络将输入图像和特征进行处理，得到不同层级的特征表示
        x_c1, x_c2, x_c3, x_c4 = features
        # x_c1:[20,96,80,80]
        # x_c2:[20,192,40,40]
        # x_c3:[20,384,20,20]
        # x_c4:[20,768,10,10]  base:[20,1024,10,10]
        # ----- garan使用l_feats时
        # l = torch.mean(l_feats, dim=2)
        # l_hlf = self.conv_l(l_hlf).squeeze(dim=2)  # 使得[B,768，20]-->[B,768]，或者使用全局最大池化或全局平均池化，通常也有很好的效果
        l_hlf = hlf_model(all_hidden_states, x_c4)
        # l_pharseAtt = pharAtt_model(x_c4, l_feats, l_mask)
        # l_mgcn = mGAT_model(x_c4, node_features, adjacency_matrix)
        # l_mgcn_cpu = l_mgcn.cpu().detach().numpy()
        # l_pooler_cpu = l_pooler.cpu().detach().numpy()
        # plot_heatmaps(l_mgcn_cpu, l_pooler_cpu, "Heatmap of l_mgcn", "Heatmap of l_pooler")
        x, x_garan = self.classifier(l_hlf, x_c4, x_c3, x_c2, x_c1)  # 利用分类器对这些特征进行分类  x:[20,2,80,80]
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)  # 使用双线性插值方法将输出的特征图调整到与输入图像相同的尺寸
        x_garan = F.interpolate(x_garan, size=input_shape, mode='bilinear', align_corners=True)  # 使用双线性插值方法将输出的特征图调整到与输入图像相同的尺寸
        return x, x_garan  # 返回调整后的特征图  [20,2,320,320]


class LAVT(_LAVTSimpleDecode):
    pass


###############################################
# LAVT One: put BERT inside the overall model #这一部分不用
###############################################
class _LAVTOneSimpleDecode(nn.Module):
    def __init__(self, backbone, classifier, args):
        super(_LAVTOneSimpleDecode, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.text_encoder = BertModel.from_pretrained(args.ck_bert)
        # ----改
        # self.text_encoder.pooler = None  # pooler_output

    def forward(self, x, text, l_mask):
        print("------lavtone")
        input_shape = x.shape[-2:]
        ### language inference ###
        l_feats = self.text_encoder(text, attention_mask=l_mask)[0]  # (6, 10, 768)
        # ----加
        l_pooler = self.text_encoder(text, attention_mask=l_mask)[1]
        # print("l_feats_pooler:",l_feats_pooler.shape)  # [20,6400,96]
        # ----
        l_feats = l_feats.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy   [B,dim,20] N_l单词数
        l_mask = l_mask.unsqueeze(dim=-1)  # (batch, N_l, 1)
        ##########################
        features = self.backbone(x, l_feats, l_pooler, l_mask)  # encoder，得到4个特征
        x_c1, x_c2, x_c3, x_c4 = features
        x = self.classifier(l_pooler, x_c4, x_c3, x_c2, x_c1)  # decoder,输入是4个特征
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)

        return x


class LAVTOne(_LAVTOneSimpleDecode):
    pass
