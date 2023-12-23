import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features):
            super(GraphAttentionLayer, self).__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.fc_features = nn.Linear(in_features, out_features).cuda()  # 连接层，用于将输入的节点特征（即features）转换为out_features维的特征。
            self.fc_multimodal = nn.Linear(768, out_features).cuda()  # 全连接层，用于将输入的多模态特征（即multimodal_features）转换为out_features维的特征
            self.attn = nn.Linear(2 * out_features, 1).cuda()  # 全连接层，用于计算注意力分数。它的输入是两个节点的特征（每个都是out_features维），因此输入维度是2 * out_features，输出是一个标量（即注意力分数），所以输出维度是1

    def forward(self, features, adj, multimodal_features):
        # global average pooling on multimodal_features and adjust its shape
        # features:[20,20,1,768]
        # adj:[20,20,20]
        # multi:[20,768,10,10]
        # 将多模态特征做了全局平均池化，使得不同大小的输入都能转换成统一的形状[B, C, 1, 1]
        h_multimodal = F.adaptive_avg_pool2d(multimodal_features, (1, 1))  # shape: [B, C, 1, 1] [20,768,1,1]
        # 将池化后的多模态特征进行reshape并通过全连接层，得到[B, C']的输出，其中C'是out_features
        h_multimodal = self.fc_multimodal(h_multimodal.view(h_multimodal.size(0), -1))  # shape: [B, C] [20,768]
        # 这行代码将多模态特征的形状变为 [B, 1, C']，以便于与节点特征进行广播操作
        h_multimodal = h_multimodal.view(h_multimodal.size(0), 1, -1)  # shape: [B, 1, C] [20,1,768]

        # transform features  将输入的节点特征通过全连接层，得到[B, N, C']的输出，其中N是节点数，C'是out_features
        h_features = self.fc_features(features.squeeze(2))  # shape: [B, N, C]  [20,20,768]

        # add features and multimodal_features 将节点特征和多模态特征进行相加，得到加权后的节点特征。由于两者的形状分别为[B, N, C']和[B, 1, C']，所以多模态特征会被广播到与节点特征相同的形状，然后进行相加。
        h_input = h_features + h_multimodal  # h_multimodal is broadcasted to [B, N, C] [20,20,768]

        # use the combined features for forward propagation
        # 为每一对节点生成一个包含两个节点的特征的新向量，这个新向量会被用于计算这两个节点之间的注意力得分
        a_input = torch.cat([h_input.unsqueeze(2).repeat(1, 1, h_input.size(1), 1),
                             h_input.unsqueeze(1).repeat(1, h_input.size(1), 1, 1)], dim=-1)  # [20,20,20,1536]
        a_input = a_input.sum(dim=2)  # sum over the node dimension  [20,20,1536]
        # 计算了每一对节点之间的原始注意力得分  [20,20,1]
        e = F.leaky_relu(
            self.attn(a_input.view(a_input.size(0) * a_input.size(1), -1)).view(a_input.size(0), a_input.size(1), -1))
        zero_vec = -9e15 * torch.ones_like(e).cuda()  # [20,20,1]
        # 根据邻接矩阵将无关节点（即邻接矩阵中对应位置为0的节点对）的注意力得分设置为一个非常小的值（接近于负无穷大） [20,20,20]
        attention = torch.where(adj > 0, e, zero_vec)
        # 将每一行的原始注意力得分转换为概率分布（即让它们的和为1）
        attention = F.softmax(attention, dim=-1)
        # 根据注意力得分对节点特征进行加权求和，得到新的节点表示  [20,20,768]
        h_prime = torch.bmm(attention, h_input)
        return h_prime


class GAT(nn.Module):
    def __init__(self, nfeat, nhid):
        super(GAT, self).__init__()
        self.attentions = GraphAttentionLayer(nfeat, nhid).cuda()
        self.out_att = GraphAttentionLayer(nhid, 768).cuda()

    def forward(self, multimodal_features, features, adj):
        # 对输入的节点特征进行dropout操作，以防止过拟合。
        x = F.dropout(features, 0.6, training=self.training)
        # 将dropout后的节点特征、邻接矩阵和多模态特征输入到第一个图注意力层，得到新的节点表示，并对其进行激活函数处理。
        x = F.elu(self.attentions(x, adj, multimodal_features))
        # 对第一个图注意力层的输出进行dropout操作，以防止过拟合
        x = F.dropout(x, 0.6, training=self.training)
        x = self.out_att(x, adj, multimodal_features)
        x = x.mean(dim=1)  # average pooling over the node dimension 对节点维度进行平均池化的操作
        return x  # [B,C]

