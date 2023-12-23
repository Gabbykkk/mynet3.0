import torch
import torch.nn as nn
from torch.nn import functional as F


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features).cuda())
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features).cuda())
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid).cuda()
        self.gc2 = GraphConvolution(nhid, 768).cuda()  # 输出特征数设置为768
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj)).cuda()
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj).cuda()
        return x  # 不再使用softmax，因为我们不是在做分类任务


class MultimodalGCN(nn.Module):
    def __init__(self, nfeat, nhid, dropout):  # nfeat :
        super(MultimodalGCN, self).__init__()
        self.gcn_model = GCN(nfeat, nhid, dropout)
        self.fc = nn.Linear(768*10*10, 768).cuda()  # 将多模态特征调整到与节点特征相同的维度

    def forward(self, multimodal_features, node_features, adjacency_matrix):
        # 将多模态特征调整到与节点特征相同的维度
        multimodal_features = self.fc(multimodal_features.view(multimodal_features.size(0), -1)).cuda()  # [20, 768]
        multimodal_features = multimodal_features.unsqueeze(1).unsqueeze(1)  # [20, 1, 1, 768]

        # 将多模态特征添加到节点特征中，作为一个新的全局节点
        extended_node_features = torch.cat([node_features, multimodal_features], dim=1)  # [20, 21, 1, 768]

        # 添加全局节点到邻接矩阵中
        N = adjacency_matrix.size(1)
        extended_adjacency_matrix = torch.zeros(adjacency_matrix.size(0), N+1, N+1).cuda()  # [20, 21, 21]
        extended_adjacency_matrix[:, :N, :N] = adjacency_matrix  # [20, 21, 21]
        extended_adjacency_matrix[:, N, :] = 1  # 全局节点与所有其他节点相连
        extended_adjacency_matrix[:, :, N] = 1  # [20, 21, 21]

        # 归一化邻接矩阵
        rowsum = extended_adjacency_matrix.sum(2)  # [20, 21]
        d_inv_sqrt = rowsum.pow(-0.5).unsqueeze(2)  # [20, 21, 1]
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        extended_adjacency_matrix = extended_adjacency_matrix * d_inv_sqrt  # [20, 21, 21]
        extended_adjacency_matrix = extended_adjacency_matrix.transpose(1, 2) * d_inv_sqrt  # [20, 21, 21]

        # 现在你可以使用扩展的节点特征和邻接矩阵来运行你的GCN模型
        node_outputs = self.gcn_model(extended_node_features.squeeze(2), extended_adjacency_matrix)  # [20, 21, 768]

        # 对所有节点的输出进行平均，得到一个[20, 768]的输出
        output = node_outputs.mean(dim=1)

        return output


# class GraphConvolution(nn.Module):
#     """
#     Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
#     """
#     def __init__(self, in_features, out_features, bias=True):
#         super(GraphConvolution, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features).cuda())
#         if bias:
#             self.bias = nn.Parameter(torch.FloatTensor(out_features).cuda())
#         else:
#             self.register_parameter('bias', None)
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         nn.init.kaiming_uniform_(self.weight)
#         if self.bias is not None:
#             nn.init.zeros_(self.bias)
#
#     def forward(self, input, adj):
#         support = torch.matmul(input, self.weight)
#         output = torch.matmul(adj, support)
#         if self.bias is not None:
#             return output + self.bias
#         else:
#             return output
#
#
# class GCN(nn.Module):
#     def __init__(self, nfeat, nhid, dropout):
#         super(GCN, self).__init__()
#
#         self.gc1 = GraphConvolution(nfeat, nhid).cuda()
#         self.gc2 = GraphConvolution(nhid, 768).cuda()  # 设置输出特征数为768
#         self.dropout = dropout
#
#     def forward(self, x, adj):
#         x = F.relu(self.gc1(x, adj)).cuda()
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = self.gc2(x, adj).cuda()
#         return x  # 不再使用softmax，因为不是进行分类任务
#
#
# class MultimodalGCN(nn.Module):
#     def __init__(self, nfeat, nhid, dropout):
#         super(MultimodalGCN, self).__init__()
#         self.gcn_model = GCN(nfeat, nhid, dropout)
#
#     def forward(self, node_features, adjacency_matrix):
#         # 不再添加多模态特征
#
#         # 直接运行GCN模型
#         node_outputs = self.gcn_model(node_features.squeeze(2), adjacency_matrix)
#
#         # 对所有节点的输出进行平均，得到一个[20, 768]的输出
#         output = node_outputs.mean(dim=1)
#
#         return output
