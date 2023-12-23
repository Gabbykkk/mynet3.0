import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class PhraseAttention(nn.Module):
  def __init__(self, input_dim):
    super(PhraseAttention, self).__init__()
    # initialize pivot
    self.fc = nn.Linear(input_dim, 1).cuda()
    self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 5))  # 这将使 H 和 W 分别变为 4 和 5

  def forward(self, context, embedded, input_labels):
    """
    Inputs:
    - context : Variable float (batch, seq_len, input_dim)  上下文向量
    - embedded: Variable float (batch, seq_len, word_vec_size)  嵌入向量
    - input_labels: Variable long (batch, seq_len)  输入标签
    Outputs:
    - attn    : Variable float (batch, seq_len)
    - weighted_emb: Variable float (batch, word_vec_size)
    """
    context = self.adaptive_pool(context)  # 形状为 (B, C, 4, 5)
    batch, channel, h, w = context.size()
    context = context.view(batch, channel, h * w).permute(0, 2, 1).cuda()  # [20,20,768]
    embedded = embedded.permute(0, 2, 1)
    input_labels = input_labels.squeeze(2)

    cxt_scores = self.fc(context).squeeze(2)  # (batch, seq_len) [20,20] 通过全连接层计算上下文向量的分数，然后通过squeeze操作移除维度为1的维度。
    attn = F.softmax(cxt_scores)  # (batch, seq_len), attn.sum(1) = 1. 使用softmax函数将上下文分数转换为注意力权重，这样所有的权重之和为1

    # mask zeros
    is_not_zero = (input_labels!=0).float()  # (batch, seq_len) 创建一个掩码向量，其中输入标签不为0的位置为1，否则为0。这用于忽略输入中的填充部分。
    attn = attn * is_not_zero  # (batch, seq_len) 应用掩码向量，将注意力权重中对应输入标签为0（填充部分）的位置设为0。
    attn = attn / attn.sum(1).view(attn.size(0), 1).expand(attn.size(0), attn.size(1))  # (batch, seq_len)  重新归一化注意力权重，使得非零权重之和为1

    # compute weighted embedding
    attn3 = attn.unsqueeze(1)  # (batch, 1, seq_len) 增加一个维度，为下一步的矩阵乘法做准备
    weighted_emb = torch.bmm(attn3, embedded)  # (batch, 1, word_vec_size) 使用注意力权重对嵌入向量进行加权，得到加权嵌入向量。
    weighted_emb = weighted_emb.squeeze(1)  # (batch, word_vec_size) 移除维度为1的维度，得到最终的加权嵌入向量。

    return weighted_emb  # 这个类的输出是注意力权重和加权嵌入向量，它们都是根据上下文信息计算得到的。
