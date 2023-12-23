import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        # print(hidden_dim)  # 768
        # print(num_heads)  12

        assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"

        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(1024, hidden_dim)
        self.v_linear = nn.Linear(1024, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, query, value, key):
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)

        Q = Q.view(query.shape[0], -1, self.num_heads, self.head_dim).permute(2, 0, 1, 3)
        K = K.view(key.shape[0], -1, self.num_heads, self.head_dim).permute(2, 0, 1, 3)
        V = V.view(value.shape[0], -1, self.num_heads, self.head_dim).permute(2, 0, 1, 3)

        energy = torch.matmul(Q, K.transpose(-2, -1)) / self.head_dim**0.5
        attention = torch.softmax(energy, dim=-1)
        out = torch.matmul(attention, V).permute(1, 2, 0, 3).contiguous()
        out = out.view(query.shape[0], -1, self.hidden_dim)
        out = self.fc_out(out)

        return out


class hlf(nn.Module):
    def __init__(self, hidden_dim, all_hidden_states):
        super(hlf, self).__init__()
        self.mha = MultiHeadAttention(hidden_dim=hidden_dim, num_heads=len(all_hidden_states))
        self.dropout = nn.Dropout(0.5)
        self.ln = nn.LayerNorm(hidden_dim)  # Use Layer Normalization instead of Batch Normalization
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # 全局平均池化层
        self.conv_l = nn.Conv1d(in_channels=768, out_channels=768, kernel_size=20, stride=20)
        self.weights = nn.Parameter(torch.rand(len(all_hidden_states), requires_grad=True))  # Add weights for linear combination

    def forward(self, all_hidden_states, multimodal_feats):
        attention_outputs = []
        # multimodal_feats:[20,768,10,10]
        # hidden_state:[20,20,768]
        batch, channel, h, w = multimodal_feats.size()

        multimodal_feats = multimodal_feats.view(batch, channel, h*w).permute(0, 2, 1)
        for hidden_state in all_hidden_states:
            attention_output = self.mha(hidden_state.cuda(), multimodal_feats.cuda(), multimodal_feats.cuda())
            attention_outputs.append(attention_output)

        # Apply weights and sum for linear combination
        l_mha = sum(w * out for w, out in zip(self.weights, attention_outputs))
        l_mha = self.dropout(l_mha)
        l_mha = self.ln(l_mha)  # Apply Batch Normalization
        l_hlf = l_mha.permute(0, 2, 1)  # [20,768,20]
        # print(l_hlf.shape)
        l_hlf = self.conv_l(l_hlf).squeeze(dim=2)  # 使得[B,768，20]-->[B,768]，或者使用全局最大池化或全局平均池化，通常也有很好的效果

        return l_hlf