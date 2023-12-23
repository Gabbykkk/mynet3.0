import torch
from torch import nn

class CBP(nn.Module):
    def __init__(self, input_dim, sum_pool=False):
        super(CBP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = input_dim
        self.sum_pool = sum_pool

        # 初始化稀疏随机矩阵参数
        self.sparse_sketch_matrix1 = nn.Parameter(torch.randn(self.output_dim, self.input_dim), requires_grad=False)
        self.sparse_sketch_matrix2 = nn.Parameter(torch.randn(self.output_dim, self.input_dim), requires_grad=False)

    def forward(self, l_feats, l_pooler):
        l_feats = torch.mean(l_feats,dim=2)
        assert l_feats.size(1) == self.input_dim and l_pooler.size(1) == self.input_dim

        batch_size = l_feats.size(0)

        # 稀疏矩阵乘法
        sketch_1 = torch.matmul(l_feats, self.sparse_sketch_matrix1.t())
        sketch_2 = torch.matmul(l_pooler, self.sparse_sketch_matrix2.t())

        # 傅里叶变换
        fft1 = torch.fft.fft(sketch_1, dim=1)
        fft2 = torch.fft.fft(sketch_2, dim=1)

        # 乘积计算
        fft_product = fft1 * fft2

        # 逆傅里叶变换
        cbp = torch.fft.ifft(fft_product, dim=1).real

        # 返回输出
        if self.sum_pool:
            cbp = cbp.sum(dim=0)

        return cbp
