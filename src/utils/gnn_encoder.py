from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# 添加 MLP, dropout, Linear, 激活， 之前
class GNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNEncoder, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# 使用 MLP 替换 gnn
# 结构特征被重复计算