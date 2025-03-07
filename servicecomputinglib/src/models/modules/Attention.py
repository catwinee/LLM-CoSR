import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()

        # 定义注意力参数
        self.W_query = nn.Linear(input_dim, input_dim, bias=False)
        nn.init.xavier_uniform_(self.W_query.weight)
        self.W_key = nn.Linear(input_dim, input_dim, bias=False)
        nn.init.xavier_uniform_(self.W_key.weight)
        self.W_value = nn.Linear(input_dim, input_dim, bias=False)
        nn.init.xavier_uniform_(self.W_value.weight)
        self.a = nn.Parameter(torch.randn(1, 2 * input_dim))
        self.RELU = torch.nn.LeakyReLU()

    def forward(self, query, invocation_matrix):
        # 计算注意力权重
        query_proj = self.W_query(query)
        key_proj = self.W_key(query)
        scores = self.RELU(torch.matmul(torch.cat([query_proj, key_proj], dim=1), self.a.T))
        # scores = torch.mul(scores, invocation_matrix)
        scores = scores.masked_fill(invocation_matrix == 0, float(-1000))
        # scores = torch.matmul(query_proj, key_proj.transpose(-2, -1))
        attention_weights = F.softmax(scores, dim=-1)

        # 计算加权和
        value_proj = self.W_value(query)
        weighted_sum = torch.matmul(attention_weights, value_proj)

        return weighted_sum


class MotifAttention(nn.Module):
    def __init__(self, input_dim, node_num, heads_num):
        super().__init__()
        self.Wr = nn.Parameter(torch.randn(heads_num, input_dim//heads_num, 1))
        self.q = nn.Parameter(torch.randn(heads_num, node_num, 1))
        self.activation = torch.nn.ELU()
        self.head_nums = heads_num
        self.input_dim = input_dim
        self.output_Linear = nn.Linear(input_dim, input_dim)

    def forward(self, query_matrix):
        shape1 = query_matrix.shape
        query_matrix = query_matrix.view(query_matrix.shape[0], -1, self.head_nums, self.input_dim//self.head_nums).transpose(2, 1).transpose(1, 0)
        query_matrix1 = torch.bmm(query_matrix.reshape(query_matrix.shape[0], -1, query_matrix.shape[-1]), self.Wr).view(shape1[0], -1, shape1[-2]).transpose(0, 1)
        scores = torch.matmul(query_matrix1, self.q)
        attention_weights = F.softmax(scores, dim=-2)
        query_matrix = self.activation(torch.bmm(query_matrix.transpose(0, 2).transpose(-1, -2).reshape(self.head_nums, -1, shape1[0]), attention_weights)).view(self.head_nums, -1, self.input_dim//self.head_nums).transpose(0, 1)
        return self.output_Linear(query_matrix.reshape(-1, shape1[-1]))# 这一行有点小问题


class SelfMultiHeadAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=3):
        super(SelfMultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.RELU = torch.nn.LeakyReLU()
        self.query_linear = nn.Linear(input_dim, hidden_dim)
        self.key_linear = nn.Linear(input_dim, hidden_dim)
        self.value_linear = nn.Linear(input_dim, hidden_dim)
        self.output_linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, query, mask=None):
        batch_size = query.size(0)

        # 线性变换得到查询、键和值
        query = self.query_linear(query)
        key = self.key_linear(query)
        value = self.value_linear(query)

        # 将查询、键和值分割成多个头
        query = query.view(batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        key = key.view(batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        value = value.view(batch_size, self.num_heads, self.head_dim).transpose(0, 1)

        # 计算注意力得分
        scores = self.RELU(torch.bmm(query, key.transpose(-2, -1)))
        # scores = torch.mul(scores, mask)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float(-1000))

        # 注意力权重
        attention_weights = torch.softmax(scores, dim=-1)

        # 加权求和得到注意力表示
        attention_output = torch.matmul(attention_weights, value)

        # 将多个头的输出拼接并线性变换
        attention_output = attention_output.transpose(1, 0).contiguous().view(batch_size,
                                                                              self.num_heads * self.head_dim)
        attention_output = self.output_linear(attention_output)

        return attention_output


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(TransformerEncoder, self).__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=4, dim_feedforward=hidden_dim),
            num_layers=num_layers
        )

    def forward(self, input_data):
        output = self.transformer(input_data)
        return output[:, -1, :]


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=8):
        super(MultiHeadAttention, self).__init__()

        while input_dim % num_heads != 0:
            num_heads -= 1
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.query_linear = nn.Linear(input_dim, hidden_dim)
        self.key_linear = nn.Linear(input_dim, hidden_dim)
        self.value_linear = nn.Linear(input_dim, hidden_dim)
        self.output_linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 线性变换得到查询、键和值
        query = self.query_linear(query)
        key = self.key_linear(key)
        value = self.value_linear(value)

        # 将查询、键和值分割成多个头
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力得分
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim).float())

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float(-10000))

        # 注意力权重
        attention_weights = torch.softmax(scores, dim=-1)

        # 加权求和得到注意力表示
        attention_output = torch.matmul(attention_weights, value)

        # 将多个头的输出拼接并线性变换
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1,
                                                                              self.num_heads * self.head_dim)
        attention_output = self.output_linear(attention_output)

        return attention_output, attention_weights

class MaskedSelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MaskedSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
        self.apply(initialize_weights)

    def forward(self, values, keys, query, mask):
        if query.dim() < 3:
            values, keys, query, mask = values.unsqueeze(0), keys.unsqueeze(0), query.unsqueeze(0), mask.unsqueeze(0)
        mask = mask.unsqueeze(0)  # .repeat(1, self.heads, 1, 1)
        N_q = query.shape[0]
        N_k = keys.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N_k, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N_k, key_len, self.heads, self.head_dim)
        queries = query.reshape(N_q, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Scaled dot-product attention with mask
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask.transpose(-1, -2) == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N_q, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        if out.shape[0] == 1:
            out = out.squeeze(0)
        return out


from servicecomputinglib.src.utils.utils import initialize_weights

# class HHANAttention(nn.Module):
#     def __init__(self, input_dim, heads=4, num_layers=3, *args, **kwargs):
#         super(HHANAttention, self).__init__()
#
#         self.layers = nn.ModuleList(
#             [MaskedSelfAttention(input_dim, heads) for _ in range(num_layers)]
#         )
#
#     def forward(self, mashup, edges, mask=None):
#         for layer in self.layers:
#             mashup = layer(edges, mashup, mashup, mask)
#             edges = layer(mashup, edges, edges, mask.T)
#         return mashup

class HHANAttention(nn.Module):
    def __init__(self, input_dim, heads=4, num_layers=3, *args, **kwargs):
        super(HHANAttention, self).__init__()

        self.layers = nn.ModuleList(
            [MaskedSelfAttention(input_dim, heads) for _ in range(num_layers)]
        )

    def forward(self, node, edges, mask=None):
        for layer in self.layers:
            node = layer(edges, edges, node, mask)
            edges = layer(node, node, edges, mask.T)
        return node


import torch
import torch.nn as nn


from servicecomputinglib.src.utils.utils import KNNSparsify, ProbabilitySparsify, Discretize, AddEye, LinearTransform, NonLinearize, Symmetrize, Normalize, cal_similarity_graph, InnerProductSimilarity, CosineSimilarity, WeightedCosine, MLPRefineSimilarity

device = torch.device("cuda")

import torch.nn.init as init


class Attentive(nn.Module):
    def __init__(self, size):
        super(Attentive, self).__init__()
        self.w = nn.Parameter(torch.rand(size), requires_grad=True)

    def forward(self, x):
        return x @ torch.diag(self.w)

import torch
import torch.nn as nn

class BaseLearner1(nn.Module):
    """Abstract base class for graph learner"""
    def __init__(self, metric, processors):
        super(BaseLearner1, self).__init__()

        self.metric = metric
        self.processors = processors
class AttLearner(BaseLearner1):
    """Attentive Learner"""

    def __init__(self, metric, processors, nlayers, size, activation):

        super(AttLearner, self).__init__(metric, processors)
        self.nlayers = nlayers
        self.layers = nn.ModuleList()
        for _ in range(self.nlayers):
            self.layers.append(Attentive(size))
        self.activation = activation

    def internal_forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != (self.nlayers - 1):
                x = self.activation(x)
        return x

    def forward(self, features):
        z = self.internal_forward(features)
        z = F.normalize(z, dim=1, p=2)
        similarities = self.metric(z)
        for processor in self.processors:
            similarities = processor(similarities)
        similarities = F.relu(similarities)
        return similarities

