import torch.nn as nn
import torch
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F

class AdaptiveNRI(nn.Module):
    def __init__(self, api_embed_channels, num_api, do_prob): # 128, 128, 256, 945
        super(AdaptiveNRI, self).__init__()
        self.api_embed_channels = api_embed_channels
        self.num_api = num_api

        self.msg_passing = Node2Node(api_embed_channels * 2, api_embed_channels * 2, api_embed_channels * 2, do_prob)  # [935, 935, 935]
        self.mlp = MLP(api_embed_channels * 2, api_embed_channels * 2, api_embed_channels * 2, do_prob)  # [935, 935, 935]
        self.reduce_mlp = nn.Sequential(
            nn.Linear(num_api, api_embed_channels * 3),
            nn.ReLU(inplace=True),
            nn.Linear(api_embed_channels * 3, api_embed_channels)
        )
        self.increase_mlp = nn.Sequential(
            nn.Linear(api_embed_channels, api_embed_channels * 3),
            nn.ReLU(inplace=True),
            nn.Linear(api_embed_channels * 3, num_api),
            nn.Sigmoid()
        )
        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def forward(self, api_embeds, adjacency_matrix, edge_index): # [n, api_out_channels + api_nums]
        output = self.reduce_mlp(adjacency_matrix)
        output = torch.cat([api_embeds, output], dim=1)
        output = torch.cat([api_embeds, api_embeds], dim=1)
        output = self.msg_passing(output, edge_index)
        output = self.mlp(output)
        output = output[:, -self.api_embed_channels:]
        output = self.increase_mlp(output)
        return output


class MLP(nn.Module):
    def __init__(self, in_channels, hid_channels, out_channels, do_prob=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hid_channels)
        self.fc2 = nn.Linear(hid_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, inputs):
        x = F.elu(self.fc1(inputs))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = F.elu(self.fc2(x))
        return x
        # return self.batch_norm(x)

class Node2Node(MessagePassing):
    def __init__(self, in_channels, hid_channels, out_channels, do_prob=0.):
        super().__init__(aggr='add')
        self.mlp1 = MLP(in_channels * 2, hid_channels, out_channels, do_prob)
        self.mlp2 = MLP(in_channels, hid_channels, out_channels, do_prob)

    def forward(self, x, edge_index):
        # message aggreate update
        return self.propagate(edge_index=edge_index, x=x)

    def message(self, x_i, x_j):
        tmp = torch.cat([x_i, x_j], dim=-1)
        return self.mlp1(tmp) # in [edge_nums, node_features * 2] out [edge_nums, node_features]

    def update(self, inputs):
        return self.mlp2(inputs)

