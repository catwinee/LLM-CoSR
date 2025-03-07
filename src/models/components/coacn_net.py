import json
import os

import numpy as np
import torch
from torch import nn
from torch_geometric.nn import LightGCN


class COACNNet(nn.Module):
    def __init__(self, hparams: dict):
        super(COACNNet, self).__init__()
        self.mashup_embed_channels = hparams['mashup_embed_channels']
        self.api_embed_channels = hparams['api_embed_channels']
        self.domain_embed_channels = hparams['domain_embed_channels']
        self.feature_dim = hparams['feature_dim']
        self.beta = hparams['hp_beta']
        self.num_gcn_layer = hparams['hp_num_gcn_layer']
        self.weight_gcn_layer = hparams['hp_weight_gcn_layer']
        data_dir = hparams['data_dir']
        self.register_buffer('mashup_embed', torch.from_numpy(np.load(os.path.join(data_dir, hparams['mashup_embed_path']))))
        self.num_mashup = self.mashup_embed.size(0)
        self.register_buffer('domain_embed', torch.from_numpy(np.load(os.path.join(data_dir, hparams['domain_embed_path']))))
        self.num_domain = self.domain_embed.size(0)
        self.register_buffer('api_embed', torch.from_numpy(np.load(os.path.join(data_dir, hparams['api_embed_path']))))
        self.num_api = self.api_embed.size(0)
        self.register_buffer('invoked_matrix', torch.from_numpy(np.load(os.path.join(data_dir, hparams['invoked_matrix_path']))))

        A_1 = torch.cat((torch.zeros((self.num_mashup, self.num_mashup), dtype=torch.float32), self.invoked_matrix),
                        dim=1)
        A_2 = torch.cat((torch.transpose(self.invoked_matrix, 0, 1), torch.zeros((self.num_api, self.num_api))), dim=1)
        A = torch.cat((A_1, A_2), dim=0).long()  # (num_mashup + num_api, num_mashup + num_api)
        self.register_buffer('A', A)
        # D = torch.sum(A, dim=1)
        # for i in range(len(D)):
        #     if (D[i] - 0.) < 1e-6:
        #         D[i] = 1.
        # D = torch.diag(torch.pow(D, -0.5))
        # self.register_buffer('A_head', torch.matmul(torch.matmul(D, A), D))

        self.sde_fc = nn.Sequential(
            nn.Linear(in_features=self.mashup_embed_channels, out_features=self.feature_dim),
            nn.Sigmoid(),
        )
        self.sde_fc_value = nn.Sequential(
            nn.Linear(in_features=self.domain_embed_channels, out_features=self.feature_dim),
            nn.Sigmoid(),
        )
        self.sde_fc_key = nn.Sequential(
            nn.Linear(in_features=self.domain_embed_channels, out_features=self.feature_dim),
            nn.Sigmoid(),
        )
        self.sie_fc = nn.Sequential(
            nn.Linear(in_features=self.api_embed_channels, out_features=self.feature_dim),
            nn.Sigmoid(),
        )
        self.light_gcn = LightGCN(num_nodes=self.num_mashup + self.num_api, embedding_dim=self.feature_dim, num_layers=self.num_gcn_layer)

    def forward(self, x: torch.Tensor):
        batch_size = x.size(0)

        # Service Domain Enhancement
        v_mi = self.sde_fc(x)
        v_mi = torch.unsqueeze(v_mi, 1)  # (size_batch, 1, dim_embedding)
        v_mi = torch.unsqueeze(v_mi, 1)  # (size_batch, 1, 1, dim_embedding)

        v_value = self.sde_fc_value(self.domain_embed)  # (num_domain, dim_embedding)
        v_key = self.sde_fc_key(self.domain_embed)  # (num_domain, dim_embedding)
        # v_value = torch.unsqueeze(v_value, 2)
        v_key = v_key.unsqueeze(2)  # (num_domain, dim_embedding, 1)

        al_matrix = torch.matmul(v_mi, v_key)
        al_matrix = al_matrix.view(batch_size, -1)  # (size_batch, num_domain)
        alpha_sum = torch.sum(al_matrix, dim=1)  # (size_batch, )
        alpha = torch.div(al_matrix, alpha_sum.unsqueeze(1))  # (size_batch, num_domain)
        alpha = alpha.view(batch_size, -1)
        s_m = torch.mul(alpha.unsqueeze(-1), v_value)
        s_m = s_m.squeeze(-1)  # (size_batch, num_domain, dim_embedding)
        s_m = torch.sum(s_m, dim=1)  # (size_batch, dim_embedding)
        v_mi = v_mi.view(batch_size, self.feature_dim)
        z_m = (1 - self.beta) * s_m + self.beta * v_mi  # (size_batch, dim_embedding)

        # Structured Information Extraction
        v_m = self.sde_fc(self.mashup_embed)
        v_s = self.sie_fc(self.api_embed)

        embed = torch.cat((v_m, v_s), dim=0)  # (num_mashup + num_api, dim_embedding)
        self.light_gcn.embedding = nn.Embedding.from_pretrained(embed)

        # # X = self.weight_gcn_layer[0] * X_k
        # for i in range(self.num_gcn_layer):
        #     X = torch.matmul(self.A_head, X)
        #     X = X + self.weight_gcn_layer[i] * X
        embed = self.light_gcn.get_embedding(self.A.to_sparse().indices())

        O = embed[-self.num_api:]  # (num_api, dim_embedding)
        z_m = torch.unsqueeze(z_m, dim=1)
        z_m = torch.unsqueeze(z_m, dim=1)
        O = torch.unsqueeze(O, dim=2)
        pred = torch.matmul(z_m, O)
        pred = pred.view(batch_size, self.num_api)
        # pred = self.sigmoid(pred)
        return pred


if __name__ == '__main__':
    with open('../../../data/api_mashup/mashup_related_api.json', 'r', encoding='utf-8') as f:
        apis = json.load(f)
    print(len(apis))
