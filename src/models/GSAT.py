import itertools
import json
import math
import os.path
import pickle
import re

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb
from torch.nn.utils.rnn import pad_sequence

from servicecomputinglib.src.utils.metrics import Precision, NormalizedDCG
from servicecomputinglib.src.models.modules.Attention import SelfAttention, MotifAttention, MultiHeadAttention
from servicecomputinglib.src.utils.utils import cluster_pic_show, random_walk, select_indices

from torchmetrics import MaxMetric
from servicecomputinglib.src.models.modules.random_walk import random_walk_subgraph_pyg
from servicecomputinglib.src.models.modules.augs import Augmentor
from servicecomputinglib.src.models.modules.GNN import GraphNet

from servicecomputinglib.src.utils.metrics import OverlapRate, Count, Coverage, SavingRate
from servicecomputinglib.src.utils.utils import jaccard_similarity, LambdaRankLoss
import random
import json
import re
from src.utils.metrics import F1ScoreMetrics

class GSAT(pl.LightningModule):
    @property
    def device(self):
        return self._device

    def __init__(self, data_dir, lr, head_num, weight_decay, embedding_model='openai', fine_tuning=False,
                 topk=None, dataset='Mashup', enhanced=False, simple=False, mode='cover', tag_method='chatgpt', alpha=25, R=1, sample_mode='highest'):

        # TODO: 新增了四个参数，R, alpha, mode和tag_method，记得configs/model/XX.yaml中也要加上这四个参数

        super().__init__()
        self.dataset = dataset
        if topk is None:
            topk = [10, 15, 5]
        self.fine_tuning = fine_tuning
        invocation_path = '/../api_mashup/train_partial_invocation_seed=12345.pkl'

        if embedding_model == 'openai':
            pre_data_dir = data_dir + '/preprocessed_data/openai_emb/'
            if enhanced is False:
                # self.api_embeds = torch.stack(torch.load(pre_data_dir + 'api_openai_text_embedding.pt'), dim=0).to('cuda')
                # self.mashup_embeds = torch.stack(torch.load(pre_data_dir + 'mashup_openai_text_embedding.pt'), dim=0).to('cuda')
                # self.mashup_embeds = np.load(os.path.join(self.data_dir, self.mashup_path))
                # self.api_embeds = np.load(os.path.join(self.data_dir, self.api_path))
                self.api_embeds = torch.tensor(np.load(data_dir + '/../api_mashup/embeddings/partial/text_bert_api_embeddings.npy'), device='cuda')
                self.mashup_embeds = torch.tensor(np.load(data_dir + '/../api_mashup/embeddings/text_bert_mashup_embeddings.npy'), device='cuda')
            else:
                self.api_embeds = torch.stack(torch.load(data_dir + 'api_chatgpt_openai_text_embedding.pt'), dim=0)
                self.mashup_embeds = torch.stack(torch.load(data_dir + 'mashup_chatgpt_openai_text_embedding.pt'), dim=0)
            if simple is True:
                emb_layer = torch.nn.Sequential(torch.nn.Linear(self.num_api+self.num_mashup, (self.num_api+self.num_mashup)//2),
                                                torch.nn.ReLU(),
                                                torch.nn.Linear((self.num_api+self.num_mashup)//2, self.api_embeds.shape[-1]),
                                                torch.nn.ReLU(),
                                                torch.nn.Linear(self.api_embeds.shape[-1], self.api_embeds.shape[-1]))
                self.mashup_embeds = torch.zeross(self.num_mashup, self.num_mashup)
                for i in range(self.mashup_embeds.shape[0]):
                    self.mashup_embeds[i][i] = 1
                self.mashup_embeds = emb_layer(self.mashup_embeds)
        elif embedding_model == 'bert' and fine_tuning is True:
            pre_data_dir = data_dir + '/preprocessed_data/description/'
            self.mashup_embeds = torch.load(pre_data_dir + 'pre_bert_mashup_embedding.pt').to(self.device)
            self.api_embeds = torch.load(pre_data_dir + 'pre_bert_api_embedding.pt').to(self.device)
        elif embedding_model == 'bert' and fine_tuning is False:
            pre_data_dir = data_dir + '/preprocessed_data/bert/'
            self.api_embeds = torch.load(pre_data_dir + 'bert_apis_embeddings.emb')
            self.mashup_embeds = torch.load(pre_data_dir + 'bert_mashup_embeddings.emb')[1]
        else:
            pre_data_dir = data_dir + '/preprocessed_data/word2vec/'
            self.api_embeds = torch.stack(torch.load(pre_data_dir + 'api_word2vec_text_embedding.pt'), dim=0)
            self.mashup_embeds = torch.stack(torch.load(pre_data_dir + 'mashup_word2vec_text_embedding.pt'), dim=0)
        # edges = torch.load(data_dir + edge_path)
        invocation_df = pickle.load(open(data_dir + invocation_path, "rb"))
        edge_index = [[], []]
        for mashup_index, api_list in zip(invocation_df['X'], invocation_df['Y']):
            edge_index[0].extend([mashup_index] * len(api_list))
            edge_index[1].extend([api + 4557 for api in api_list])
        edges = torch.tensor(edge_index, dtype=int).T # [num_edges, 2]
        subgraph_path = data_dir + '/preprocessed_data/openai_emb/sub_graph_list.pt'
        self.input_channel = self.api_embeds.shape[1]
        self.hidden_channel = int(self.api_embeds.shape[1] / 2)
        self.num_api = self.api_embeds.shape[0]
        self.num_mashup = self.mashup_embeds.shape[0]

        self.device = torch.device('cuda:0')

        graph = torch.zeros(self.num_mashup + self.num_api, self.num_mashup + self.num_api).to(self.device)
        invert_graph = torch.ones(self.num_mashup, self.num_api).to(self.device)
        enhenced_edges = edges.numpy().T

        api_hot_dict = {}
        for edge in enhenced_edges.T:
            if edge[1] - self.num_mashup in api_hot_dict:
                api_hot_dict[edge[1] - self.num_mashup] += 1
            else:
                api_hot_dict[edge[1] - self.num_mashup] = 1
        for api in api_hot_dict:
            api_hot_dict[api] = math.log10(api_hot_dict[api] + 1)
        self.api_hot_wight = torch.zeros(self.num_api, device='cuda')
        for api in api_hot_dict:
            self.api_hot_wight[api] = api_hot_dict[api]

        self.graph_line = torch.zeros([self.num_mashup, self.num_api], device='cuda')
        for edge in enhenced_edges.T:
            self.graph_line[edge[0]][edge[1] - self.num_mashup] = 1

        for edge in enhenced_edges.T:
            graph[edge[0]][edge[1]] = 1
            graph[edge[1]][edge[0]] = 1
            invert_graph[edge[0]][edge[1] - self.num_mashup] = 0

        self.graph = graph
        self.invert_graph = invert_graph
        if os.path.exists(data_dir + '/preprocessed_data/motif.list') is False:
            self.motif_invocation = self.format_samples(data_dir)
        else:
            with open(data_dir + '/preprocessed_data/motif.list', 'rb') as f:
                self.motif_invocation = pickle.load(f)
        self.motif_invocation.insert(0, self.graph)
        for i in range(len(self.motif_invocation) - 3):
            self.motif_invocation.pop()
        self.motif_invocation = [self.graph]
        self.head_nums = head_num
        self.lr = lr
        self.weight_decay = weight_decay
        self.topk = topk

        self.pool = 'mean'
        self.val_auc_best = MaxMetric()
        self.criterion = torch.nn.MultiLabelSoftMarginLoss()

        if os.path.exists(subgraph_path) is False:
            self.sub_graph_list = {}
            edge_index = torch.tensor(edges).T
            for node in range(self.num_mashup + self.num_api):
                subgraph = random_walk_subgraph_pyg(torch.tensor(edge_index), node, 20)
                self.sub_graph_list[node] = subgraph
            torch.save(self.sub_graph_list, subgraph_path)
        else:
            self.sub_graph_list = torch.load(subgraph_path)

        self.mashup_api = {}
        self.api_mashup = {}
        for edge in enhenced_edges.T:
            if int(edge[0]) not in self.mashup_api:
                self.mashup_api[int(edge[0])] = [int(edge[1])]
            else:
                self.mashup_api[int(edge[0])].append(int(edge[1]))
            if int(edge[1]) not in self.api_mashup:
                self.api_mashup[int(edge[1])] = [int(edge[0])]
            else:
                self.api_mashup[int(edge[1])].append(int(edge[0]))
        self.mashup_api.update(self.api_mashup)

        self.node_sequence_list = []
        for node in range(graph.shape[0]):
            node_sequence = random_walk(self.mashup_api, node, steps=20)
            self.node_sequence_list.append(node_sequence)

        self.mashup_attention = MultiHeadAttention(self.input_channel, self.input_channel)
        self.api_attention = MultiHeadAttention(self.input_channel, self.input_channel)

        node_attention = []
        for i in range(3):
            node_attention.append(nn.Linear(self.input_channel, self.input_channel))
            node_attention.append(SelfAttention(self.input_channel))
        self.node_attention2 = nn.ModuleList(node_attention)
        self.motif_attention2 = MotifAttention(self.input_channel, self.num_mashup + self.num_api, self.head_nums)
        self.RELU = nn.LeakyReLU()
        self.gnn = GraphNet(self.input_channel, self.input_channel, self.input_channel)

        self.mashups_MLP = nn.Sequential(
            nn.Linear(self.input_channel, self.hidden_channel),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_channel, self.hidden_channel),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_channel, self.input_channel)
        )
        self.api_MLP = nn.Sequential(
            nn.Linear(self.input_channel, self.hidden_channel),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_channel, self.hidden_channel),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_channel, self.input_channel)
        )
        # self.proj_Linear1 = nn.Linear(self.input_channel + len(self.tag_word_id), self.input_channel)
        self.proj_Linear2 = nn.Linear(self.input_channel, self.hidden_channel)
        self.mashup_attention = MultiHeadAttention(self.input_channel, self.input_channel)
        self.api_attention = MultiHeadAttention(self.input_channel, self.input_channel)
        self.GRU = nn.GRU(self.input_channel, self.input_channel)
        self.augmentor = Augmentor(aug_ratio=0.2)

        self.cosine_similarity = nn.CosineSimilarity(dim=1)

        self.max = 0
        self.f1 = F1ScoreMetrics(top_k=5).to('cuda')

    def forward(self, users):
        mashup_embeds = self.mashup_embeds
        api_embeds = self.api_embeds
        mashup_api_enhanced = torch.matmul(self.graph_line * self.api_hot_wight, api_embeds)
        mashup_embeds += mashup_api_enhanced
        nodes = torch.cat([mashup_embeds, api_embeds], dim=0).to(self.device)

        embeddings = nodes.to(self.device).to(torch.float32)

        node_sequence_list = []
        for node_sequence in self.node_sequence_list:
            node_sequence1 = nodes[node_sequence]
            node_sequence_list.append(node_sequence1)
        if self.dataset == 'Mashup':
            with torch.no_grad():
                node_sequence_list = pad_sequence(node_sequence_list, batch_first=True).to(torch.float32)
        else:
            node_sequence_list = pad_sequence(node_sequence_list, batch_first=True)
        enhanced_embedding = self.GRU(node_sequence_list)[0][:, 0, :]

        input_matrix = []
        for i, matrix1 in enumerate(self.motif_invocation):
            if i < 2:
                input1 = self.node_attention2[2 * i](embeddings)
                input1 = self.RELU(input1)
                input1 = self.node_attention2[2 * i - 1](input1, torch.tensor(matrix1.to(self.device)))
                input_matrix.append((input1).unsqueeze(0))
        input1 = torch.cat(input_matrix, dim=0)
        if input1.dim() > 2:
            input1 = self.motif_attention2(input1)
        embeddings = input1 + embeddings + enhanced_embedding

        mashup_embeddings = embeddings[:self.num_mashup].unsqueeze(0)
        mashup_embeddings = self.api_attention(mashup_embeddings, mashup_embeddings, mashup_embeddings)[0]
        mashup_embeddings = self.mashup_attention(mashup_embeddings, mashup_embeddings, mashup_embeddings)

        mashup_embeddings = self.mashups_MLP(mashup_embeddings[0].squeeze(0))
        api_embeddings = self.api_MLP(embeddings[self.num_mashup:])
        output = torch.cat([mashup_embeddings, api_embeddings], dim=0)

        return output

    def format_samples(self, data_dir):

        def motif_2_calculation(invocation):
            d = torch.sum(invocation, dim=1)
            D = torch.zeros(self.num_mashup + self.num_api, self.num_mashup + self.num_api)
            for index, i in enumerate(d.tolist()):
                if index < self.num_mashup:
                    D[index][index] = i
            M = D * invocation
            for row_index in range(self.num_mashup):
                item_list = torch.nonzero(invocation[row_index])
                if item_list.shape[0] > 1:
                    combination = list(itertools.combinations(item_list.squeeze(-1), 2))
                    for elem in combination:
                        M[elem[0]][elem[1]] += 1
            return M

        graph_copy = self.graph.to(torch.device('cpu'))
        motif2 = motif_2_calculation(graph_copy)
        with open(data_dir + '/preprocessed_data/motif.list', 'wb') as f:
            pickle.dump([motif2], f)
        return [motif2]

    def training_step(self, batch, batch_idx):
        # user, pos_item, _ = batch['users'], batch['pos_items'], batch['neg_items']
        # output_layer = self.forward(user)
        # preference = torch.matmul(output_layer[user], output_layer[self.num_mashup:].T)
        mashups, labels, idx = batch
        output = self.forward(idx)
        preds = torch.matmul(output[idx], output[self.num_mashup:].T)
        loss = self.criterion(preds, labels) + torch.mean(self.cosine_similarity(preds, labels))
        self.log("loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # user, pos_items, _ = batch['users'], batch['pos_items'], batch['neg_items']
        # out = self.forward(user)
        # out = torch.matmul(out[user], out[self.num_mashup:].transpose(0, 1))
        # preds = torch.sigmoid(out)
        # preds = torch.mul(preds, self.invert_graph[user])
        # self.log('val/F1', self.f1(preds, pos_items.long()), on_step=False, on_epoch=True, prog_bar=True)
        # user, pos_items, _ = batch['users'], batch['pos_items'], batch['neg_items']
        mashups, labels, idx = batch
        out = self.forward(idx)
        out = torch.matmul(out[idx], out[self.num_mashup:].T)
        preds = torch.sigmoid(out)
        preds = torch.mul(preds, self.invert_graph[idx])
        self.log('val/F1', self.f1(preds, labels.long()), on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        # user, pos_items, _ = batch['users'], batch['pos_items'], batch['neg_items']
        # out = self.forward(user)
        # out = torch.matmul(out[user], out[self.num_mashup:].transpose(0, 1))
        # preds = torch.sigmoid(out)
        # preds = torch.mul(preds, self.invert_graph[user])
        mashups, labels, idx = batch
        out = self.forward(idx)
        out = torch.matmul(out[idx], out[self.num_mashup:].T)
        preds = torch.sigmoid(out)
        preds = torch.mul(preds, self.invert_graph[idx])
        return {
            'preds': preds,
            'targets': labels.long()
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
        return [optimizer], [scheduler]

    def device(self, value):
        self._device = value
