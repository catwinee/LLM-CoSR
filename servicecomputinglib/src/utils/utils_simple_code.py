import logging
import os
import pickle
import warnings
from typing import List, Sequence

import numpy as np
import torch
import random

import pytorch_lightning as pl
import rich.syntax
import rich.tree
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import torch.nn as nn
from itertools import combinations
import json


def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def extras(config: DictConfig) -> None:
    """A couple of optional utilities, controlled by main config file:
    - disabling warnings
    - easier access to debug mode
    - forcing debug friendly configuration

    Modifies DictConfig in place.

    Args:
        config (DictConfig): Configuration composed by Hydra.
    """

    log = get_logger()

    # enable adding new keys to config
    OmegaConf.set_struct(config, False)

    # disable python warnings if <config.ignore_warnings=True>
    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # set <config.trainer.fast_dev_run=True> if <config.debug=True>
    if config.get("debug"):
        log.info("Running in debug mode! <config.debug=True>")
        config.trainer.fast_dev_run = True

    # force debugger friendly configuration if <config.trainer.fast_dev_run=True>
    if config.trainer.get("fast_dev_run"):
        log.info("Forcing debugger friendly configuration! <config.trainer.fast_dev_run=True>")
        # Debuggers don't like GPUs or multiprocessing
        if config.trainer.get("gpus"):
            config.trainer.gpus = 0
        if config.datamodule.get("pin_memory"):
            config.datamodule.pin_memory = False
        if config.datamodule.get("num_workers"):
            config.datamodule.num_workers = 0

    # disable adding new keys to config
    OmegaConf.set_struct(config, True)


@rank_zero_only
def print_config(
    config: DictConfig,
    fields: Sequence[str] = (
        "trainer",
        "model",
        "datamodule",
        "callbacks",
        "logger",
        "seed",
    ),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    with open("config_tree.txt", "w") as fp:
        rich.print(tree, file=fp)


def empty(*args, **kwargs):
    pass


@rank_zero_only
def log_hyperparameters(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.Logger],
) -> None:
    """This method controls which parameters from Hydra config are saved by Lightning loggers.

    Additionaly saves:
        - number of trainable model parameters
    """

    hparams = {}

    # choose which parts of hydra config will be saved to loggers
    hparams["trainer"] = config["trainer"]
    hparams["model"] = config["model"]
    hparams["datamodule"] = config["datamodule"]
    if "seed" in config:
        hparams["seed"] = config["seed"]
    if "callbacks" in config:
        hparams["callbacks"] = config["callbacks"]

    # save number of model parameters
    hparams["model/params_total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params_trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params_not_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)

    # disable logging any more hyperparameters for all loggers
    # this is just a trick to prevent trainer from logging hparams of model,
    # since we already did that above
    trainer.logger.log_hyperparams = empty


def finish(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.Logger],
) -> None:
    """Makes sure everything closed properly."""

    # without this sweeps with wandb logger might crash!
    for lg in logger:
        if isinstance(lg, pl.loggers.wandb.WandbLogger):
            import wandb

            wandb.finish()

def random_walk(graph, start_node, steps):
    current_node = start_node
    node_sequence = [current_node]

    for _ in range(steps):
        if current_node in graph:
            neighbors = graph[current_node]
        else:
            neighbors = []
        if len(neighbors) > 0 and random.randint(1, 5) < 4:
            current_node = random.choice(neighbors)
            node_sequence.append(current_node)
        else:
            node_sequence.append(current_node)
    return node_sequence

def co_occurrence_frequency(matrix):
    m, n = matrix.shape
    co_occurrence_matrix = np.zeros((n, n), dtype=int)

    for row in matrix:
        non_zero_indices = np.where(row != 0)[0]
        for i in range(len(non_zero_indices)):
            for j in range(i+1, len(non_zero_indices)):
                co_occurrence_matrix[non_zero_indices[i], non_zero_indices[j]] += 1
                co_occurrence_matrix[non_zero_indices[j], non_zero_indices[i]] += 1
    row_sums = co_occurrence_matrix.sum(axis=1)
    co_occurrence_matrix = co_occurrence_matrix / (row_sums[:, np.newaxis]+0.01)
    np.fill_diagonal(co_occurrence_matrix, 1)
    return torch.tensor(co_occurrence_matrix).to(torch.float32)


def cluster_pic_show(data, cluster=6):
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data)
    # 使用t-SNE降维到二维
    tsne = TSNE(n_components=2)
    data_tsne = tsne.fit_transform(data)
    # 使用K-means对降维后的数据进行聚类
    kmeans = KMeans(n_clusters=cluster)
    labels = kmeans.fit_predict(data_tsne)

    # 绘制聚类结果
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(data_pca[:, 0], data_pca[:, 1], c=labels, cmap='viridis', alpha=0.5)
    plt.title('PCA')
    plt.subplot(1, 2, 2)
    plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=labels, cmap='viridis', alpha=0.5)
    plt.title('t-SNE')
    # plt.xlabel('Dimension 1')
    # plt.ylabel('Dimension 2')
    plt.show()


def initialize_weights(m, method='uniform'):
    if isinstance(m, nn.Linear):
        if method == 'uniform':
            # 1. 均匀分布初始化
            nn.init.uniform_(m.weight, -0.1, 0.1)
            nn.init.uniform_(m.bias, -0.1, 0.1)
        elif method == 'normal':
            # 2. 正态分布初始化
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            nn.init.normal_(m.bias, mean=0.0, std=0.02)
        elif method == 'xavier':
            # 3. Xavier初始化
            nn.init.xavier_uniform_(m.weight)
            nn.init.xavier_normal_(m.weight)
        elif method == 'he':
            # 4. He初始化
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        elif method == 'constant':
            # 5. 常数初始化
            nn.init.constant_(m.weight, 0.1)
            nn.init.constant_(m.bias, 0.1)
        elif method == 'zero':
            # 6. 零初始化
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        elif method == 'orthogonal':
            # 7. 正交初始化
            nn.init.orthogonal_(m.weight)
            nn.init.orthogonal_(m.bias)
        elif method == 'sparse':
            # 8. 稀疏初始化
            nn.init.sparse_(m.weight, sparsity=0.1)
            nn.init.sparse_(m.bias, sparsity=0.1)
        else:
            raise ValueError('Invalid initialization method: %s' % method)


def create_graph_and_invert_graph(num_mashup, num_api, edges, device):
    # 总的节点数
    num_nodes = num_mashup + num_api

    # 初始化稀疏矩阵的索引列表和值列表
    indices = []
    invert_indices = []

    # 遍历边列表，填充稀疏矩阵的索引和值
    for edge in edges.T:
        i, j = edge[0].item(), edge[1].item()
        indices.append([i, j])
        indices.append([j, i])
        invert_indices.append([i, j])
        invert_indices.append([j, i])

    # 构建稀疏矩阵
    indices = torch.tensor(indices, dtype=torch.long).t().to(device)
    values = torch.ones(indices.size(1), dtype=torch.float32).to(device)

    graph = torch.sparse.FloatTensor(indices, values, torch.Size([num_nodes, num_nodes]))

    # 构建稀疏逆矩阵
    invert_indices = torch.tensor(invert_indices, dtype=torch.long).t().to(device)
    invert_values = torch.zeros(invert_indices.size(1), dtype=torch.float32).to(device)

    invert_graph = torch.sparse.FloatTensor(invert_indices, invert_values, torch.Size([num_nodes, num_nodes]))

    # 稀疏矩阵转为稠密矩阵，方便后续操作
    dense_graph = graph.to_dense()
    dense_invert_graph = 1 - invert_graph.to_dense()

    return dense_graph, dense_invert_graph


import numpy as np


def get_positional_encoding(max_position, d_model):
    position = np.arange(max_position)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    pos_encoding = np.zeros((max_position, d_model))
    pos_encoding[:, 0::2] = np.sin(position * div_term)
    pos_encoding[:, 1::2] = np.cos(position * div_term)

    return torch.tensor(pos_encoding)


def response_parsing(response, dataset='Mashup'):

    def find_strings_in_target(str1, target_str):
        if str1 in target_str:
            start = target_str.find(str1)
            return (start)
        return 1000000

    api_list = []
    if dataset == 'Mashup':
        with open('rubbish/data/Mashup/preprocessed_data/api_names', 'rb') as f:
            api_names = pickle.load(f)
        api_len_names = sorted(api_names, key=len, reverse=True)
        for api in api_len_names:
            start = find_strings_in_target(api, response)
            index = api_names.index(api)
            api_list.append((index, start))
        sorted_list = sorted(api_list, key=lambda x: x[1])
        top_10_indexes = [item[0] for item in sorted_list[:10]]
    else:
        top_10_indexes = None
    return top_10_indexes

# Write a function that tells me if a number is prime
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True


class SKLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)


def find_highest_rank_item(items):
    # 假设等级范围为1到5，其中5为最高等级
    max_rank = 100
    max_rank_item = None
    for item in items:
        rank = item['rank']  # 假设每个item是一个字典，并且有一个'rank'键表示它的等级
        if rank < max_rank:
            max_rank = rank
            max_rank_item = item
            if max_rank == 0:  # 提前停止搜索，如果已经找到了最高等级5
                break
    return max_rank_item


def jaccard_similarity(set1, set2):
    # 计算交集
    intersection = len(set1.intersection(set2)) - len(set1 - set2) - len(set2 - set1)
    # 计算并集
    union = len(set1.union(set2))
    # 计算Jaccard相似系数
    similarity = intersection / union
    return similarity


class LambdaRankLoss(nn.Module):
    def  __init__(self, relavance_matrix, api_tag_embedding, tag_set_all, mode):
        super(LambdaRankLoss, self).__init__()
        with open('data/Mashup/label/raw/all.json', "r") as f:
            tag_set1 = json.load(f)
        tag_set = [tag_set1[i]['manual'] for i in tag_set1]
        self.tag_num = len(tag_set_all)
        self.api_tag_embedding = api_tag_embedding
        self.num_api = len(tag_set)
        self.mode = mode
        # self.rank = torch.zeros((self.num_api, self.num_api), dtype=torch.float32)
        self.union_matrix = torch.zeros((self.num_api, self.num_api, len(tag_set_all)), dtype=torch.float32)
        for i,j in combinations(range(len(tag_set)), 2):
            self.union_matrix[i, j] = torch.min(self.api_tag_embedding[i], self.api_tag_embedding[j])
        self.related_matrix = relavance_matrix
        # for i, j in combinations(range(len(tag_set)), 2):
        #     self.rank[i, j] = len(set(tag_set[i]).intersection(set(tag_set[j]))) - len(set(tag_set[i]) - set(tag_set[j])) - len(set(tag_set[j]) - set(tag_set[i]))
        #     self.rank[i, j] = self.rank[j, i]

    def forward(self, y_scores, y_preds, M):
        """
        y_score: 不是预测的标签，而是
        y_true: 真实标签
        qid: 查询ID
        """
        loss = torch.tensor(0.0).to(y_scores[0].device)
        for y_score, y_pred, m in zip(y_scores, y_preds, M):
            lambda_val = self.compute_lambda(y_score, torch.tensor(y_pred).to(y_score.device), m, self.mode)
            # loss[count] = torch.sum(lambda_val * y_true)
            if lambda_val.isnan() or lambda_val.isinf():
                lambda_val = self.compute_lambda(y_score, torch.tensor(y_pred).to(y_score.device), m, self.mode)
            if loss.isnan() or loss.isinf():
                lambda_val = self.compute_lambda(y_score, torch.tensor(y_pred).to(y_score.device), m, self.mode)
            loss += lambda_val
        return loss/len(y_scores)

    def compute_lambda(self, y_score, y_pred, M, mode):

        mode = mode.split('+')
        device = y_score.device

        tag_set = self.api_tag_embedding[y_pred.tolist()]
        set_union_list, _ = torch.cummax(tag_set, dim=0)
        len_set = len(set_union_list)
        log_values = torch.log2(torch.arange(2, y_pred.shape[0] + 2, dtype=torch.float32)).to(device)

        n = len(set_union_list)

        Q = torch.zeros((n, n, n, self.tag_num))
        for k in range(n):
            for i in range(k+1):
                for j in range(k+1, n):
                    Q[k, i, j] = torch.max(torch.clamp(set_union_list[k]-tag_set[i], min=0), tag_set[j])
        # Q = torch.zeros((n, n, n, self.tag_num))
        # target_set = torch.randn((n, self.tag_num))

        # 扩展 set_union_list 以适应不同维度
        # expanded_set_union_list = set_union_list.unsqueeze(1).unsqueeze(2).expand(-1, n, n, -1).to(device)
        # expanded_target_set_i = tag_set.unsqueeze(0).unsqueeze(2).expand(n, n, n, -1).to(device)
        # expanded_target_set_j = tag_set.unsqueeze(0).unsqueeze(1).expand(n, n, n, -1).to(device)
        #
        # # 计算 Q 矩阵的值
        # mask = torch.arange(n).unsqueeze(1).unsqueeze(2) <= torch.arange(n).unsqueeze(0).unsqueeze(2)
        # mask = mask & (torch.arange(n).unsqueeze(1) >= torch.arange(n).unsqueeze(0))
        # mask = mask.unsqueeze(-1).expand(-1, -1, -1, self.tag_num).to(device)
        #
        # Q = torch.where(
        #     mask,
        #     torch.max(torch.clamp(expanded_set_union_list - expanded_target_set_i, min=0), expanded_target_set_j),
        #     expanded_set_union_list
        # ).to(device)

        Z1 = torch.zeros((len(set_union_list), len(set_union_list))).to(device)
        Z2 = torch.zeros((len(set_union_list), len(set_union_list))).to(device)
        Z3 = torch.zeros((len(set_union_list), len(set_union_list))).to(device)
        Z4 = torch.zeros((len(set_union_list), len(set_union_list))).to(device)
        min_k = torch.tensor(1.0).to(device)

        if 'cover' in mode:
            # # 计算 M 和 set_union_list 的差异，并 clamp 到 0 以上
            # M_diff = torch.clamp(M.unsqueeze(0) - set_union_list.unsqueeze(1), min=0).to(device)
            # # 计算 M 和 Q 的差异，并 clamp 到 0 以上
            # Q_diff = torch.clamp(M.unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device) - Q, min=0)
            # # 计算非零元素的数量
            # M_count_nonzero = M_diff.count_nonzero(dim=2)
            # Q_count_nonzero = Q_diff.count_nonzero(dim=3)
            # # 计算归一化的差值
            # diff_normalized = (M_count_nonzero.unsqueeze(2) - Q_count_nonzero) / (M.count_nonzero()+torch.tensor(1.0)).to(device)
            # # 生成索引矩阵
            # index_tensor = torch.arange(len(set_union_list))
            # # 生成每对 (i, j) 的 k 的范围掩码
            # # k_mask = (index_tensor.unsqueeze(1) >= index_tensor.unsqueeze(0)).float().unsqueeze(-1).to(device)
            # # 应用 log_values 和掩码，计算 Z
            # log_values_matrix = log_values.unsqueeze(0).unsqueeze(0).expand_as(diff_normalized).to(device)
            # Z1 = (diff_normalized * log_values_matrix).sum(dim=2)
            # # 最终对 Z 进行归一化
            # distance = (index_tensor.unsqueeze(1) - index_tensor.unsqueeze(0)).float().abs().to(device)
            # Z1 /= torch.where(distance == 0, torch.ones_like(distance), distance)
            for i in range(len(set_union_list)):
                for j in range(len(set_union_list)):
                    if j > i:
                        for k in range(i, j):
                            Z1[i, j] += (torch.count_nonzero(torch.clamp(M - set_union_list[k], min=0))
                                 - torch.count_nonzero(torch.clamp(M - Q[k, i, j], min=0))) / ((torch.count_nonzero(M)+1) * log_values[k])
                        Z1[i, j] = Z1[i, j] / (j - i + 1)
                        # if Z1[i, j].isnan() or Z1[i, j].isinf():
                        #     Z1[i, j] = Z1[i, j] / (j - i + 1)
        if 'save' in mode:
            # # 计算 M 和 set_union_list 的差异，并 clamp 到 0 以上
            # M_diff = torch.clamp(M.unsqueeze(0) - set_union_list.unsqueeze(1), min=0).to(device)
            # # 计算 M 和 Q 的差异，并 clamp 到 0 以上
            # Q_diff = torch.clamp(Q - M.unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device), min=0)
            # # 计算非零元素的数量
            # M_count_nonzero = M_diff.count_nonzero(dim=2)
            # Q_count_nonzero = Q_diff.count_nonzero(dim=3)
            # # 计算归一化的差值
            # diff_normalized = (M_count_nonzero.unsqueeze(2) - Q_count_nonzero) / (M.count_nonzero()+torch.tensor(1.0)).to(device)
            # # 生成索引矩阵
            # index_tensor = torch.arange(len(set_union_list))
            # # 生成每对 (i, j) 的 k 的范围掩码
            # # k_mask = (index_tensor.unsqueeze(1) >= index_tensor.unsqueeze(0)).float().unsqueeze(-1).to(device)
            # # 应用 log_values 和掩码，计算 Z
            # log_values_matrix = log_values.unsqueeze(0).unsqueeze(0).expand_as(diff_normalized).to(device)
            # Z2 = (diff_normalized * log_values_matrix).sum(dim=2)
            # # 最终对 Z 进行归一化
            # distance = (index_tensor.unsqueeze(1) - index_tensor.unsqueeze(0)).float().abs().to(device)
            # Z2 /= torch.where(distance == 0, torch.ones_like(distance), distance)

            for i in range(len(set_union_list)):
                for j in range(len(set_union_list)):
                    if j > i:
                        for k in range(i, j):
                            Z2[i, j] += (torch.count_nonzero(torch.clamp(set_union_list[k] - M, min=0)) / torch.count_nonzero(set_union_list[k] + 1)
                                - torch.count_nonzero(torch.clamp(Q[k, i, j] - M, min=0)) / (torch.count_nonzero(Q[k, i, j]) + 1)) / log_values[k]
                        Z2[i, j] = Z2[i, j] / (j - i + 1)
        if 'dependence' in mode: #
            # y_true_nonzero = torch.count_nonzero(tag_set, dim=1).to(device)
            # set_union_nonzero = torch.count_nonzero(set_union_list, dim=1).to(device)
            # Q_nonzero = torch.count_nonzero(Q, dim=-1).to(device)
            # # 生成所有可能的 (i, j, k) 的组合
            # i_range = torch.arange(len(set_union_list)).unsqueeze(1).expand(len(set_union_list), len(set_union_list))
            # j_range = torch.arange(len(set_union_list)).unsqueeze(0).expand(len(set_union_list), len(set_union_list))
            # k_range = torch.arange(len(set_union_list))
            # # 计算差值矩阵
            # diff_y_true = (y_true_nonzero[j_range] - y_true_nonzero[i_range]).unsqueeze(-1)
            # diff_set_union = set_union_nonzero.unsqueeze(0).unsqueeze(0) - Q_nonzero
            # # 计算 k 的掩码，确定哪些 k 值在 i 和 j 之间
            # k_mask = ((k_range.unsqueeze(0).unsqueeze(0) >= torch.min(i_range, j_range).unsqueeze(-1)) & (k_range.unsqueeze(0).unsqueeze(0) < torch.max(i_range, j_range).unsqueeze(-1))).to(device)
            # # 计算归一化的差值并乘以 log_values
            # k_normalized = (diff_y_true - diff_set_union) / (torch.count_nonzero(tag_set[k_range], dim=1)+torch.tensor(1.0)).to(device)
            # weighted_diff = k_normalized * log_values.unsqueeze(0).unsqueeze(0)
            # # 应用掩码并计算 Z 矩阵
            # Z3 = torch.sum(weighted_diff * k_mask, dim=-1)
            #
            # # 最终对 Z 进行归一化
            # distance = (i_range - j_range).float().abs().to(device)
            # Z3 /= torch.where(distance == 0, torch.ones_like(distance), distance)

            for i in range(len(set_union_list)):
                for j in range(len(set_union_list)):
                    if j > i:
                        for k in range(i, j):
                            Z3[i, j] += (-torch.count_nonzero(set_union_list[k]) / (torch.count_nonzero(tag_set[:k])+1)
                                        + torch.count_nonzero(Q[k, i, j]) / (torch.count_nonzero(tag_set[:i])
                                        + torch.count_nonzero(tag_set[j]) + torch.count_nonzero(tag_set[i+1, k] + 1))) / log_values[k]
                        Z3[i, j] = Z3[i, j] / (j - i + 1)
        if 'tolerance' in mode:
            selected_matrix = self.union_matrix[y_pred.to(self.union_matrix.device)][:, y_pred.to(self.union_matrix.device), :]
            # max_across_rows, _ = selected_matrix.cummax(dim=0)  # 行最大值
            # max_across_cols, _ = max_across_rows.cummax(dim=1)  # 列最大值
            # # 初始化最终结果向量
            # max_final1 = torch.zeros((len(set_union_list), self.union_matrix.shape[2]))
            # # 从预计算的累计最大值中提取所需的最大值
            # for k in range(len(set_union_list)):
            #     max_final1[k] = max_across_cols[k, k]

            for i in range(len(set_union_list)):
                for j in range(i, len(set_union_list)):
                    if j > i:
                        for k in range(i, j):
                            selected_index = torch.tensor([q for q in range(k) if i != q]+[j])
                            max_final1 = selected_matrix[selected_index][:, selected_index, :]
                            max_final1 = torch.any(max_final1, dim=(0, 1))
                            max_final2 = selected_matrix[:k]
                            max_final2 = torch.any(max_final2, dim=(0, 1))
                            # max_final2 = torch.max(selected_matrix[selected_index, selected_index].view(-1, self.union_matrix.shape[2]), dim=0).values
                            Z4[i, j] += (torch.count_nonzero(torch.min(M, max_final2))
                                         - torch.count_nonzero(torch.min(M, max_final1))) / ((torch.count_nonzero(M) + 1) * log_values[k])
                        Z4[i, j] /= (j - i + 1)
            Z4 = torch.add(Z4.T, Z4)
        if 'count' in mode:
            condition = (M.unsqueeze(0) >= set_union_list).all(dim=1)
            first_true_index = (condition.int() == 1).nonzero().flatten()
            if first_true_index.nelement() == 0:
                first_true_index = torch.tensor(len(condition))
            else:
                first_true_index = first_true_index[0]
            # 找到第一个满足条件的 k
            min_k = 1 / torch.log2(torch.tensor(2) + first_true_index)


        Z = (Z1 + Z2 + Z3 + Z4) * min_k # TODO:min_k的结合机制还要在讨论


        # y_true_cpu = y_true.to(self.related_matrix.device)
        y_score_col = y_score[:, None]
        score_diff_matrix = y_score_col - y_score_col.T
        lambda1 = 1 / (1 + torch.exp(score_diff_matrix))

        # 以下是最顶层的聚合
        # delta_Z = torch.where(Z<0, -Z, Z)
        # lambda_add_minus = create_custom_tensor(n_samples).to(device)
        samples_score = lambda1 * Z

        samples_score = (torch.sum(samples_score)/torch.tensor(samples_score.shape[0]+1)).to(device)

        return torch.max(samples_score, torch.tensor(0.0).to(device) * samples_score)


def create_custom_tensor(n):
    # 创建一个 n x n 的零矩阵
    tensor = torch.zeros((n, n), dtype=torch.float32)

    # 下三角区域（不包括对角线）设置为 -1
    # torch.tril_indices 返回下三角的索引（不包括对角线）
    lower_indices = torch.tril_indices(n, n, 1)
    tensor[lower_indices[0], lower_indices[1]] = -1

    # 上三角区域（不包括对角线）设置为 1
    # torch.triu_indices 返回上三角的索引（不包括对角线）
    upper_indices = torch.triu_indices(n, n, -1)
    tensor[upper_indices[0], upper_indices[1]] = 1

    return tensor


def compute_Q_matrix(R):
    n = R.size(0)

    # 扩展 R 的维度
    R_expanded = R.unsqueeze(0).expand(n, n)

    # 计算上三角部分和
    upper_sum = torch.triu(R_expanded, diagonal=1)
    upper_sum = torch.cumsum(upper_sum, dim=1)

    # 计算下三角部分和
    lower_sum = torch.tril(R_expanded, diagonal=-1)
    lower_sum = torch.flip(torch.cumsum(torch.flip(lower_sum, dims=[1]), dim=1), dims=[1])

    # 两部分相加得到 Q 矩阵
    Q_matrix = upper_sum + lower_sum

    return Q_matrix


if __name__ == '__main__':
    y_scores = torch.tensor([0.5, 0.4, 0.3, 0.2, 0.1], dtype=torch.float32).unsqueeze(0)
    y_preds = torch.tensor([1, 4, 3, 2, 5], dtype=torch.int32).unsqueeze(0)
    y_trues = torch.tensor([1, 2, 3], dtype=torch.int32).unsqueeze(0)
    tag_set = '../../data/Mashup/label/related_matrix.pt'
    loss = LambdaRankLoss(tag_set)
    a = loss(y_scores, y_preds, y_trues, 3)
    print(a)