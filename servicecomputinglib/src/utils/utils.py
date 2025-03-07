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

import torch.nn.functional as F
from sklearn.neighbors import kneighbors_graph
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli


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
            if m.bias is not None:
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
                    # a = torch.max(tag_set[:i-1], dim=0).values
                    # b = torch.max(tag_set[i+1:k], dim=0).values
                    # c = torch.cat((a, b, tag_set[j].unsqueeze(0)), dim=0)
                    # c = torch.max(c, dim=0).values
                    Q[k, i, j] = torch.max(torch.cat((tag_set[:i-1], tag_set[i+1:k], tag_set[j].unsqueeze(0)), dim=0), dim=0).values
                    # Q[k, i, j] = torch.max(torch.clamp(set_union_list[k]-tag_set[i], min=0), tag_set[j])
        # Q = torch.zeros((n, n, n, self.tag_num))
        # target_set = torch.randn((n, self.tag_num))
        Q = Q.to(device)

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
            # for i in range(len(set_union_list)):

            Z1 = torch.zeros((len(set_union_list), len(set_union_list))).to(device)
            # 使用 cummax 得到 set_union_list
            set_union_list, _ = torch.cummax(tag_set, dim=0)
            set_union_list = set_union_list.to(device)
            # 计算 M 的非零元素计数（用于归一化）
            M_nonzero_count = torch.count_nonzero(M) + 1
            # 扩展 M 以支持广播计算
            M_expanded = M.unsqueeze(0).to(device)  # M 形状变为 (1, 1, n, self.tag_num)
            # 循环计算 i 和 j 的值
            for i in range(len(set_union_list)):
                for j in range(i + 1, len(set_union_list)):
                    indices = torch.arange(i, j)
                    set_union_list_k = set_union_list[indices]
                    M_clamped = torch.clamp(M_expanded - set_union_list_k, min=0)
                    Q_clamped = torch.clamp(M_expanded - Q[indices, i, j], min=0)
                    M_nonzero = torch.count_nonzero(M_clamped, dim=1)
                    Q_nonzero = torch.count_nonzero(Q_clamped, dim=1)
                    Z1[i, j] = torch.sum((M_nonzero - Q_nonzero) / (M_nonzero_count.to(device) * log_values[indices]))

        if 'save' in mode:
            for i in range(len(set_union_list)):
                for j in range(i+1, len(set_union_list)):
                    for k in range(i, j):
                        Z2[i, j] += (torch.count_nonzero(torch.clamp(set_union_list[k].to(device) - M.to(device), min=0)) / (torch.count_nonzero(set_union_list[k].to(device)) + 1)
                            - torch.count_nonzero(torch.clamp(Q[k, i, j] - M.to(device), min=0)) / (torch.count_nonzero(Q[k, i, j]) + 1)) / log_values[k]
                    Z2[i, j] = Z2[i, j] / (j - i + 1)
        if 'dependence' in mode: #
            for i in range(len(set_union_list)):
                for j in range(i+1, len(set_union_list)):
                    for k in range(i, j):
                        Z3[i, j] += (-torch.count_nonzero(set_union_list[k]) / (torch.count_nonzero(tag_set[:k])+1)
                                    + torch.count_nonzero(Q[k, i, j]) / (torch.count_nonzero(tag_set[:i])
                                    + torch.count_nonzero(tag_set[j]) + torch.count_nonzero(tag_set[i+1, k] + 1))) / log_values[k]
                    Z3[i, j] = Z3[i, j] / (j - i + 1)
        if 'tolerance' in mode:
            selected_matrix = self.union_matrix[y_pred.to(self.union_matrix.device)][:, y_pred.to(self.union_matrix.device), :]
            # if torch.sum(selected_matrix) != 0:
            #     print(1)
            for i in range(len(set_union_list)):
                for j in range(i+1, len(set_union_list)):
                    for k in range(i, j):
                        selected_index = torch.tensor([q for q in range(k+1) if i != q]+[j])
                        max_final1 = selected_matrix[selected_index][:, selected_index, :]
                        max_final1 = torch.any(max_final1, dim=(0, 1))
                        max_final2 = selected_matrix[:k+1]
                        max_final2 = torch.any(max_final2, dim=(0, 1))
                        # if not torch.equal(max_final2, max_final1):
                        #     print(1)
                        # if (torch.count_nonzero(torch.min(M, max_final2)) - torch.count_nonzero(torch.min(M, max_final1))) > 0:
                        #     print(1)
                        Z4[i, j] += (torch.count_nonzero(torch.min(M, max_final2))
                                     - torch.count_nonzero(torch.min(M, max_final1))) / ((torch.count_nonzero(M) + 1) * log_values[k])
                    # Z4[i, j] /= (j - i + 1)
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


        Z = (Z1 + Z2 + Z3 + Z4)# TODO:min_k的结合机制还要在讨论


        # y_true_cpu = y_true.to(self.related_matrix.device)
        y_score_col = y_score[:, None]
        score_diff_matrix = y_score_col - y_score_col.T
        lambda1 = 1 / (1 + torch.exp(score_diff_matrix))

        # 以下是最顶层的聚合
        # delta_Z = torch.where(Z<0, -Z, Z)
        # lambda_add_minus = create_custom_tensor(n_samples).to(device)
        samples_score = lambda1 * Z
        # samples_score = torch.max(samples_score, torch.tensor(0.0).to(device) * samples_score)
        samples_score = (torch.sum(samples_score)/torch.tensor(samples_score.shape[0]+1)).to(device)

        # return torch.max(samples_score, torch.tensor(0.0).to(device) * samples_score)

        return samples_score

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

def select_indices(tensor1, n, q):
    n_rows = tensor1.size(0)
    q_indices = list(range(q))  # 用于填补的索引
    result = []

    # 找到每一行的最大值
    row_max_values, _ = torch.max(tensor1, dim=1, keepdim=True)

    # 使用 torch.where 找到每一行中最大值的位置
    max_indices = torch.where(tensor1 == row_max_values)

    # 整理每行的索引
    max_indices_list = [[] for _ in range(n_rows)]
    for row, col in zip(max_indices[0], max_indices[1]):
        max_indices_list[row.item()].append(col.item())

    # 为每一行随机选取 n 个索引
    for indices in max_indices_list:
        if len(indices) >= n:
            # 随机选取 n 个索引
            selected_indices = random.sample(indices, n)
        else:
            # 先选择所有最大值的索引
            selected_indices = indices
            # 从 q_indices 中随机选取补足到 n 个
            # selected_indices += random.sample(q_indices, n - len(indices))

        result.append(selected_indices)

    return result

def select_negative_samples(label: torch.Tensor, negative_sample_ratio: int = 5):
    r"""select negative samples in training stage.

    Args:
        label (List[np.ndarray]): Label indicating the APIs called by mashup.
        negative_sample_ratio (int): Ratio of negative to positive in the training stage. (default: :obj:`5`)

    Returns:
        indices of positive samples, indices of negative samples, indices of all samples, and new label.
    """
    num_candidate = label.size(0)
    positive_idx = label.nonzero(as_tuple=True)[0]
    if len(positive_idx) > 0:
        positive_idx = positive_idx.cpu().numpy()
    else:
        positive_idx = torch.tensor([0], dtype=torch.int64)
    negative_idx = np.random.choice(np.delete(np.arange(num_candidate), positive_idx),
                                    size=negative_sample_ratio * len(positive_idx), replace=False)
    sample_idx = np.concatenate((positive_idx, negative_idx), axis=None)
    label_new = torch.tensor([1] * len(positive_idx) + [0] * len(negative_idx), dtype=torch.float32)
    return positive_idx, negative_idx, sample_idx, label_new.cuda()
class KNNSparsify:
    def __init__(self, k, discrete=False, self_loop=True):
        super(KNNSparsify, self).__init__()
        self.k = k
        self.discrete = discrete
        self.self_loop = self_loop

    def __call__(self, adj):
        _, indices = adj.topk(k=int(self.k + 1), dim=-1)
        assert torch.max(indices) < adj.shape[1]
        mask = torch.zeros(adj.shape).to(adj.device)
        mask[torch.arange(adj.shape[0]).view(-1, 1), indices] = 1.

        mask.requires_grad = False
        if self.discrete:
            sparse_adj = mask.to(torch.float)
        else:
            sparse_adj = adj * mask

        if not self.self_loop:
            sparse_adj.fill_diagonal_(0)
        return sparse_adj


class ThresholdSparsify:
    def __init__(self, threshold):
        super(ThresholdSparsify, self).__init__()
        self.threshold = threshold

    def __call__(self, adj):
        return torch.where(adj < self.threshold, torch.zeros_like(adj), adj)


class ProbabilitySparsify:
    def __init__(self, temperature=0.1):
        self.temperature = temperature

    def __call__(self, prob):
        prob = torch.clamp(prob, 0.01, 0.99)
        adj = RelaxedBernoulli(temperature=torch.Tensor([self.temperature]).to(prob.device),
                               probs=prob).rsample()
        eps = 0.5
        mask = (adj > eps).detach().float()
        adj = adj * mask + 0.0 * (1 - mask)
        return adj


class Discretize:
    def __init__(self):
        super(Discretize, self).__init__()

    def __call__(self, adj):
        adj[adj != 0] = 1.0
        return adj


class AddEye:
    def __init__(self):
        super(AddEye, self).__init__()

    def __call__(self, adj):
        adj += torch.eye(adj.shape[0]).to(adj.device)
        return adj


class LinearTransform:
    def __init__(self, alpha):
        super(LinearTransform, self).__init__()
        self.alpha = alpha

    def __call__(self, adj):
        adj = adj * self.alpha - self.alpha
        return adj


class NonLinearize:
    def __init__(self, non_linearity='relu', alpha=1.0):
        super(NonLinearize, self).__init__()
        self.non_linearity = non_linearity
        self.alpha = alpha

    def __call__(self, adj):
        if self.non_linearity == 'elu':
            return F.elu(adj) + 1
        elif self.non_linearity == 'relu':
            return F.relu(adj)
        elif self.non_linearity == 'none':
            return adj
        else:
            raise NameError('We dont support the non-linearity yet')


class Symmetrize:
    def __init__(self):
        super(Symmetrize, self).__init__()

    def __call__(self, adj):
        return (adj + adj.T) / 2


class Normalize:
    def __init__(self, mode='sym', eos=1e-10):
        super(Normalize, self).__init__()
        self.mode = mode
        self.EOS = eos

    def __call__(self, adj):
        if self.mode == "sym":
            inv_sqrt_degree = 1. / (torch.sqrt(adj.sum(dim=1, keepdim=False)) + self.EOS)
            return inv_sqrt_degree[:, None] * adj * inv_sqrt_degree[None, :]
        elif self.mode == "row":
            inv_degree = 1. / (adj.sum(dim=1, keepdim=False) + self.EOS)
            return inv_degree[:, None] * adj
        elif self.mode == "row_softmax":
            return F.softmax(adj, dim=1)
        elif self.mode == "row_softmax_sparse":
            return torch.sparse.softmax(adj.to_sparse(), dim=1).to_dense()
        else:
            raise Exception('We dont support the normalization mode')


import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def SetSeed(seed):
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def get_random_mask(features, r, scale, dataset):
    nones = torch.sum(features > 0.0).float()
    nzeros = features.shape[0] * features.shape[1] - nones
    pzeros = nones / nzeros / r * scale

    probs = torch.zeros(features.shape).to(device)
    probs[features == 0.0] = pzeros
    probs[features > 0.0] = 1 / r

    mask = torch.bernoulli(probs)

    return mask


def top_k(raw_graph, k):
    _, indices = raw_graph.topk(k=int(k), dim=-1)

    mask = torch.zeros(raw_graph.shape).to(device)
    mask[torch.arange(raw_graph.shape[0]).view(-1, 1), indices] = 1.

    mask.requires_grad = False
    sparse_graph = raw_graph * mask

    return sparse_graph


def knn_fast(X, k, b):
    X = torch.nn.functional.normalize(X, dim=1, p=2)

    index = 0
    values = torch.zeros(X.shape[0] * (k + 1)).cuda()
    rows = torch.zeros(X.shape[0] * (k + 1)).cuda()
    cols = torch.zeros(X.shape[0] * (k + 1)).cuda()
    norm_row = torch.zeros(X.shape[0]).cuda()
    norm_col = torch.zeros(X.shape[0]).cuda()

    while index < X.shape[0]:
        if (index + b) > (X.shape[0]):
            end = X.shape[0]
        else:
            end = index + b

        sub_tensor = X[index:index + b]
        similarities = torch.mm(sub_tensor, X.t())
        vals, inds = similarities.topk(k=k + 1, dim=-1)

        values[index * (k + 1):(end) * (k + 1)] = vals.view(-1)
        cols[index * (k + 1):(end) * (k + 1)] = inds.view(-1)
        rows[index * (k + 1):(end) * (k + 1)] = torch.arange(index, end).view(-1, 1).repeat(1, k + 1).view(-1)
        norm_row[index: end] = torch.sum(vals, dim=1)
        vals = vals.float()
        norm_col.index_add_(-1, inds.view(-1), vals.view(-1))
        index += b

    norm = norm_row + norm_col
    rows = rows.long()
    cols = cols.long()
    values *= (torch.pow(norm[rows], -0.5) * torch.pow(norm[cols], -0.5))

    return rows, cols, values


def get_homophily(adj, labels, sparse):
    if sparse == 1:
        src, dst = adj.edges()
    else:
        src, dst = adj.detach().nonzero().t()
    homophily_ratio = 1.0 * torch.sum((labels[src] == labels[dst])) / src.shape[0]

    return homophily_ratio

def cal_similarity_graph(node_embeddings, right_node_embedding=None):
    if right_node_embedding is None:
        similarity_graph = torch.mm(node_embeddings, node_embeddings.t())
    else:
        similarity_graph = torch.mm(node_embeddings, right_node_embedding.t())
    return similarity_graph


class InnerProductSimilarity:
    def __init__(self):
        super(InnerProductSimilarity, self).__init__()

    def __call__(self, embeddings, right_embeddings=None):
        similarities = cal_similarity_graph(embeddings, right_embeddings)
        return similarities


class CosineSimilarity:
    def __init__(self):
        super(CosineSimilarity, self).__init__()

    def __call__(self, embeddings, right_embeddings=None):
        if right_embeddings is None:
            embeddings = F.normalize(embeddings, dim=1, p=2)
            similarities = cal_similarity_graph(embeddings)
        else:
            embeddings = F.normalize(embeddings, dim=1, p=2)
            right_embeddings = F.normalize(right_embeddings, dim=1, p=2)
            similarities = cal_similarity_graph(embeddings, right_embeddings)
        return similarities


class WeightedCosine(nn.Module):
    def __init__(self, input_size, num_pers):
        super().__init__()
        self.weight_tensor = torch.Tensor(num_pers, input_size)
        self.weight_tensor = nn.Parameter(nn.init.xavier_uniform_(self.weight_tensor))

    def __call__(self, embeddings):
        expand_weight_tensor = self.weight_tensor.unsqueeze(1)
        if len(embeddings.shape) == 3:
            expand_weight_tensor = expand_weight_tensor.unsqueeze(1)

        embeddings_fc = embeddings.unsqueeze(0) * expand_weight_tensor
        embeddings_norm = F.normalize(embeddings_fc, p=2, dim=-1)
        attention = torch.matmul(
            embeddings_norm, embeddings_norm.transpose(-1, -2)
        ).mean(0)
        return attention


class MLPRefineSimilarity(nn.Module):
    def __init__(self, hid, dropout):
        super(MLPRefineSimilarity, self).__init__()
        self.gen_mlp = nn.Linear(2 * hid, 1)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, embeddings, v_indices):
        num_node = embeddings.shape[0]
        f1 = embeddings[v_indices[0]]
        f2 = embeddings[v_indices[1]]
        ff = torch.cat([f1, f2], dim=-1)
        temp = self.gen_mlp(self.dropout(ff)).reshape(-1)
        z_matrix = torch.sparse.FloatTensor(v_indices, temp, (num_node, num_node)).to_dense()
        return z_matrix


if __name__ == '__main__':
    y_scores = torch.tensor([0.5, 0.4, 0.3, 0.2, 0.1], dtype=torch.float32).unsqueeze(0)
    y_preds = torch.tensor([1, 4, 3, 2, 5], dtype=torch.int32).unsqueeze(0)
    y_trues = torch.tensor([1, 2, 3], dtype=torch.int32).unsqueeze(0)
    tag_set = '../../data/Mashup/label/related_matrix.pt'
    loss = LambdaRankLoss(tag_set)
    a = loss(y_scores, y_preds, y_trues, 3)
    print(a)