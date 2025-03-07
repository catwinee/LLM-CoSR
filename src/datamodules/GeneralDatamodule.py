import os
import random

import hydra
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
import json
from clearml import Dataset
from functools import reduce
import operator
from src.datamodules.components.GSATDataset import GSATDataset
import numpy as np


class GeneralDatamodule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers, train_ratio, val_ratio, gpus, n_negs=64, dataset='Mashup'):
        super().__init__()
        self.data_dir = data_dir
        if dataset == 'Mashup':
            self.api_num = 1216  # 分类数
            self.mashup_num = 6424
        elif dataset == 'Youshu':
            self.api_num = 32770
            self.mashup_num = 8039
        else:
            self.api_num = 1216
            self.mashup_num = 6424
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.val_len = 0
        self.gpus = gpus
        self.n_negs = n_negs
        self.dataset = dataset
        self.prepare_data()

    def prepare_data(self) -> None:
        train_edges = torch.load(self.data_dir + '/train_edges.emb')
        val_edges = torch.load(self.data_dir + '/val_edges.emb')
        test_edges = torch.load(self.data_dir + '/test_edges.emb')
        self.train_dataset = GSATDataset(
            train_edges, self.mashup_num, self.api_num, self.gpus, self.dataset, self.n_negs
        )
        self.val_dataset = GSATDataset(
            val_edges, self.mashup_num, self.api_num, self.gpus, self.dataset, self.n_negs
        )
        self.test_dataset = GSATDataset(
            test_edges, self.mashup_num, self.api_num, self.gpus, self.dataset, self.n_negs
        )

    def train_dataloader(self):  # 训练集
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):  # 验证数据集，是否过拟合，本身不参与训练
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def test_dataloader(self):  # 测试集
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)



