import os
from typing import Any, Optional

import pickle
import pandas as pd
import hydra
import numpy as np
import torch.optim
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
from scipy.sparse import coo_matrix, csr_matrix


from src.utils.data_processer import select_negative_samples
from src.utils.metrics import F1ScoreMetrics

from src.models.components.lightgcl_net import LightGCL_Net


class LightGCL(LightningModule):
    def __init__(
        self,
        data_dir: str,
        api_embed_path: str,
        mashup_embed_path: str,
        negative_samples_ratio: int,
        lr: float,
        weight_decay: float,
    ):
        super(LightGCL, self).__init__()
        self.save_hyperparameters()

        self.register_buffer('api_embed', torch.from_numpy(np.load(os.path.join(data_dir, api_embed_path))))
        self.register_buffer('mashup_embed', torch.from_numpy(np.load(os.path.join(data_dir, mashup_embed_path))))
        self.negative_samples_ratio = negative_samples_ratio

        invocation_path = hydra.utils.get_original_cwd() + '/data/api_mashup/train_partial_invocation_seed=12345.pkl'
        df = pd.read_pickle(invocation_path)

        all_users = df['X'].unique()
        all_items = np.unique(np.concatenate(df['Y'].values))

        user2idx = {u: idx for idx, u in enumerate(all_users)}
        item2idx = {i: idx for idx, i in enumerate(all_items)}

        rows, cols, data = [], [], []
        for _, row in df.iterrows():
            user = row['X']
            items = row['Y']
            for item in items:
                rows.append(user2idx[user])  # 用户索引作为行
                cols.append(item2idx[item])  # 物品索引作为列
                data.append(1)               # 交互值为1

        num_mashups = self.mashup_embed.shape[0]
        num_apis = self.api_embed.shape[0]
        adj = coo_matrix((data, (rows, cols)), shape=(num_mashups, num_apis), dtype=np.float32)

        rowD = np.array(adj.sum(axis=1)).squeeze()
        colD = np.array(adj.sum(axis=0)).squeeze()

        adj_data = adj.data.copy()
        for i in range(len(adj_data)):
            user = adj.row[i]
            item = adj.col[i]
            adj_data[i] /= np.sqrt(rowD[user] * colD[item])

        adj_norm = coo_matrix((adj_data, (adj.row, adj.col)), shape=adj.shape)
        adj_norm = adj_norm.tocsr()

        def scipy_sparse_mat_to_torch_sparse_tensor(sparse_mx):
            sparse_mx = sparse_mx.tocoo()
            indices = torch.LongTensor(np.vstack((sparse_mx.row, sparse_mx.col)))
            values = torch.FloatTensor(sparse_mx.data)
            shape = torch.Size(sparse_mx.shape)
            return torch.sparse_coo_tensor(indices, values, shape)

        adj_norm_tensor = scipy_sparse_mat_to_torch_sparse_tensor(adj_norm)
        adj_norm_tensor = adj_norm_tensor.coalesce().cuda()

        adj = adj_norm_tensor.coalesce().cpu()
        svd_q = 64

        svd_u, s, svd_v = torch.svd_lowrank(adj, q=svd_q)

        u_mul_s = svd_u @ torch.diag(s)
        v_mul_s = svd_v @ torch.diag(s)

        del s

        self.model = LightGCL_Net(
            n_u = num_mashups,
            n_i = num_apis,
            d = 64,
            u_mul_s = u_mul_s,
            v_mul_s = v_mul_s,
            ut = svd_u.T,
            vt = svd_v.T,
            train_csr = adj_norm,
            adj_norm = adj_norm_tensor,
            l = 2,
            temp = 0.2,
            lambda_1 = 0.2,
            lambda_2 = 1e-7,
            dropout = 0.2,
            device = 'cuda'
        )

        self.f1 = F1ScoreMetrics(top_k=5).to('cuda')

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        x, y, idx = batch
        batch_size = x.shape[0]
        loss = torch.tensor(0, dtype=torch.float32, device='cuda')
        for index, y_item in zip(idx, y):
            positive_idx, negative_idx, sample_idx, target = select_negative_samples(
                y_item, self.negative_samples_ratio)
            loss_part, _, _ = self.model(index.unsqueeze(0), positive_idx, positive_idx, negative_idx, test=False)
            loss += loss_part

        loss = loss / batch_size
        self.log('train/loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> Optional[STEP_OUTPUT]:
        x, y, idx = batch
        result = []
        for index, y_item in zip(idx, y):
            positive_idx, negative_idx, sample_idx, target = select_negative_samples(
                y_item, self.negative_samples_ratio)
            result.append(self.model(index.unsqueeze(0), positive_idx, positive_idx, negative_idx, test=True))

        preds = torch.cat(result)
        self.log('val/F1', self.f1(preds.float(), y), on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int) -> Optional[STEP_OUTPUT]:
        x, y, idx = batch
        result = []
        for index, y_item in zip(idx, y):
            positive_idx, negative_idx, sample_idx, target = select_negative_samples(
                y_item, self.negative_samples_ratio)
            result.append(self.model(index.unsqueeze(0), positive_idx, positive_idx, negative_idx, test=True))

        preds = torch.cat(result)
        return {
            'mashup_emb': preds,
            'preds': preds.float(),
            'targets': y
        }

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
