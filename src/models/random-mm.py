import os
from typing import Any, Optional

import pickle
import pandas as pd
import numpy as np
import torch
import torch.optim
import torchmetrics
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
import torch.nn.functional as F

from src.utils.metrics import F1ScoreMetrics
from src.utils.gnn_encoder import GNNEncoder


class MM(LightningModule):
    def __init__(
        self,
        data_dir: str,
        api_path: str,
        mashup_path: str,
        hidden_channels: int,
        lr: float,
        weight_decay: float,
        dropout_rate: float = 0.2,
        contrastive_weight: float = 1,
        ord: str = "03",
        view_name: str = "seed=12345-1-views-dict-50-70%-10.pkl",
        negative_ratio: int = 10,
        remove_ratio: int = 0.5,
    ):
        super(MM, self).__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.negative_ratio = negative_ratio
        self.remove_ratio = remove_ratio

        self.api_embed = nn.Parameter(torch.from_numpy(np.load(os.path.join(data_dir, api_path))))

        views_dict = pickle.load(open(os.path.join(data_dir, f"../views/{view_name}"), "rb"))
        self.pos_view = views_dict['pos_view']
        self.neg_view = views_dict['neg_view']

        self.num_apis = self.api_embed.shape[0]
        self.embed_channels = self.api_embed.shape[1]

        self.mlp = nn.Sequential(
            nn.Linear(self.embed_channels, hidden_channels),
            # nn.Linear(self.embed_channels, self.embed_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_channels, self.embed_channels),
            nn.ReLU()
        )
        self.gnn_encoder = GNNEncoder(self.embed_channels, hidden_channels, self.embed_channels)

        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.f1 = F1ScoreMetrics(top_k=5).to('cuda')

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        mashups, labels, idx = batch
        output = self.mlp(mashups) # [bs, dim]
        
        contrastive_loss, count = 0, 0
        for x_item, index in zip(output, idx):
            if self.pos_view[index].numel() == 0:
                continue
            count += 1
            x_item = x_item.unsqueeze(0).repeat(self.num_apis, 1)

            neg_view = self.neg_view[index]
            num_edges = neg_view.size(1)
            selected_indices = torch.randperm(num_edges)[:int(num_edges * (1 - self.remove_ratio))]

            pos_view = neg_view[:, selected_indices]

            pos_embed = self.gnn_encoder(self.api_embed, pos_view)
            neg_embed = self.gnn_encoder(self.api_embed, neg_view)

            pos_similarity = F.cosine_similarity(x_item, pos_embed)
            neg_similarity = F.cosine_similarity(x_item, neg_embed)

            contrastive_loss += -torch.log(torch.exp(pos_similarity) / (torch.exp(pos_similarity) + torch.exp(neg_similarity)))

        pos_indices = torch.unique(torch.nonzero(labels)[:, 1])  # [num_pos]
        
        num_neg = self.negative_ratio * pos_indices.shape[0]
        all_indices = torch.arange(self.num_apis, device=labels.device)
        neg_mask = ~torch.isin(all_indices, pos_indices)
        neg_candidates = all_indices[neg_mask]
        
        neg_indices = neg_candidates[torch.randperm(len(neg_candidates))[:num_neg]]
        
        selected_indices = torch.cat([pos_indices, neg_indices])
        
        api_embed_subset = self.api_embed[selected_indices]  # [selected_num, D]
        logits_subset = torch.mm(output, api_embed_subset.t())  # [bs, selected_num]
        
        labels_subset = labels[:, selected_indices]  # [bs, selected_num]
        
        loss = self.criterion(logits_subset, labels_subset.float())

        if count != 0:
            loss += torch.mean(contrastive_loss) / count

        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> Optional[STEP_OUTPUT]:
        mashups, labels = batch # [bs, embed]
        output = self.mlp(mashups)
        output = torch.mm(output, self.api_embed.t())
        self.log('val/F1', self.f1(output, labels), on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int) -> Optional[STEP_OUTPUT]:
        mashups, labels = batch
        mashup_emb = self.mlp(mashups)  # [bs, dim] 用户/服务的嵌入
        output = torch.mm(mashup_emb, self.api_embed.t())  # [bs, num_apis]

        return {
            'mashup_emb': mashup_emb,
            'preds': output,
            'targets': labels
        }

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
