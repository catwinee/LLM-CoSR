import os
from typing import Any, Optional, List

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
        dropout_rate: float = 0.5,
        contrastive_weight: float = 20,
        # view_name_list: List[str] = [
        #     "1-views-dict-50-70%-10.pkl",
        #     "2-views-dict-50-70%-10.pkl",
        # ]
        view_name: str = "seed=12345-1-views-dict-50-70%-10.pkl"
    ):
        super(MM, self).__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay

        self.api_embed = nn.Parameter(torch.from_numpy(np.load(os.path.join(data_dir, api_path))))
        # self.register_buffer('api_embed', torch.from_numpy(np.load(os.path.join(data_dir, api_path))))
        # views_dict_list = [
        #     pickle.load(open(os.path.join(data_dir, f"../views/{view_name}"), "rb")) for view_name in view_name_list
        # ]
        # self.pos_view_list = [views_dict['pos_view'] for views_dict in views_dict_list]
        # self.neg_view = views_dict_list[0]['neg_view']
        views_dict = pickle.load(open(os.path.join(data_dir, f"../views/{view_name}"), "rb")) 
        self.pos_view = views_dict['pos_view'] 
        self.neg_view = views_dict['neg_view']
        self.num_apis = self.api_embed.shape[0]
        self.embed_channels = self.api_embed.shape[1]
        self.contrastive_weight = contrastive_weight

        self.mlp = nn.Sequential(
            nn.Linear(self.embed_channels, hidden_channels),
            # nn.Linear(self.embed_channels, self.embed_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_channels, self.embed_channels),
            nn.ReLU()
        )
        self.fusion_layer = nn.Sequential(
            nn.Linear(2 * self.embed_channels, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 1)
        )
        self.gnn_encoder = GNNEncoder(self.embed_channels, hidden_channels, self.embed_channels)
        # 泛化性： 1. 类似数据集 2. 讨论理论，讨论与验证其他工作的是否正确

        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.f1 = F1ScoreMetrics(top_k=5).to('cuda')

    # def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
    #     mashups, labels = batch # mashups: [bs, dim]
    #     output = self.mlp(mashups) # [bs, dim]
    #     output = torch.mm(output, self.api_embed.t()) # [bs, num_apis]
    #     loss = self.criterion(output, labels.float())
    #     self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=False)
    #     return loss

    # def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
    #     mashups, labels, index = batch  # mashups: [bs, dim]
    #     mashup_feat = self.mlp(mashups) # [bs, dim]
    #     # mashup_proj = F.normalize(self.proj(mashups), dim=-1)
        
    #     pos_view_list = [
    #         torch.cat([pos_view[idx] for idx in index], dim=1) for pos_view in self.pos_view_list
    #     ]
    #     neg_view = torch.cat([self.neg_view[idx] for idx in index], dim=1)

    #     if pos_view_list[0].numel() != 0:
    #         # pos_embed_list = [
    #         #     F.normalize(self.gnn_encoder(self.api_embed, pos_view), dim=-1) for pos_view in pos_view_list
    #         # ]
    #         # neg_embed = F.normalize(self.gnn_encoder(self.api_embed, neg_view), dim=-1)

    #         # mashup_expand = mashup_proj.unsqueeze(1).repeat(
    #         #     1, self.num_apis, 1)
    #         # pos_similarity = F.cosine_similarity(
    #         #     mashup_expand, pos_embed.unsqueeze(0).repeat(mashup_expand.shape[0], 1, 1)
    #         # )
    #         # neg_similarity = F.cosine_similarity(
    #         #     mashup_expand, neg_embed.unsqueeze(0).repeat(mashup_expand.shape[0], 1, 1)
    #         # )

    #         # contrastive_loss = -torch.log(
    #         #     torch.exp(pos_similarity) / (torch.exp(pos_similarity) + torch.exp(neg_similarity))
    #         # )

    #         pos_embeds = [
    #             F.normalize(self.gnn_encoder(self.api_embed, pos_view), dim=-1)
    #             for pos_view in pos_view_list
    #         ]
    #         neg_embed = F.normalize(self.gnn_encoder(self.api_embed, neg_view), dim=-1)

    #         weight_params = nn.Parameter(torch.ones(len(pos_embeds)))
    #         weights = F.softmax(weight_params, dim=0)

    #         agg_pos_embed = sum(w * e for w, e in zip(weights, pos_embeds))

    #         mashup_expand = mashup_feat.unsqueeze(1).repeat(
    #             1, self.num_apis, 1)
    #         pos_similarity = F.cosine_similarity(
    #             mashup_expand, agg_pos_embed.unsqueeze(0).repeat(mashup_expand.shape[0], 1, 1)
    #         )
    #         neg_similarity = F.cosine_similarity(
    #             mashup_expand, neg_embed.unsqueeze(0).repeat(mashup_expand.shape[0], 1, 1)
    #         )

    #         contrastive_loss = -torch.log(
    #             torch.exp(pos_similarity) / (torch.exp(pos_similarity) + torch.exp(neg_similarity))
    #         )

    #     pos_indices = torch.unique(torch.where(labels.bool())[1])  # [num_pos]
        
    #     num_neg = pos_indices.shape[0]
    #     all_indices = torch.arange(self.num_apis, device=labels.device)
    #     neg_mask = ~torch.isin(all_indices, pos_indices)
    #     neg_indices = all_indices[neg_mask][torch.randperm(sum(neg_mask))[:num_neg]]
        
    #     selected_indices = torch.cat([pos_indices, neg_indices])  # [M]
    #     api_subset = self.api_embed[selected_indices]  # [M, dim]
        
    #     combined = torch.cat([
    #         mashup_feat.unsqueeze(1).expand(-1, api_subset.size(0), -1),
    #         api_subset.unsqueeze(0).expand(mashup_feat.size(0), -1, -1)
    #     ], dim=-1)  # [bs, M, 2*dim]
        
    #     preds_subset = self.fusion_layer(combined).squeeze(-1)  # [bs, M]
    #     labels_subset = labels[:, selected_indices]  # [bs, M]
    #     loss = self.criterion(preds_subset, labels_subset.float())
    #     loss += torch.mean(contrastive_loss) * self.contrastive_weight

    #     self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
    #     return loss

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        mashups, labels, idx = batch  # mashups: [bs, dim]
        mashup_feat = self.mlp(mashups)  # [bs, dim]
        
        contrastive_loss, count = 0, 0
        for x_item, index in zip(mashups, idx):
            if self.pos_view[index].numel() == 0:
                continue
            count += 1
            x_item = x_item.unsqueeze(0).repeat(self.num_apis, 1)
            pos_view = self.pos_view[index]
            neg_view = self.neg_view[index]

            pos_embed = self.gnn_encoder(self.api_embed, pos_view)
            neg_embed = self.gnn_encoder(self.api_embed, neg_view)

            pos_similarity = F.cosine_similarity(x_item, pos_embed)
            neg_similarity = F.cosine_similarity(x_item, neg_embed)

            contrastive_loss += -torch.log(torch.exp(pos_similarity) / (torch.exp(pos_similarity) + torch.exp(neg_similarity)))

        pos_mask = labels.bool()
        pos_indices = torch.unique(torch.where(pos_mask)[1])  # [num_pos]
        
        num_neg = pos_indices.shape[0]
        all_indices = torch.arange(self.num_apis, device=labels.device)
        neg_mask = ~torch.isin(all_indices, pos_indices)
        neg_indices = all_indices[neg_mask][torch.randperm(sum(neg_mask))[:num_neg]]
        
        selected_indices = torch.cat([pos_indices, neg_indices])  # [M]
        api_subset = self.api_embed[selected_indices]  # [M, dim]
        
        mashup_expanded = mashup_feat.unsqueeze(1)          # [bs, 1, dim]
        api_expanded = api_subset.unsqueeze(0)              # [1, M, dim]
        
        combined = torch.cat([
            mashup_expanded.expand(-1, api_subset.size(0), -1),
            api_expanded.expand(mashup_feat.size(0), -1, -1)
        ], dim=-1)  # [bs, M, 2*dim]
        
        preds_subset = self.fusion_layer(combined).squeeze(-1)  # [bs, M]

        
        labels_subset = labels[:, selected_indices]  # [bs, M]
        loss = self.criterion(preds_subset, labels_subset.float())

        if count != 0:
            loss += torch.mean(contrastive_loss) * self.contrastive_weight / count

        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> Optional[STEP_OUTPUT]:
        mashups, labels = batch
        mashup_feat = self.mlp(mashups)
        mashup_expanded = mashup_feat.unsqueeze(1)
        api_expanded = self.api_embed.unsqueeze(0)

        combined = torch.cat([
            mashup_expanded.expand(-1, self.num_apis, -1),
            api_expanded.expand(mashup_feat.size(0), -1, -1)
        ], dim=-1)
        
        preds = self.fusion_layer(combined).squeeze(-1)
        
        self.log('val/F1', self.f1(preds, labels), 
                on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int) -> Optional[STEP_OUTPUT]:
        mashups, labels = batch
        mashup_feat = self.mlp(mashups)
        mashup_expanded = mashup_feat.unsqueeze(1)
        api_expanded = self.api_embed.unsqueeze(0)

        combined = torch.cat([
            mashup_expanded.expand(-1, self.num_apis, -1),
            api_expanded.expand(mashup_feat.size(0), -1, -1)
        ], dim=-1)
        
        preds = self.fusion_layer(combined).squeeze(-1)
        return {
            'preds': preds,
            'targets': labels
        }


    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
