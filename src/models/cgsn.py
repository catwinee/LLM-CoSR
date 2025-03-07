import os
from typing import Any, Optional

import numpy as np
import torch
import torch.optim
import torchmetrics
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
import torch.nn.functional as F
from src.utils.data_processer import select_negative_samples

class CGSN(LightningModule):
    def __init__(
        self,
        data_dir: str,
        api_embed_path: str,
        mashup_embed_channels: int,
        api_embed_channels: int,
        conv_output_channels: int,
        negative_samples_ratio: int,
        lr: float,
        weight_decay: float,
    ):
        super(CGSN, self).__init__()
        self.save_hyperparameters()
        self.register_buffer('api_embed', torch.from_numpy(np.load(os.path.join(data_dir, api_embed_path))))
        self.num_api = self.api_embed.size(0)
        self.negative_samples_ratio = negative_samples_ratio

        self.conv1 = nn.Conv1d(in_channels=mashup_embed_channels, out_channels=conv_output_channels, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=conv_output_channels, out_channels=api_embed_channels, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=self.num_api, out_channels=api_embed_channels, kernel_size=1)
        self.criterion = nn.CrossEntropyLoss()
        self.contrastive_loss = nn.CosineEmbeddingLoss(margin=0.5)
        self.f1 = torchmetrics.F1Score(top_k=5)

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        x, y = batch
        loss = torch.tensor(0, dtype=torch.float32)

        x = x.unsqueeze(0)

        mashup = F.relu(self.conv1(x.transpose(1, 2)))
        mashup = F.relu(self.conv2(mashup)).transpose(1, 2).squeeze(0)
        
        output = torch.mm(mashup, self.api_embed.t())
        pred = F.relu(self.conv3(output.unsqueeze(2))).squeeze(2)

        loss = self.criterion(output, y.float())

        contrastive_loss = 0
        x = x.squeeze(0)
        for idx, (x_item, y_item) in enumerate(zip(x, y)):
            positive_idx, negative_idx, sample_idx, target = select_negative_samples(
                y_item, self.negative_samples_ratio
            )

            embed = pred[idx]  # (num_apis,)

            positive_samples = self.api_embed[positive_idx]  # (n_positives, api_channels)
            negative_samples = self.api_embed[negative_idx]  # (n_negatives, api_channels)

            positive_similarity = F.cosine_similarity(embed.repeat(len(positive_idx), 1), positive_samples)
            negative_similarity = F.cosine_similarity(embed.repeat(len(negative_idx), 1), negative_samples)

            positive_loss = torch.mean(1 - positive_similarity)
            negative_loss = torch.mean(F.relu(negative_similarity))
            contrastive_loss += positive_loss + negative_loss

        loss += contrastive_loss / len(x)
        self.log('train/loss', loss)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> Optional[STEP_OUTPUT]:
        mashups, labels = batch
        batch_size = mashups.size(0)

        mashups = mashups.unsqueeze(1)
        
        mashup = F.relu(self.conv1(mashups.transpose(1, 2)))
        mashup = F.relu(self.conv2(mashup)).transpose(1, 2)
        
        preds = torch.bmm(mashup, self.api_embed.unsqueeze(0).repeat(batch_size, 1, 1).transpose(1, 2))
        preds = preds.view(batch_size, self.num_api)
        self.log('val/F1', self.f1(preds, labels), on_step=False, on_epoch=True, prog_bar=True)
        return {
            'preds': preds,
            'targets': labels
        }

    def test_step(self, batch: Any, batch_idx: int) -> Optional[STEP_OUTPUT]:
        mashups, labels = batch
        batch_size = mashups.size(0)

        mashups = mashups.unsqueeze(1)
        
        mashup_map = F.relu(self.conv1(mashups.transpose(1, 2)))
        mashup_map = F.relu(self.conv2(mashup_map)).transpose(1, 2)
        
        preds = torch.bmm(mashup_map, self.api_embed.unsqueeze(0).repeat(batch_size, 1, 1).transpose(1, 2))
        preds = preds.view(batch_size, self.num_api)
        return {
            'preds': preds,
            'targets': labels
        }

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )