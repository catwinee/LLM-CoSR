import os
from typing import Any, Optional

import numpy as np
import torch.optim
import torchmetrics
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
from src.utils.metrics import F1ScoreMetrics


class T2L2WithoutPropagation(LightningModule):
    r"""A modification of T2L2 model, which remove the propagation component from T2L2. The original model is from
    the article "T2L2: A Tiny Three Linear Layers Model for Service Mashup Creation".

    Args:
        data_dir (str): Path to the folder where the data is located.
        api_embed_path (str): Path to embeddings vectors of api that is relative to :attr:`data_dir`.
        mashup_embed_channels (int): Size of each embedding vector of mashup.
        api_embed_channels (int): Size of each embedding vector of mashup.
        mlp_output_channels (int): Size of each output of the third linear layer.
        negative_samples_ratio (int): Ratio of negative to positive in the training stage.
        lr (float): Learning rate (default: :obj:`1e-3`).
        weight_decay (float): weight decay (default: :obj:`1e-5`).
    """

    def __init__(
        self,
        data_dir: str,
        api_embed_path: str,
        mashup_embed_channels: int,
        api_embed_channels: int,
        mlp_output_channels: int,
        negative_samples_ratio: int,
        lr: float,
        weight_decay: float,
    ):
        super(T2L2WithoutPropagation, self).__init__()
        self.save_hyperparameters()

        self.register_buffer('api_embed', torch.from_numpy(np.load(os.path.join(data_dir, api_embed_path))))
        self.num_api = self.api_embed.size(0)
        self.negative_samples_ratio = negative_samples_ratio

        self.vector_space_map = nn.Linear(in_features=mashup_embed_channels, out_features=api_embed_channels)
        self.match_linear = nn.Sequential(
            nn.Linear(in_features=2 * api_embed_channels, out_features=mlp_output_channels),
            nn.ReLU(),
            nn.Linear(in_features=mlp_output_channels, out_features=1)
        )
        self.sigmoid = nn.Sigmoid()

        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.f1 = F1ScoreMetrics(top_k=5)

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        mashups, api_indices, labels = batch
        map_mashups = self.vector_space_map(mashups)
        apis = self.api_embed[api_indices]
        input_feature = torch.cat((map_mashups, apis), dim=-1)
        preds = self.match_linear(input_feature)
        preds = preds.view(-1)
        loss = self.criterion(preds, labels.float())
        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> Optional[STEP_OUTPUT]:
        mashups, labels = batch
        batch_size = mashups.size(0)
        mashups = self.vector_space_map(mashups)
        mashups = mashups.unsqueeze(1).repeat(1, self.num_api, 1)
        apis = self.api_embed.unsqueeze(0).repeat(batch_size, 1, 1)
        preds = torch.cat((mashups, apis), dim=-1)
        preds = self.match_linear(preds)
        preds = preds.view(batch_size, self.num_api)
        self.log('val/F1', self.f1(preds, labels), on_step=False, on_epoch=True, prog_bar=True)
        return {
            'preds': preds,
            'targets': labels
        }

    def test_step(self, batch: Any, batch_idx: int) -> Optional[STEP_OUTPUT]:
        mashups, labels = batch
        batch_size = mashups.size(0)
        mashups = self.vector_space_map(mashups)
        mashups = mashups.unsqueeze(1).repeat(1, self.num_api, 1)
        apis = self.api_embed.unsqueeze(0).repeat(batch_size, 1, 1)
        preds = torch.cat((mashups, apis), dim=-1)
        preds = self.match_linear(preds)
        preds = preds.view(batch_size, self.num_api)
        return {
            'preds': preds,
            'targets': labels
        }

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )

