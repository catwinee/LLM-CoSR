import os.path
import numpy as np
import torch
import torchmetrics
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
from typing import Any, Optional

from src.utils.metrics import F1ScoreMetrics

class MLP(LightningModule):
    r"""A MLP with two linear layers.

    Args:
        data_dir (str): Path to the folder where the data is located.
        api_embed_path (str): Path to embeddings vectors of api that is relative to :attr:`data_dir`.
        mlp_output_channels (int): Size of each output of the first linear layer.
        mashup_embed_channels (int): Size of each embedding vector of mashup.
        lr (float): Learning rate (default: :obj:`1e-3`).
        weight_decay (float): weight decay (default: :obj:`1e-5`).
    """
    def __init__(
        self,
        data_dir,
        api_embed_path: str,
        mlp_output_channels: int,
        mashup_embed_channels: int,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
    ):
        r"""A MLP with two linear layers.

        Args:
            data_dir (str): Path to the folder where the data is located.
            api_embed_path (str): Path to embeddings vectors of api that is relative to :attr:`data_dir`.
            mlp_output_channels (int): Size of each output of the first linear layer.
            mashup_embed_channels (int): Size of each embedding vector of mashup
            lr: Learning rate (default: :obj:`1e-3`).
            weight_decay: weight decay (default: :obj:`1e-5`).
        """
        super(MLP, self).__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.register_buffer('api_embeds', torch.from_numpy(np.load(os.path.join(data_dir, api_embed_path))))
        self.num_api = self.api_embeds.size(0)
        self.api_embed_channels = self.api_embeds.size(1)

        self.linear = nn.Sequential(
            nn.Linear(mashup_embed_channels + self.api_embed_channels, mlp_output_channels),
            nn.ReLU(),
            nn.Linear(mlp_output_channels, 1)
        )

        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.f1 = F1ScoreMetrics(top_k=5)

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        mashup, api, label = batch
        input_feature = torch.cat((mashup, api), dim=1)
        preds = self.linear(input_feature)
        preds = preds.view(-1)
        loss = self.criterion(preds, label.float())
        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> Optional[STEP_OUTPUT]:
        mashups, labels = batch
        batch_size = mashups.size(0)
        mashups = mashups.unsqueeze(1).repeat(1, self.num_api, 1)
        apis = self.api_embeds.unsqueeze(0).repeat(batch_size, 1, 1)
        input_feature = torch.cat((mashups, apis), dim=-1)
        preds = self.linear(input_feature)
        preds = preds.view(batch_size, self.num_api)
        self.log('val/F1', self.f1(preds, labels), on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int) -> Optional[STEP_OUTPUT]:
        mashups, labels = batch
        batch_size = mashups.size(0)
        mashups = mashups.unsqueeze(1).repeat(1, self.num_api, 1)
        apis = self.api_embeds.unsqueeze(0).repeat(batch_size, 1, 1)
        input_feature = torch.cat((mashups, apis), dim=-1)
        preds = self.linear(input_feature)
        preds = preds.view(batch_size, self.num_api)
        return {
            'preds': preds,
            'targets': labels
        }

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
