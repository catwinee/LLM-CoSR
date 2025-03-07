import os
from typing import Any, Optional

import numpy as np
import torch.optim
import torchmetrics
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn

from src.utils.data_processer import select_negative_samples
from src.utils.metrics import F1ScoreMetrics


class T2L2(LightningModule):
    r"""A model from article "T2L2: A Tiny Three Linear Layers Model for Service Mashup Creation".

    T2L2 is a tiny model with three linear layers requiring only requires functional descriptions of services and mashups as input. The first two linear layers are used to align the representation space of services and mashups. The last linear layer is used to calculate the matching scores of services and mashups.

    Args:
        data_dir (str): Path to the folder where the data is located.
        api_embed_path (str): Path to embeddings vectors of apis. Relative to :attr:`data_dir`.
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
        super(T2L2, self).__init__()
        self.save_hyperparameters()

        self.register_buffer('api_embed', torch.from_numpy(np.load(os.path.join(data_dir, api_embed_path))))
        self.num_api = self.api_embed.size(0)
        self.negative_samples_ratio = negative_samples_ratio

        self.vector_space_map = nn.Linear(
            in_features=mashup_embed_channels,
            out_features=api_embed_channels)
        self.msg_generation = nn.Linear(
            in_features=api_embed_channels,
            out_features=api_embed_channels)
        # self.match_linear = nn.Linear(in_features=2 * api_embed_channels, out_features=1)
        self.match_linear = nn.Sequential(
            nn.Linear(in_features=2 * api_embed_channels, out_features=mlp_output_channels),
            nn.ReLU(),
            nn.Linear(in_features=mlp_output_channels, out_features=1)
        )
        self.sigmoid = nn.Sigmoid()
        self.criterion = torch.nn.BCEWithLogitsLoss(size_average=False)
        self.f1 = F1ScoreMetrics(top_k=5).to('cuda')
        # self.f1 = MultiLabelF1AtK(k=5).to('cuda')
        # self.f1 = torchmetrics.F1Score(top_k=5)

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        x, y = batch
        num_sample = 0
        loss = torch.tensor(0, dtype=torch.float32, device='cuda')
        for x_item, y_item in zip(x, y):
            # select negative samples
            mashup_map = self.vector_space_map(x_item)

            # noise
            positive_idx, negative_idx, sample_idx, target = select_negative_samples(
                y_item, 2)
            y_item[negative_idx] = True

            positive_idx, negative_idx, sample_idx, target = select_negative_samples(
                y_item, self.negative_samples_ratio)

            output = mashup_map.repeat(len(sample_idx), 1)
            output = torch.cat((output, self.api_embed[sample_idx]), dim=1)
            pred = self.match_linear(output).to('cuda')

            pred = pred.view(-1)
            num_sample += len(sample_idx)
            target = target.to('cuda')
            loss += self.criterion(pred, target)

            # update service representation
            message = self.msg_generation(mashup_map)
            api_new = message + self.api_embed[positive_idx]
            self.api_embed[positive_idx, :] = api_new.detach()

        loss = loss / num_sample
        self.log('train/loss', loss)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> Optional[STEP_OUTPUT]:
        mashups, labels = batch
        batch_size = mashups.size(0)
        mashups = self.vector_space_map(mashups)
        mashups = mashups.unsqueeze(1).repeat(1, self.num_api, 1)
        apis = self.api_embed.unsqueeze(0).repeat(batch_size, 1, 1)
        preds = torch.cat((mashups, apis), dim=-1)
        preds = self.match_linear(preds).to('cuda:0')
        preds = preds.view(batch_size, self.num_api)
        self.log('val/F1', self.f1(preds, labels), on_step=False, on_epoch=True, prog_bar=True)
        return {
            'preds': preds,
            'targets': labels
        }

    def test_step(self, batch: Any, batch_idx: int) -> Optional[STEP_OUTPUT]:
        mashups, labels = batch
        batch_size = mashups.size(0)
        mashup_emb = self.vector_space_map(mashups)
        mashups = mashup_emb.unsqueeze(1).repeat(1, self.num_api, 1)
        apis = self.api_embed.unsqueeze(0).repeat(batch_size, 1, 1)
        preds = torch.cat((mashups, apis), dim=-1)
        preds = self.match_linear(preds)
        preds = preds.view(batch_size, self.num_api)
        return {
            'mashup_emb': mashup_emb,
            'preds': preds,
            'targets': labels
        }

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
