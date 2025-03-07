import os.path
from typing import Any, Optional

import numpy as np
import torch
import torchmetrics
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn


class MISR(LightningModule):
    r"""A model from article "A Deep Neural Network With Multiplex Interactions for Cold-Start Service Recommendation".


    Args:
        data_dir (str): Path to the folder where the data is located.
        mashup_embed_path (str): Path to embeddings vectors of api that is relative to :attr:`data_dir`.
        api_embed_path (str): Path to embeddings vectors of api that is relative to :attr:`data_dir`.
        conv_in_channels (int): Size of each input of text convolution layer.
        conv_out_channels (int): Size of each output of text convolution layer.
        out_channels (int): Size of each output of feature extraction component.
        text_len (int): Size of each text of description of mashups and apis.
        lr (float): Learning rate (default: :obj:`1e-3`).
        weight_decay (float): weight decay (default: :obj:`1e-5`).
    """

    def __init__(
        self,
        data_dir: str,
        mashup_embed_path: str,
        api_embed_path: str,
        conv_in_channels: int = 300,
        conv_out_channels: int = 100,
        out_channels: int = 128,
        text_len: int = 72,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
    ):
        super(MISR, self).__init__()
        self.save_hyperparameters()

        self.mashup_embed_path = mashup_embed_path
        self.text_len = text_len
        self.out_channels = conv_out_channels

        self.register_buffer('mashup_embed', torch.from_numpy(np.load(os.path.join(data_dir, mashup_embed_path))))
        self.register_buffer('api_embed', torch.from_numpy(np.load(os.path.join(data_dir, api_embed_path))))
        self.num_api = self.api_embed.size(0)
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=conv_in_channels, out_channels=conv_out_channels, kernel_size=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=text_len)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv1d(in_channels=conv_in_channels, out_channels=conv_out_channels, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=text_len - 2)
        )
        self.conv_3 = nn.Sequential(
            nn.Conv1d(in_channels=conv_in_channels, out_channels=conv_out_channels, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(in_channels=conv_out_channels, out_channels=conv_out_channels, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=text_len - 4)
        )
        self.conv_4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3),
            nn.Conv1d(in_channels=conv_in_channels, out_channels=conv_out_channels, kernel_size=1),
            nn.MaxPool1d(kernel_size=int(text_len / 3))
        )
        self.feature_extra = nn.Sequential(
            nn.Linear(in_features=conv_out_channels * 4, out_features=conv_out_channels * 4),
            nn.ReLU(),
            nn.Linear(in_features=conv_out_channels * 4, out_features=conv_out_channels * 4),
            nn.ReLU(),
            nn.Linear(in_features=conv_out_channels * 4, out_features=conv_out_channels * 4),
            nn.ReLU(),
        )
        self.extra_feature = nn.Sequential(
            nn.Linear(in_features=conv_out_channels * 8, out_features=conv_out_channels * 4),
            nn.ReLU(),
            nn.Linear(in_features=conv_out_channels * 4, out_features=out_channels)
        )
        self.pred_map = nn.Sequential(
            nn.Linear(in_features=out_channels, out_features=int(out_channels / 2)),
            nn.ReLU(),
            nn.Linear(in_features=int(out_channels / 2), out_features=1)
        )

        self.f1 = torchmetrics.F1Score(top_k=5)
        self.criterion = nn.BCEWithLogitsLoss()
        self.lr = lr
        self.weight_decay = weight_decay

    def extra_seq(self, x):
        batch_size = x.size(0)
        input_feature = x.permute(0, 2, 1)
        c_m = [self.conv_1(input_feature), self.conv_2(input_feature), self.conv_3(input_feature),
               self.conv_4(input_feature)]
        c_m = [vec.view(batch_size, self.out_channels) for vec in c_m]
        c_m = torch.cat(c_m, dim=1)
        return self.feature_extra(c_m)

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        mashups, apis, labels = batch
        batch_size = mashups.size(0)

        # extract features of mashup and api
        v_seq_m = self.extra_seq(mashups)
        v_seq_s = self.extra_seq(apis)

        output = torch.cat((v_seq_m, v_seq_s), dim=-1)
        output = self.extra_feature(output)
        output = self.pred_map(output)
        pred = output.view(batch_size)

        loss = self.criterion(pred, labels.float())
        self.log('train/loss', loss)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> Optional[STEP_OUTPUT]:
        mashups, labels = batch
        batch_size = mashups.size(0)

        v_seq_m = self.extra_seq(mashups)  # (batch_size, channels)
        v_seq_s = self.extra_seq(self.api_embed)  # (num_api, channels)
        v_seq_m = v_seq_m.unsqueeze(dim=1).repeat(1, self.num_api, 1)
        v_seq_s = v_seq_s.unsqueeze(dim=0).repeat(batch_size, 1, 1)
        input_feature = torch.cat((v_seq_m, v_seq_s), dim=-1)
        output = self.extra_feature(input_feature)
        output = self.pred_map(output)
        preds = output.view(batch_size, self.num_api)

        self.log('val/F1', self.f1(preds, labels), on_step=False, on_epoch=True, prog_bar=False)
        return {
            'preds': preds,
            'targets': labels
        }

    def test_step(self, batch: Any, batch_idx: int) -> Optional[STEP_OUTPUT]:
        mashups, labels = batch
        batch_size = mashups.size(0)

        v_seq_m = self.extra_seq(mashups)  # (batch_size, channels)
        v_seq_s = self.extra_seq(self.api_embed)  # (num_api, channels)
        v_seq_m = v_seq_m.unsqueeze(dim=1).repeat(1, self.num_api, 1)
        v_seq_s = v_seq_s.unsqueeze(dim=0).repeat(batch_size, 1, 1)
        input_feature = torch.cat((v_seq_m, v_seq_s), dim=-1)
        output = self.extra_feature(input_feature)
        output = self.pred_map(output)
        preds = output.view(batch_size, self.num_api)

        return {
            'mashup_emb': v_seq_m,
            'preds': preds,
            'targets': labels
        }

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
