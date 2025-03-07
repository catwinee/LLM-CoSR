from typing import Any, Optional, List

import torch
import torchmetrics
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT

from src.models.components.sc_net import SCNet
from src.utils.metrics import F1ScoreMetrics


class MTFM(LightningModule):
    r"""A model from article "Mashup-Oriented Web API Recommendation via Multi-Model Fusion and Multi-Task Learning"

    Args:
        embed_channels (int): Size of each embedding vector of mashup.
        num_api (int): the number of candidate apis.
        text_len (int): Size of each text of description of mashups and apis.
        conv_kernel_size (List[int]): List of size of convolution kernel
        conv_num_kernel (int): The number of convolution kernels.
        feature_channels (int): Size of each output of feature extraction component.
        lr (float): Learning rate (default: :obj:`1e-3`).
        weight_decay (float): weight decay (default: :obj:`1e-5`).
    """

    def __init__(
        self,
        embed_channels: int,
        num_api: int,
        text_len: int,
        conv_kernel_size: List[int],
        conv_num_kernel: int,
        feature_channels: int,
        lr: float,
        weight_decay: float = 1e-5,
    ):
        super(MTFM, self).__init__()
        self.save_hyperparameters()

        self.sc_net = SCNet(self.hparams)
        self.lr = lr
        self.weight_decay = weight_decay
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.f1 = F1ScoreMetrics(top_k=5).to('cuda')
        # self.f1 = MultiLabelF1AtK(k=5).to('cuda')
        # self.f1 = torchmetrics.F1Score(top_k=5).to('cuda')

    def forward(self, x: torch.Tensor) -> Any:
        return self.sc_net(x)

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        x, target = batch
        pred = self.forward(x)
        loss = self.criterion(pred, target.float())
        self.log('train/loss', loss)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> Optional[STEP_OUTPUT]:
        x, target = batch
        pred = self.forward(x)
        self.log('val/F1', self.f1(pred, target), on_step=False, on_epoch=True, prog_bar=True)
        return {
            'preds': pred,
            'targets': target
        }

    def test_step(self, batch: Any, batch_idx: int) -> Optional[STEP_OUTPUT]:
        x, target = batch
        pred = self.forward(x)
        return {
            'preds': pred,
            'targets': target
        }

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )