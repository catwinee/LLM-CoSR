from typing import Any, Optional, List
import torch
import torchmetrics
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT

from src.models.components.coacn_net import COACNNet


class COACN(LightningModule):
    r"""A model from article "Service Recommendation for Composition Creation based on Collaborative Attention
    Convolutional Network".

    Args:
        data_dir (str): Path to the folder where the data is located.
        mashup_embed_path (str): Path to embeddings vectors of mashups. Relative to :attr:`data_dir`.
        domain_embed_path (str): Path to embeddings vectors of service domains. Relative to :attr:`data_dir`.
        api_embed_path (str): Path to embeddings vectors of APIs. Relative to :attr:`data_dir`.
        invoked_matrix_path (str): Path to invoked matrix between mashups and APIs.
        mashup_embed_channels (int): Size of each embedding vector of mashup.
        api_embed_channels (int): Size of each embedding vector of API.
        domain_embed_channels (int): Size of each embedding vector of domain.
        feature_dim (int): Size of each output of feature extractor.
        hp_beta (float): a hyper-parameter to control the proportion of service domain information
        hp_num_gcn_layer (int): The number of layers of LightGCN
        hp_weight_gcn_layer (List[int]): A list of weights of LightGCN
        lr (float): Learning rate (default: :obj:`1e-3`).
        weight_decay (float): weight decay (default: :obj:`1e-5`).
    """
    def __init__(
        self,
        data_dir: str,
        mashup_embed_path: str,
        domain_embed_path: str,
        api_embed_path: str,
        invoked_matrix_path: str,
        mashup_embed_channels: int,
        api_embed_channels: int,
        domain_embed_channels: int,
        feature_dim: int,
        hp_beta: float,
        hp_num_gcn_layer: int,
        hp_weight_gcn_layer: List[int],
        lr: float,
        weight_decay: float = 1e-5,
    ):
        super(COACN, self).__init__()
        self.save_hyperparameters()

        self.coacn_net = COACNNet(self.hparams)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.f1 = torchmetrics.F1Score(top_k=5)

    def forward(self, x: torch.Tensor) -> Any:
        return self.coacn_net(x)

    def step(self, batch: Any):
        x, target = batch
        pred = self.forward(x)
        loss = self.criterion(pred, target.float())
        return loss, pred, target

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        loss, pred, target = self.step(batch)
        self.log('train/loss', loss)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> Optional[STEP_OUTPUT]:
        loss, pred, target = self.step(batch)
        self.log('val/F1', self.f1(pred, target), on_step=False, on_epoch=True, prog_bar=False)
        return {
            'preds': pred,
            'targets': target
        }

    def test_step(self, batch: Any, batch_idx: int) -> Optional[STEP_OUTPUT]:
        loss, pred, target = self.step(batch)
        return {
            'preds': pred,
            'targets': target
        }

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
