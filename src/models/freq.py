import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
from typing import Any, Optional


class Freq(LightningModule):
    r"""A heuristic approach based on API usage frequency, which always recommends the top N frequently invoked APIs.

    """

    def __init__(
        self,
    ):
        super(Freq, self).__init__()
        self.linear = nn.Linear(1, 1)
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def on_train_start(self) -> None:
        self.propensity_score = self.trainer.datamodule.propensity_score
        self.propensity_score = self.propensity_score.to(self.device)
        self.propensity_score += 0.5

    def training_step(self, *args, **kwargs) -> STEP_OUTPUT:
        x = self.linear(torch.tensor([1.0]).to(self.device))
        return self.criterion(x, torch.tensor([1.0]).to(self.device))

    def validation_step(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        self.log('val/F1', 1.0, on_step=False, on_epoch=True, prog_bar=False)

    def test_step(self, batch: Any, batch_idx: int) -> Optional[STEP_OUTPUT]:
        mashups, labels = batch
        batch_size = mashups.size(0)
        return {
            'preds': self.propensity_score.repeat(batch_size, 1),
            'targets': labels
        }

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.parameters(), lr=0.0, weight_decay=0.0
        )