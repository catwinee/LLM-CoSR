import gc
import os.path
import numpy as np
import torch
import torchmetrics
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
from typing import Any, Optional
from src.utils.data_processer import select_negative_samples
from src.utils.utils import del_tensor_rows


class FISR(LightningModule):
    r"""A model from article "Deep Learning Framework for Online Interactive Service Recommendation in Iterative
    Mashup Development"

    Args:
        data_dir (str): Path to the folder where the data is located.
        api_embed_path (str): Path to embeddings vectors of apis. Relative to :attr:`data_dir`.
        feature_dim (str): Size of each output of interaction layer.
        negative_samples_ratio (int): Ratio of negative to positive in the training stage.
        lr (float): Learning rate (default: :obj:`1e-3`).
        weight_decay (float): weight decay (default: :obj:`1e-5`).
    """

    def __init__(
        self,
        data_dir: str,
        api_embed_path: str,
        feature_dim: int,
        negative_samples_ratio: int = 5,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
    ):
        super(FISR, self).__init__()
        self.save_hyperparameters()
        self.negative_samples_ratio = negative_samples_ratio
        self.lr = lr
        self.weight_decay = weight_decay
        self.register_buffer('api_embeds', torch.from_numpy(np.load(os.path.join(data_dir, api_embed_path))))
        self.num_api = self.api_embeds.size(0)

        self.attention_mlp = nn.Sequential(
            nn.Linear(in_features=feature_dim * 4, out_features=80),
            nn.Linear(in_features=80, out_features=1),
        )

        self.interaction = nn.Sequential(
            nn.Linear(in_features=feature_dim * 3, out_features=100),
            nn.Linear(in_features=100, out_features=50),
            nn.Linear(in_features=50, out_features=1),
        )

        self.softmax = nn.Softmax()
        self.cos_sim = nn.CosineSimilarity(dim=1)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.f1 = torchmetrics.F1Score(top_k=5)

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        x, y = batch
        num_all_samples = 0

        selected_apis = []
        loss = torch.tensor(0, dtype=torch.float32).to(self.device)
        for x_item, y_item in zip(x, y):
            # select negative samples
            positive_idx, negative_idx, sample_idx, target = select_negative_samples(
                y_item, self.negative_samples_ratio)
            num_all_samples += len(sample_idx)
            samples = self.api_embeds[sample_idx]
            # cold start
            sim = self.cos_sim(samples, x_item.repeat(samples.size(0), 1)).view(-1)
            index = torch.argmax(sim)
            selected_apis.append(samples[index].view(1, -1))
            samples = del_tensor_rows(samples, index, 1)
            num_sample = samples.size(0)
            selected_embed = torch.cat(selected_apis, dim=0)

            multi = torch.mul(samples.unsqueeze(1), selected_embed.unsqueeze(0))
            sub = torch.sub(samples.unsqueeze(1), selected_embed.unsqueeze(0))
            input = torch.cat([samples.unsqueeze(1).repeat(1, selected_embed.size(0), 1),
                               selected_embed.unsqueeze(0).repeat(num_sample, 1, 1), multi, sub], dim=-1)
            degree = self.attention_mlp(input).view(num_sample, -1)
            degree = self.softmax(degree)
            selected_embed = torch.mul(degree.unsqueeze(-1), selected_embed.unsqueeze(0))
            selected_embed = selected_embed.sum(1)
            input_feature = torch.cat([x_item.repeat(num_sample, 1), selected_embed, samples], dim=-1)
            pred = self.interaction(input_feature).view(-1)
            pred = torch.cat([pred[:index], torch.tensor([1.0]).to(self.device), pred[index:]], dim=0)
            loss += self.criterion(pred, target.cuda())
        loss = loss / num_all_samples
        self.log('train/loss', loss)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> Optional[STEP_OUTPUT]:
        x, y = batch
        selected_apis = []
        preds = []
        for x_item, y_item in zip(x, y):
            # cold start
            sim = self.cos_sim(self.api_embeds, x_item.repeat(self.num_api, 1)).view(-1)
            index = torch.argmax(sim)
            selected_apis.append(self.api_embeds[index].view(1, -1))
            samples = del_tensor_rows(self.api_embeds, index, 1)
            num_sample = samples.size(0)
            selected_embed = torch.cat(selected_apis, dim=0)

            multi = torch.mul(samples.unsqueeze(1), selected_embed.unsqueeze(0))
            sub = torch.sub(samples.unsqueeze(1), selected_embed.unsqueeze(0))
            input = torch.cat([samples.unsqueeze(1).repeat(1, selected_embed.size(0), 1),
                               selected_embed.unsqueeze(0).repeat(num_sample, 1, 1), multi, sub], dim=-1)
            degree = self.attention_mlp(input).view(num_sample, -1)
            del input
            gc.collect()
            degree = self.softmax(degree)
            selected_embed = torch.mul(degree.unsqueeze(-1), selected_embed.unsqueeze(0))
            selected_embed = selected_embed.sum(1)
            input_feature = torch.cat([x_item.repeat(num_sample, 1), selected_embed, samples], dim=-1)
            pred = self.interaction(input_feature).view(-1)
            del input_feature
            gc.collect()
            pred = torch.cat([pred[:index], torch.tensor([1.0]).to(self.device), pred[index:]], dim=0)
            preds.append(pred.unsqueeze(0))
        preds = torch.cat(preds, dim=0)
        self.log('val/F1', self.f1(preds, y), on_step=False, on_epoch=True, prog_bar=False)

    def test_step(self, batch: Any, batch_idx: int) -> Optional[STEP_OUTPUT]:
        x, y = batch
        selected_apis = []
        preds = []
        for x_item, y_item in zip(x, y):
            # cold start
            sim = self.cos_sim(self.api_embeds, x_item.repeat(self.num_api, 1)).view(-1)
            index = torch.argmax(sim)
            selected_apis.append(self.api_embeds[index].view(1, -1))
            samples = del_tensor_rows(self.api_embeds, index, 1)
            num_sample = samples.size(0)
            selected_embed = torch.cat(selected_apis, dim=0)

            multi = torch.mul(samples.unsqueeze(1), selected_embed.unsqueeze(0))
            sub = torch.sub(samples.unsqueeze(1), selected_embed.unsqueeze(0))
            input = torch.cat([samples.unsqueeze(1).repeat(1, selected_embed.size(0), 1),
                               selected_embed.unsqueeze(0).repeat(num_sample, 1, 1), multi, sub], dim=-1)
            degree = self.attention_mlp(input).view(num_sample, -1)
            del input
            gc.collect()
            degree = self.softmax(degree)
            selected_embed = torch.mul(degree.unsqueeze(-1), selected_embed.unsqueeze(0))
            selected_embed = selected_embed.sum(1)
            input_feature = torch.cat([x_item.repeat(num_sample, 1), selected_embed, samples], dim=-1)
            pred = self.interaction(input_feature).view(-1)
            del input_feature
            gc.collect()
            pred = torch.cat([pred[:index], torch.tensor([1.0]).to(self.device), pred[index:]], dim=0)
            preds.append(pred.unsqueeze(0))
        preds = torch.cat(preds, dim=0)

        return {
            'preds': preds,
            'targets': y
        }

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
