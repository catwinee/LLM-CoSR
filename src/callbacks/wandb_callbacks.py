from typing import List, Any, Optional

import os
import csv
import hydra
import numpy as np
import torch
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.types import STEP_OUTPUT
import time

from hydra.core.global_hydra import GlobalHydra
from src.utils.metrics import Precision, Recall, NormalizedDCG, F1ScoreMetrics, MAP


def get_wandb_logger(trainer: Trainer) -> WandbLogger:
    """Safely get Weights&Biases logger from Trainer."""

    if trainer.fast_dev_run:
        raise Exception(
            "Cannot use wandb callbacks since pytorch lightning disables loggers in `fast_dev_run=true` mode."
        )

    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    # if isinstance(trainer.logger, LoggerCollection):
    for logger in trainer.logger:
        if isinstance(logger, WandbLogger):
            return logger

    raise Exception(
        "You are using wandb related callback, but WandbLogger was not found for some reason..."
    )


class WatchModel(Callback):
    """Make wandb watch model at the beginning of the run."""

    def __init__(self, log: str = "gradients", log_freq: int = 100):
        self.log = log
        self.log_freq = log_freq

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        logger.watch(model=trainer.model, log="all", log_freq=self.log_freq)


class LogMetricsAndRunningTime(Callback):
    def __init__(self, top_k_list: List[int], device='cuda', name = None, seed = None):
        self.top_ks = top_k_list
        self.device = device
        self.name = name
        self.seed = seed
        self.mashup_emb = []
        self.preds = []
        self.targets = []
        self.training_epoch_times = []
        self.test_epoch_times = []

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.propensity_score = trainer.datamodule.propensity_score.to(self.device)

        if self.device == 'cuda':
            self.precisions = [Precision(top_k=k).cuda() for k in self.top_ks]
            self.recalls = [Recall(top_k=k).cuda() for k in self.top_ks]
            self.NDCGs = [torchmetrics.RetrievalNormalizedDCG(k=k).cuda() for k in self.top_ks]
            self.F1s = [F1ScoreMetrics(top_k=k).cuda() for k in self.top_ks]
            self.PSPs = [Precision(top_k=k, propensity_score=self.propensity_score).cuda() for k in self.top_ks]
            self.PSDCGs = [NormalizedDCG(top_k=k, propensity_score=self.propensity_score).cuda() for k in self.top_ks]
            self.MRR = torchmetrics.RetrievalMRR().cuda()
            self.MAPs = [MAP(top_k=k).cuda() for k in self.top_ks]
        elif self.device == 'cpu':
            self.precisions = [Precision(top_k=k) for k in self.top_ks]
            self.recalls = [Recall(top_k=k) for k in self.top_ks]
            self.NDCGs = [torchmetrics.RetrievalNormalizedDCG(top_k=k) for k in self.top_ks]
            self.F1s = [F1ScoreMetrics(top_k=k) for k in self.top_ks]
            self.PSPs = [Precision(top_k=k, propensity_score=self.propensity_score) for k in self.top_ks]
            self.PSDCGs = [NormalizedDCG(top_k=k, propensity_score=self.propensity_score) for k in self.top_ks]
            self.MRR = torchmetrics.RetrievalMRR()
            self.MAPs = [MAP(top_k=k) for k in self.top_ks]
        else:
            raise Exception('unknown device!')

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.training_epoch_start_time = time.time()

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", unused = None
    ) -> None:
        self.training_epoch_times.append(time.time() - self.training_epoch_start_time)

    def on_test_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.preds.append(outputs['preds'])
        self.targets.append(outputs['targets'])

        # self.mashup_emb.append(outputs["mashup_emb"].cpu().numpy())


    def on_test_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.test_epoch_start_time = time.time()

    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.test_epoch_times.append(time.time() - self.test_epoch_start_time)

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment
        log_data = {}
        csv_columns = ["name", "seed"]
        metrics = ["F1", "Precision", "Recall", "NDCG", "PSP", "PSDCG", "MAP"]
        for top_k in self.top_ks:
            for metric in metrics:
                csv_columns.append(f"{metric}@{top_k}")
        csv_columns.append("MRR")
        row_data = {"name": self.name, "seed": self.seed}

        for pred, target in zip(self.preds, self.targets):
            for p in self.precisions:
                p.update(pred, target)
            for r in self.recalls:
                r.update(pred, target)
            for n in self.NDCGs:
                n.update(pred, target, torch.tensor(range(pred.size(0))).unsqueeze(1).repeat(1, pred.size(1)))
            for f in self.F1s:
                f.update(pred, target)
            for p in self.PSPs:
                p.update(pred, target)
            for p in self.PSDCGs:
                p.update(pred, target)
            for p in self.MAPs:
                p.update(pred, target)
            self.MRR.update(pred, target, torch.tensor(range(pred.size(0))).unsqueeze(1).repeat(1, pred.size(1)))
        for top_k, p, r, n, f, psp, psdcg, map in zip(
            self.top_ks, self.precisions, self.recalls, self.NDCGs,
            self.F1s, self.PSPs, self.PSDCGs, self.MAPs
        ):
        # for top_k, p, r, n, f, map in zip(
        #     self.top_ks, self.precisions, self.recalls,
        #     self.NDCGs, self.F1s, self.MAPs
        # ):
            p_result = p.compute()
            r_result = r.compute()
            n_result = n.compute()
            f_result = f.compute()
            psp_result = psp.compute()
            psdcg_result = psdcg.compute()
            map_result = map.compute()

            log_data[f'test/precision@{top_k}'] = p_result
            log_data[f'test/recall@{top_k}'] = r_result
            log_data[f'test/NDCG@{top_k}'] = n_result
            log_data[f'test/F1@{top_k}'] = f_result
            log_data[f'test/PSP@{top_k}'] = psp_result
            log_data[f'test/PSDCG@{top_k}'] = psdcg_result
            log_data[f'test/MAP@{top_k}'] = map_result

            row_data[f"Precision@{top_k}"] = round(p.compute().item(), 4)
            row_data[f"Recall@{top_k}"] = round(r.compute().item(), 4)
            row_data[f"NDCG@{top_k}"] = round(n.compute().item(), 4)
            row_data[f"F1@{top_k}"] = round(f.compute().item(), 4)
            row_data[f"PSP@{top_k}"] = round(psp.compute().item(), 4)
            row_data[f"PSDCG@{top_k}"] = round(psdcg.compute().item(), 4)
            row_data[f"MAP@{top_k}"] = round(map.compute().item(), 4)

        row_data["MRR"] = round(self.MRR.compute().item(), 4)
        log_data['test/MRR'] = self.MRR.compute()

        log_data['epoch_time/training'] = np.mean(self.training_epoch_times)
        log_data['epoch_time/test'] = np.mean(self.test_epoch_times)

        experiment.log(log_data, commit=True)

        # mashup_emb = np.concatenate(self.mashup_emb, axis=0)  # [total_mashups, dim]

        # targets = torch.cat(self.targets, dim=0).cpu().numpy()  # [total_mashups, num_apis]
        # labels = np.argmax(targets, axis=1)  # [total_mashups]

        # from sklearn.manifold import TSNE
        # import matplotlib
        # matplotlib.use('Agg')
        # import matplotlib.pyplot as plt
        # from sklearn.preprocessing import StandardScaler  # 新增标准化

        # scaler = StandardScaler()
        # mashup_emb_scaled = scaler.fit_transform(mashup_emb) 

        # tsne = TSNE(
        #     n_components=2,
        #     perplexity=30,
        #     learning_rate=500,
        #     n_iter=20000,
        #     early_exaggeration=24,
        #     metric="cosine",
        #     random_state=42,
        # )
        # mashup_2d = tsne.fit_transform(mashup_emb_scaled)

        # plt.figure(figsize=(10, 8))
        # scatter = plt.scatter(
        #     mashup_2d[:, 0], mashup_2d[:, 1], c=labels, cmap='viridis', alpha=0.6
        # )
        # plt.colorbar(scatter)
        # plt.title(f"t-SNE Visualization of Mashup Embeddings for {self.name}")
        # plt.savefig(f'{hydra.utils.get_original_cwd()}/pic/tsne-{self.name}.png', bbox_inches='tight')
        # plt.close()



        # preds = torch.cat(self.preds)
        # probs = torch.sigmoid(preds)
        # _, topk_indices = torch.topk(probs, k=5, dim=1)

        # num_apis = probs.size(1)

        # recommendation_mask = torch.zeros((topk_indices.size(0), num_apis), 
        #                                 dtype=torch.long, device=topk_indices.device)
        # recommendation_mask.scatter_(1, topk_indices, 1)
        # recommend_counts = recommendation_mask.sum(dim=0).float()
        # recommend_probs = recommend_counts / topk_indices.size(0)
        # sorted_probs, sorted_indices = torch.sort(recommend_probs, descending=True)

        # print("API推荐概率:")
        # for idx, prob in zip(sorted_indices.tolist()[:10], sorted_probs.tolist()[:10]):
        #     print(f"API {idx}: {prob:.4f}")
        
        csv_file = os.path.join(hydra.utils.get_original_cwd(), "results.csv")
        with open(csv_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_columns)
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(row_data)