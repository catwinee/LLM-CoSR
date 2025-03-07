import os.path
from typing import Optional, List

import numpy as np
import pandas as pd
import torch
import hydra
import pickle
from tqdm import tqdm
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import DataLoader, random_split
from src.datamodules.components.mashup_dataset import MashupDataset
from src.utils.gsn_counting import generate_counting

import scipy.io as sio


class MashupDataModule(LightningDataModule):
    r"""A dataset of mashups and APIs. This dataset includes 4557 mashups and two types of API:

    1. partial
        932 APIs, all of which have been used at least once.

    2. total
        21495 APIs, including some unused APIs.

    Args:
        data_dir (str): Path to the folder where the data is located.
        num_candidates (int): The number of candidates of API.
        mashup_path (str): Path to mashups. Relative to :attr:`data_dir`.
        invoked_path (str): Path to invocation between mashups and APIs. Relative to :attr:`data_dir`.
        api_path (str): Path to APIs. Relative to :attr:`data_dir`.
        train_val_test_split (List[int]): List of the number of training samples, validation samples,
            and test samples.
        pair_in_training (bool): If set to :obj:`False`, will not return API embeddings in training stage.
            (default: :obj:`True`)
        negative_samples_ratio (int): Ratio of negative to positive in the training stage. (default: :obj:`5`).
        batch_size (int): The number of samples per batch to load. (default: :obj:`64`).
        num_workers (int): The number of subprocesses to use for data loading. 0 means that the data will be loaded
            in the main process. (default: :obj:`1`)
        pin_memory (bool): If set to :obj:`True`, the data loader will copy Tensors into device/CUDA pinned memory
            before returning them. (default: :obj:`False`)
        mashup_index (bool): If set to :obj:`True`, Will return the index of the mashup instead of the embedding
            vector. (default: :obj:`False`)
        api_index (bool): If set to :obj:`True`, will return the index of the API instead of the embedding
            vector. (default: :obj:`False`)
        is_orderly (bool): If set to :obj:`True`, will return data in chronological order. (default: :obj:`False`)
    """
    def __init__(
        self,
        data_dir: str,
        num_candidates: int,
        mashup_path: str,
        invoked_path: str,
        api_path: str,
        train_val_test_split: List[int],
        pair_in_training: bool = True,
        negative_samples_ratio: int = 5,
        batch_size: int = 64,
        num_workers: int = 1,
        pin_memory: bool = False,
        mashup_index: bool = False,
        api_index: bool = False,
        is_orderly: bool = False,
        is_gsn: bool = False,
        need_idx: bool = False,
        need_val_idx: bool = False,
    ):
        super(MashupDataModule, self).__init__()
        self.data_dir = data_dir
        self.num_candidates = num_candidates
        self.mashup_path = mashup_path
        self.api_path = api_path
        self.invoked_path = invoked_path
        self.pair_in_training = pair_in_training
        self.negative_samples_ratio = negative_samples_ratio
        self.train_val_test_split = train_val_test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.mashup_index = mashup_index
        self.api_index = api_index
        self.mashup_transform = None
        self.api_transform = None
        self.train_invocation = None
        self.val_invocation = None
        self.test_invocation = None
        self.is_gsn = is_gsn
        self.is_orderly = is_orderly
        self.need_idx = need_idx
        self.need_val_idx = need_val_idx
        if is_orderly:
            self.shuffle = False
        else:
            self.shuffle = True
        self.propensity_score = None

    def setup(self, stage: Optional[str] = None) -> None:
        invocation_df = pd.read_pickle(os.path.join(self.data_dir, self.invoked_path))
        mashups = np.load(os.path.join(self.data_dir, self.mashup_path))
        apis = np.load(os.path.join(self.data_dir, self.api_path))
        num_mashups = len(mashups)
        num_apis = len(apis)

        if self.mashup_index:
            self.mashup_transform = lambda x: x
        else:
            self.mashup_transform = lambda x: mashups[x]
        if self.api_index:
            self.api_transform = lambda x: x
        else:
            self.api_transform = lambda x: apis[x]
        
        seed = 12345

        if self.is_orderly:
            train_idx = list(range(self.train_val_test_split[0]))
            val_idx = list(range(self.train_val_test_split[0], self.train_val_test_split[0] + self.train_val_test_split[1]))
            test_idx = list(range(self.train_val_test_split[1], self.train_val_test_split[1] + self.train_val_test_split[2]))
        else:
            train_idx, val_idx, test_idx = random_split(
                range(num_mashups),
                self.train_val_test_split,
                generator = torch.Generator().manual_seed(seed)
            )
        self.train_invocation = invocation_df.iloc[list(train_idx)]
        # mashup -> api[]
        pickle.dump(self.train_invocation, open(
            self.data_dir + f"/train_partial_invocation_seed={seed}.pkl", "wb"))
        self.val_invocation = invocation_df.iloc[list(val_idx)]
        self.test_invocation = invocation_df.iloc[list(test_idx)]

        propensity_score = torch.ones(num_apis)
        for index in invocation_df['Y']:
            propensity_score[index] += 1
        self.propensity_score = propensity_score / torch.sum(propensity_score)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        dataset = MashupDataset(
            invocation=self.train_invocation,
            num_candidates=self.num_candidates,
            mashup_transform=self.mashup_transform,
            api_transform=self.api_transform,
            is_orderly=self.is_orderly,
            is_triple=self.pair_in_training,
            need_idx=self.need_idx,
            negative_samples_ratio=self.negative_samples_ratio,
        )
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=self.shuffle,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        dataset = MashupDataset(
            invocation=self.val_invocation,
            num_candidates=self.num_candidates,
            mashup_transform=self.mashup_transform,
            api_transform=self.api_transform,
            is_orderly=self.is_orderly,
            is_triple=False,
            need_idx=self.need_val_idx,
            negative_samples_ratio=self.negative_samples_ratio,
        )
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        dataset = MashupDataset(
            invocation=self.test_invocation,
            num_candidates=self.num_candidates,
            mashup_transform=self.mashup_transform,
            api_transform=self.api_transform,
            is_orderly=self.is_orderly,
            is_triple=False,
            need_idx=self.need_val_idx,
            negative_samples_ratio=self.negative_samples_ratio,
        )
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
