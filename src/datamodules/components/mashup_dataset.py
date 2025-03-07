import os
import hydra
import pickle
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from datetime import datetime


class MashupDataset(Dataset):
    """

    """
    def __init__(
        self,
        invocation: pd.DataFrame,
        num_candidates: int,
        mashup_transform=None,
        api_transform=None,
        is_orderly: bool = False,
        is_triple: bool = False,
        need_idx: bool = False,
        negative_samples_ratio: int = 5,
    ):
        super().__init__()
        Xs = invocation['X'].tolist()
        Ys = invocation['Y'].tolist()
        if is_orderly:
            times = invocation['time']
            pack = list(zip(Xs, Ys, times))
            pack.sort(key=lambda x: datetime.strptime(x[2], '%m.%d.%Y'))
            Xs = []
            Ys = []
            for item in pack:
                Xs.append(item[0])
                Ys.append(item[1])
        self.num_candidates = num_candidates
        self.is_triple = is_triple
        self.need_idx = need_idx
        self.mashup_transform = mashup_transform
        self.api_transform = api_transform
        self.mashups, self.apis, self.labels = [], [], []

        if is_triple:
            for mashup_idx, invoked_apis in zip(Xs, Ys):
                # positive samples
                self.mashups.extend([mashup_idx for _ in invoked_apis])
                self.apis.extend(invoked_apis)
                self.labels.extend([1 for _ in invoked_apis])
                # negative samples
                negative_apis = np.random.choice(np.delete(np.arange(num_candidates), invoked_apis),
                                                 size=negative_samples_ratio * len(invoked_apis), replace=False)
                self.mashups.extend([mashup_idx for _ in negative_apis])
                self.apis.extend(negative_apis)
                self.labels.extend([0 for _ in negative_apis])
        else:
            self.mashups = Xs
            self.labels = Ys

    def __len__(self):
        return len(self.mashups)

    def __getitem__(self, idx):
        mashup = self.mashup_transform(self.mashups[idx])
        mashup = torch.from_numpy(mashup)
        if self.is_triple:
            api = self.api_transform(self.apis[idx])
            if isinstance(api, np.ndarray):
                api = torch.from_numpy(api)
            label = torch.tensor(self.labels[idx], dtype=torch.int64)
            return mashup, api, label
        label = nn.functional.one_hot(torch.LongTensor(self.labels[idx]), num_classes=self.num_candidates)
        label = label.sum(dim=0)
        if self.need_idx:
            return mashup, label, idx
        return mashup, label