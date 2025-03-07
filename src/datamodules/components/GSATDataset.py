import random
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class GSATDataset(Dataset):
    def __init__(self, edges, mashup_nums, api_nums, gpus, dataset, n_negs=8) -> None:
        super().__init__()
        self.mashup_nums = mashup_nums
        self.api_nums = api_nums
        self.edges = edges # [K, 2]，K表示表中存在的连接数量
        self.device = torch.device("cuda:0") if gpus else torch.device("cpu")
        self.n_negs = n_negs  # 采样多少负例
        self.user_set = defaultdict(list)
        self.dataset = dataset
        for u_id, i_id in edges:
            self.user_set[int(u_id)].append(int(i_id))
        self.user_list = []
        self.api_list = []
        for user in self.user_set:
            self.user_list.append(user)
            self.api_list.append(self.user_set[user])


    def __len__(self):
        return len(self.user_set)

    def __getitem__(self, index):
        return self.get_feed_dict(index)

    def get_feed_dict(self, index, K=1):

        def sampling(user, train_set, n):
            neg_items = []
            negitems = []
            for i in range(n):  # sample n times
                while True:
                    negitem = random.choice(range(self.api_nums))
                    if negitem not in train_set[user]:
                        break
                negitems.append(negitem)
            neg_items.append(negitems)
            return neg_items

        feed_dict = {}
        feed_dict['users'] = self.user_list[index]
        api_list = torch.zeros(self.api_nums)

        for api in self.api_list[index]:
            if self.dataset == 'Mashup':
                if type(api) == list:
                    for i in api:
                        api_list[i-self.mashup_nums] = 1
                else:
                    api_list[api-self.mashup_nums] = 1
            else:
                if type(api) == list:
                    for i in api:
                        api_list[i] = 1
                else:
                    api_list[api] = 1
        feed_dict['pos_items'] = api_list
        neg_items = sampling(self.user_list[index], self.user_set, self.n_negs * K)[0][:len(self.api_list[index])]


        neg_api_list = torch.zeros(self.api_nums)

        if type(neg_items) == list:
            for i in neg_items:
                neg_api_list[i] = 1
        else:
            neg_api_list[neg_items] = 1
        feed_dict['neg_items'] = neg_api_list
        # if torch.nonzero(feed_dict['neg_items'], as_tuple=False).shape[0] != torch.nonzero(feed_dict['pos_items'], as_tuple=False).shape[0]:
        #     print('error')
        # feed_dict['neg_items'] = torch.LongTensor(sampling(self.user_list[index],
        #                                                    self.user_set,
        #                                                    self.n_negs * K)).to(self.device)
        return feed_dict