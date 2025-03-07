import pickle
import os
import torch

remove_rate, minimal = 50, 10
file_name = f"data/views/views-{remove_rate}%-{minimal}.pkl"

with open(file_name, "rb") as f:
    views = pickle.load(f)
    pos_view = views['pos_view']
    neg_view = views['neg_view']

    num_pos, num_neg = 0, 0
    for i in range(len(pos_view)):
        if len(neg_view[i] != 0):
            num_pos += pos_view[i].shape[1]
            num_neg += neg_view[i].shape[1]
    
    print(num_pos / num_neg)
