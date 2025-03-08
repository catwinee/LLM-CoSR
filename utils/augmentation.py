from typing import Dict

import os, json, ast, torch, pickle, random
import numpy as np
import pandas as pd
import concurrent
from tqdm import tqdm
from openai import OpenAI
from itertools import combinations


class AugmentGenerator:
    def __init__(
        self,
        remove_rate: int,
        minimal: int,
        data_dir: str = os.path.join(os.getcwd(), "data"),
        api_path: str = 'api_mashup/raw/cleaned_apis_data.txt',
        mashup_path: str = 'api_mashup/raw/active_mashups_data.txt',
        invoked_path: str = f'api_mashup/train_partial_invocation_seed=12345.pkl',
        mapping_path: str = 'api-mapping.pkl',
    ):
        self.mashups = json.load(open(os.path.join(data_dir, mashup_path), encoding="utf-8"))
        self.total_apis = json.load(open(os.path.join(data_dir, api_path), encoding="utf-8"))
        self.invocation_df = pd.read_pickle(os.path.join(data_dir, invoked_path))
        mapping = pd.read_pickle(os.path.join(data_dir, mapping_path))

        self.apis = [None] * len(mapping)
        for k, v in mapping.items():
            self.apis[v] = self.total_apis[k]

        self.num_mashups = len(self.mashups)
        self.num_apis = len(self.apis)

        api_key = open("hs.key", "r").read().strip()
        base_url = "https://ark.cn-beijing.volces.com/api/v3"
        self.model = "ep-20250206145346-jn9kv"

        self.prompt = f"""You are a graph data processing assistant. Given a graph's edge index (`edge_index`) and textual descriptions of its nodes,
 your task is to remove abnormal edges according to their titles, tags, and description, then return the updated `edge_index`.
### Your Task: Update edge_index: A list of edges after removing abnormal edges. You should remove about {remove_rate}% of given edges.
### Output: - Updated `edge_index`: A list of edges after removing all abnormal edges.
### Example Output: [[0, 1], [0, 3], [1, 3], [2, 3], [2, 4], [3, 4]]
### REMEMBER Don't explain, just follow the output format and return a python list! You should remove about {remove_rate}% of given edges.
"""
        self.remove_rate = remove_rate
        self.minimal = minimal
        self.client = OpenAI(api_key=api_key, base_url=base_url)
    
    def get_response(self, edge_index, node_set, idx, test=True):
        if (len(edge_index) < 0.7 * self.minimal):
            print("Too less, skip.")
            keep_num = int(len(edge_index) * (1 - self.remove_rate/100))
            response_view = random.sample(edge_index, keep_num)
            return response_view, edge_index, idx
        input = "Edge_index: " + str(edge_index) + "Node Information: "
        for node in node_set:
            input += f"Node {node}:" + str(self.apis[node])
        for _ in range(3):
            response_view = self.llm(input, edge_index)
            if len(response_view) <= (110 - self.remove_rate) / 100 * len(edge_index):
                return response_view, edge_index, idx
            elif len(response_view[0]) != 2: 
                print("Shape Error")
            else:
                print(len(response_view), ", ", len(edge_index), ": ", len(response_view) / len(edge_index))

        print("Fail too much times. ")
        keep_num = int(len(edge_index) * (1 - self.remove_rate/100))
        response_view = random.sample(edge_index, keep_num)
        return response_view, edge_index, idx

    def dry_run(self, edge_index, node_set, idx):
        if (len(edge_index) < 0.7 * self.minimal):
            print("Too less, skip.")
            keep_num = int(len(edge_index) * (1 - self.remove_rate/100))
            response_view = random.sample(edge_index, keep_num)
            return response_view, edge_index, idx
        input = "Edge_index: " + str(edge_index) + "Node Information: "
        for node in node_set:
            input += f"Node {node}:" + str(self.apis[node])
        for _ in range(5):
            response_view = edge_index[::2]
            if len(response_view) <= (110 - self.remove_rate) / 100 * len(edge_index):
                return response_view, edge_index, idx
            elif len(response_view[0]) != 2: 
                print("Shape Error")
            else:
                print(len(response_view), ", ", len(edge_index), ": ", len(response_view) / len(edge_index))

        print("Fail too much times. ")
        keep_num = int(len(edge_index) * (1 - self.remove_rate/100))
        response_view = random.sample(edge_index, keep_num)
        return response_view, edge_index, idx

    def api_api(self):
        file_name = f"data/views/views-{self.remove_rate}%-{self.minimal}.pkl"
        if (os.path.exists(file_name)):
            print("OverWrite?")
            return

        neg_view = [torch.tensor([], dtype=torch.long, device='cuda')] * self.num_mashups
        pos_view = [torch.tensor([], dtype=torch.long, device='cuda')] * self.num_mashups
        api_api_view = [torch.tensor([], dtype=torch.long, device='cuda')] * self.num_mashups
        for idx, node_indices in zip(self.invocation_df['X'], self.invocation_df['Y']):
            edge_index = list(combinations(node_indices, 2))
            edge_index = [list(edge) for edge in edge_index]
            api_api_view[idx] = edge_index

        with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
            futures = []
            total_view, node_set = [], set()
            for idx in range(self.num_mashups):
            # for idx in range(200): # for test
                if api_api_view[idx] is None:
                    continue
                else:
                    total_view.extend(api_api_view[idx])
                    node_set.update(*api_api_view[idx])
                if len(total_view) > self.minimal:
                    future = executor.submit(self.get_response, total_view, node_set, idx)
                    futures.append(future)
                    total_view, node_set = [], set()
            
            future = executor.submit(self.get_response, total_view, node_set, idx)
            futures.append(future)

            with tqdm(total=len(futures), file=sys.stdout) as pbar:
                for future in concurrent.futures.as_completed(futures):
                    response_view, total_view, idx = future.result()
                    response_view = torch.tensor(response_view, dtype=torch.long, device='cuda').t()
                    total_view = torch.tensor(total_view, dtype=torch.long, device='cuda').t()
                    neg_view[idx], pos_view[idx] = total_view, response_view
                    
                    pbar.update(1)
                    pbar.refresh()

        try:
            num_pos, num_neg = 0, 0
            for i in range(len(pos_view)):
                if len(neg_view[i] != 0):
                    num_pos += pos_view[i].shape[1]
                    num_neg += neg_view[i].shape[1]
            
            print("Real remove rate: ", 1 - num_pos / num_neg)
        except:
            pass

        with open(file_name, "wb") as f:
            pickle.dump({"pos_view": pos_view, "neg_view": neg_view}, f)

    def llm(self, input, edge_index):
        messages = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": input}
        ]

        for _ in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stream=False
                ).choices[0].message.content
                edge_list = ast.literal_eval(response)
            except Exception as e:
                print('Parse failed:', e, response)
            else:
                return edge_list

        print("Failed too many times.")
        keep_num = int(len(edge_index) * (1 - self.remove_rate/100))
        response_view = random.sample(edge_index, keep_num)
        return response_view

if __name__ == '__main__':
    import sys
    AugmentGenerator(int(sys.argv[1]), 20).api_api()
    # remove_rate, minimal
