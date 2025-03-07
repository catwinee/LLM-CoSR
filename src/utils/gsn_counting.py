from typing import List

import os, pickle, torch
from tqdm import tqdm
from src.utils.gsn_utils import automorphism_orbits, generate_subgraph, subgraph_counts2ids

def generate_counting(
        data_dir: str,
        id_type: str,
        k: int,
        subgraph_edge_list,
        edge_index: List[List[int]],
        num_nodes: int,
        split_size: int = 500,
        step_size: int = 100,
        induced: bool = False,
        is_directed: bool = True,
        method: str = "overwrite",
    ):

    file_path = data_dir + "/{}-k={}-n={}-sp={}-st={}-m={}.pkl".format(id_type, k, num_nodes, split_size, step_size, method)

    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            counting = pickle.load(file)
        tqdm.write('already existed, load in ' + file_path)
    else:
        subgraph_dicts = []
        subgraph_edge_list = generate_subgraph(id_type, k, subgraph_edge_list)
        for edge_list in subgraph_edge_list:
            subgraph, orbit_partition, orbit_membership, aut_count = automorphism_orbits (
                edge_list = edge_list,
                directed = False,
                directed_orbits = False
            )
            subgraph_dicts.append({
                'subgraph': subgraph, 
                'orbit_partition': orbit_partition,
                'orbit_membership': orbit_membership, 
                'aut_count': aut_count
            })

        num_edges = len(edge_index[0])
        counting = None

        print(f"ID_type: {id_type}, K: {k}, Split_size: {split_size}, Step_size: {step_size}, Method: {method}")
        for start_index in tqdm(range(0, num_edges, step_size)):
            partial_edge_index = edge_index[:, start_index: min(num_edges, start_index + split_size)]
            partial_counting = subgraph_counts2ids(
                partial_edge_index,
                num_nodes,
                subgraph_dicts,
                induced,
                is_directed
            )
            if counting is None:
                counting = partial_counting.float()
            else:
                mask = torch.any(partial_counting != 0.0, dim=1)
                if method == "overwrite":
                    counting[mask] = partial_counting.float()[mask]
                elif method == "add":
                    counting[mask] += partial_counting.float()[mask]

        with open(file_path, 'wb') as file:
            pickle.dump(counting, file)

    return counting