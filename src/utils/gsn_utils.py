import os
from typing import Optional
from tqdm import tqdm

import graph_tool as gt
import graph_tool.topology as gt_topology
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
import networkx as nx
import types
from torch_geometric.utils import remove_self_loops

# Now re-implement torch_geometrics's degree function without that package:
def maybe_num_nodes(edge_index, num_nodes=None):
    if num_nodes is not None:
        return num_nodes
    elif isinstance(edge_index, Tensor):
        return int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0
    else:
        return max(edge_index.size(0), edge_index.size(1))


def degree(index: Tensor, num_nodes: Optional[int] = None,
           dtype: Optional[torch.dtype] = None) -> Tensor:
    N = maybe_num_nodes(index, num_nodes)
    out = torch.zeros((N,), dtype=dtype, device=index.device)
    one = torch.ones((index.size(0),), dtype=out.dtype, device=out.device)
    return out.scatter_add_(0, index, one)


def subgraph_isomorphism_vertex_counts(
        edge_index,
        subgraph_dict,
        induced,
        num_nodes,
        is_directed=False
    ):
    directed = is_directed

    G_gt = gt.Graph(directed=directed)
    G_gt.add_edge_list(list(edge_index.transpose(1, 0).cpu().numpy()))
    gt.generation.remove_self_loops(G_gt)
    gt.generation.remove_parallel_edges(G_gt)

    # compute all subgraph isomorphisms
    sub_iso = gt_topology.subgraph_isomorphism(
        subgraph_dict['subgraph'],
        G_gt,
        induced=induced,
        subgraph=True,
        generator=True
    )

    ## num_nodes should be explicitly set for the following edge case:
    ## when there is an isolated vertex whose index is larger
    ## than the maximum available index in the edge_index

    counts = np.zeros((num_nodes, len(subgraph_dict['orbit_partition'])))
    for sub_iso_curr in tqdm(sub_iso, position=2, leave=False, desc='iso', delay=5):
        for i, node in tqdm(enumerate(sub_iso_curr), position=3, leave=False, desc='node', delay=5):
            # increase the count for each orbit
            counts[node, subgraph_dict['orbit_membership'][i]] += 1
    counts = counts / subgraph_dict['aut_count']

    counts = torch.tensor(counts)

    return counts


def automorphism_orbits(edge_list, print_msgs=False, **kwargs):
    ##### vertex automorphism orbits #####

    directed = kwargs['directed'] if 'directed' in kwargs else False

    graph = gt.Graph(directed=directed)
    gt.Graph()
    graph.add_edge_list(edge_list)
    gt.generation.remove_self_loops(graph)
    gt.generation.remove_parallel_edges(graph)

    # compute the vertex automorphism group
    aut_group = gt_topology.subgraph_isomorphism(graph, graph, induced=False, subgraph=True, generator=False)

    orbit_membership = {}
    for v in graph.get_vertices():
        orbit_membership[v] = v

    # whenever two nodes can be mapped via some automorphism, they are assigned the same orbit
    for aut in aut_group:
        for original, vertex in enumerate(aut):
            role = min(original, orbit_membership[vertex])
            orbit_membership[vertex] = role

    orbit_membership_list = [[], []]
    for vertex, om_curr in orbit_membership.items():
        orbit_membership_list[0].append(vertex)
        orbit_membership_list[1].append(om_curr)

    # make orbit list contiguous (i.e. 0,1,2,...O)
    _, contiguous_orbit_membership = np.unique(orbit_membership_list[1], return_inverse=True)

    orbit_membership = {vertex: contiguous_orbit_membership[i] for i, vertex in enumerate(orbit_membership_list[0])}

    orbit_partition = {}
    for vertex, orbit in orbit_membership.items():
        orbit_partition[orbit] = [vertex] if orbit not in orbit_partition else orbit_partition[orbit] + [vertex]

    aut_count = len(aut_group)

    if print_msgs:
        print('Orbit partition of given substructure: {}'.format(orbit_partition))
        print('Number of orbits: {}'.format(len(orbit_partition)))
        print('Automorphism count: {}'.format(aut_count))

    return graph, orbit_partition, orbit_membership, aut_count


def pad_tensors(tensors, pad_value=0):
    """
    为了将他count出来的结果第1维度长度一致，只能进行padding了
    """
    # 确定第一维的最大长度
    max_len = max(tensor.size(1) for tensor in tensors)

    # pad每个tensor到最大长度
    padded_tensors = []
    for tensor in tensors:
        # 计算需要在每侧填充的长度
        padding = (0, max_len - tensor.size(1))
        # 在第一维进行padding
        padded_tensor = F.pad(tensor, pad=(0, max_len - tensor.size(1)), value=pad_value)
        padded_tensors.append(padded_tensor)

    return padded_tensors

def generate_subgraph(id_type, k, custom_edge_list):

    ###### choose the substructures: usually loaded from networkx,
    ###### except for 'all_simple_graphs' where they need to be precomputed,
    ###### or when a custom edge list is provided in the input by the user

    if id_type in ['cycle_graph',
                           'path_graph',
                           'complete_graph',
                           'binomial_tree',
                           'star_graph',
                           'nonisomorphic_trees']:
        k_max = k
        k_min = 2 if id_type == 'star_graph' else 3
        custom_edge_list = get_custom_edge_list(list(range(k_min, k_max + 1)), id_type)

    elif id_type in ['cycle_graph_chosen_k',
                             'path_graph_chosen_k',
                             'complete_graph_chosen_k',
                             'binomial_tree_chosen_k',
                             'star_graph_chosen_k',
                             'nonisomorphic_trees_chosen_k']:
        custom_edge_list = get_custom_edge_list(list(range(k)), id_type.replace('_chosen_k',''))

    elif id_type in ['all_simple_graphs']:

        k_max = k
        k_min = 3
        filename = None
        custom_edge_list = get_custom_edge_list(list(range(k_min, k_max + 1)), filename = filename)

    elif id_type in ['all_simple_graphs_chosen_k']:
        filename = None
        custom_edge_list = get_custom_edge_list(k, filename = filename)

    elif id_type in ['diamond_graph']:
        graph_nx = nx.diamond_graph()
        custom_edge_list = [list(graph_nx.edges)]

    elif id_type == 'custom':
        assert custom_edge_list is not None, "Custom edge list must be provided."

    else:
        raise NotImplementedError("Identifiers {} are not currently supported.".format(id_type))

    return custom_edge_list
def get_custom_edge_list(ks, substructure_type=None, filename=None):
    '''
        Instantiates a list of `edge_list`s representing substructures
        of type `substructure_type` with sizes specified by `ks`.
    '''
    if substructure_type is None and filename is None:
        raise ValueError('You must specify either a type or a filename where to read substructures from.')
    edge_lists = []
    for k in ks:
        if substructure_type is not None:
            graphs_nx = getattr(nx, substructure_type)(k)
        else:
            graphs_nx = nx.read_graph6(os.path.join(filename, 'graph{}c.g6'.format(k)))
        if isinstance(graphs_nx, list) or isinstance(graphs_nx, types.GeneratorType):
            edge_lists += [list(graph_nx.edges) for graph_nx in graphs_nx]
        else:
            edge_lists.append(list(graphs_nx.edges))
    return edge_lists


def subgraph_counts2ids(edge_index ,num_nodes , subgraph_dicts, induced, is_directed):
    #### Remove self loops and then assign the structural identifiers by computing subgraph isomorphisms ####


    edge_index = remove_self_loops(edge_index)[0]

    counting = None
    for subgraph_dict in tqdm(subgraph_dicts, position=1, leave=False, delay=1, desc='Subgraph'):
        count = subgraph_isomorphism_vertex_counts(edge_index,
                                                      subgraph_dict,
                                                      induced,
                                                      num_nodes,
                                                      is_directed)
        counting = count if counting is None else torch.cat((counting, count), 1)


    return counting