import torch.nn as nn
import torch
import numpy as np
import dgl
from dgl import function

class AdaptiveHyperNN(nn.Module):
    def __init__(self,  api_embed_channels, aggre_type, allow_zero_in_degree=False, device = None):
        super(AdaptiveHyperNN, self).__init__()
        self.GNN = GNN(api_embed_channels, aggre_type, allow_zero_in_degree, device)
        self.MLP = MLP(api_embed_channels * 3, api_embed_channels, 2, device)
        self.softmax = nn.Softmax(dim=0)
        self.edge_exist_mlp = nn.Sequential(nn.Linear(api_embed_channels * 2, 1), nn.Sigmoid())

    def forward(self, Xs, Ys, api_embeds):
        graphs = []
        graphs_edges = []
        graphs_nodes_feats = []
        for y in Ys:
            invoked_apis = y.nonzero(as_tuple=True)[0]
            api_feats = api_embeds[invoked_apis]
            u, v = (np.ones((invoked_apis.shape[0], invoked_apis.shape[0]))).nonzero()
            graphs_nodes_feats.append(api_feats)
            graphs_edges.append((u, v))
            g = (dgl.graph((u, v))).to("cuda:0")
            g.ndata['feat'] = api_feats
            graphs.append(g)
        outputs = self.GNN(graphs)
        res = []
        for output, x in zip(outputs, Xs):
            x = torch.unsqueeze(x, 0).repeat(output.shape[0], 1)
            output = torch.cat([output, x], dim=1)
            #output = self.MLP(output)
            #res.append(self.softmax(output))
            res.append(self.edge_exist_mlp(output))
        return res


class MLP(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, device = None):
        super(MLP, self).__init__()
        self.device = device
        self.mlp = nn.Sequential(
            nn.Linear(input_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, output_channels)
        )

    def forward(self, input):
        return self.mlp(input)


class GNN(nn.Module):
    def __init__(self, api_embed_channels, aggre_type, allow_zero_in_degree=False, device = None):
        super(GNN, self).__init__()
        self.device = device
        self._aggre_type = aggre_type
        self._allow_zero_in_degree = allow_zero_in_degree
        self.api_embed_channels = api_embed_channels
        self.mlp = MLP(api_embed_channels * 3, api_embed_channels * 2, api_embed_channels)
        self.n2v_mlp1 = nn.Sequential(
            nn.Linear(api_embed_channels * 2, api_embed_channels),
            #nn.ReLU(inplace=True),
            #TODO Nor
        )
        self.n2n_mlp2 = nn.Sequential(
            nn.Linear(api_embed_channels * 2, api_embed_channels),
            #nn.ReLU(inplace=True),
            # TODO Nor?
        )
        self.n2v_mlp3 = nn.Sequential(
            nn.Linear(api_embed_channels * 2, api_embed_channels),
            #nn.ReLU(inplace=True)
        )

        #self.edge_exist_mlp = nn.Sequential(nn.Linear(api_embed_channels * 2, 1), nn.Sigmoid())



    def forward(self, graphs):
        # if not self._allow_zero_in_degree:
        #     for graph in graphs:
        #         if (graph.in_degrees() == 0).any():
        #             raise dgl.DGLError(
        #                 "There are 0-in-degree nodes in the graph, "
        #                 "output for those nodes will be invalid. "
        #                 "This is harmful for some applications, "
        #                 "causing silent performance regression. "
        #                 "Adding self-loop on the input graph by "
        #                 "calling `g = dgl.add_self_loop(g)` will resolve "
        #                 "the issue. Setting ``allow_zero_in_degree`` "
        #                 "to be `True` when constructing this module will "
        #                 "suppress the check and let the code run."
        #             )

        # def concat_message_function(edges):
        #     return {'cat_feat': torch.cat([edges.src['feat'], edges.dst['feat']], dim=1)}
        edges = []
        for graph in graphs:
            # graph.apply_edges(concat_message_function)
            graph.update_all(lambda  edges: {'m': self.n2v_mlp1(torch.cat([edges.src['feat'], edges.dst['feat']], dim=1))}, # n -> e
                             function.mean("m", "e2n1")
                             )
            graph.ndata["e2n1"] = self.n2n_mlp2(torch.cat([graph.ndata['feat'], graph.ndata['e2n1']], dim=1))
            graph.apply_edges(lambda edges: {"ef": self.n2v_mlp3(torch.cat([edges.src['e2n1'], edges.dst['e2n1']], dim=1))})
            #graph.edata['ew'] = self.edge_exist_mlp(graph.edata['ef'])
            edges.append(graph.edata['ef'])
            #graph.update_all(lambda edges: {'cat_feat': torch.cat([edges.src['feat'], edges.dst['feat']], dim=1)}, function.mean("cat_feat", "m"))

        # if self._aggre_type == 'mean':
        #     for graph in graphs:
        #         graph.apply_nodes(function.mean("m", "cat_feat"))
        # elif self._aggre_type == 'sum':
        #     for graph in graphs:
        #         graph.apply_nodes(function.sum("m", "cat_feat"))

        # for graph in graphs:
        #     feature = (torch.cat([graph.ndata['m'], graph.ndata['feat']], dim=1)).to("cuda:0")
        #     graph.ndata['feat'] = self.mlp(feature)
        #
        # edges = []
        # for graph in graphs:
        #     graph.apply_edges(lambda edges: {'cat_feat': torch.cat([edges.src['feat'], edges.dst['feat']], dim=1)})
        #     edges.append(graph.edata['cat_feat'])
        return edges