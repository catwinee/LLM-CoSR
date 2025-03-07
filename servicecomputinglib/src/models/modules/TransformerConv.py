from torch import nn

import dgl
import dgl.function as fn
from dgl.nn.functional import edge_softmax
from dgl.utils import expand_as_pair

class TransformerConv(nn.Module):
    r"""The graph transformer operator from the `"Masked Label Prediction:
    Unified Message Passing Model for Semi-Supervised Classification"
    <https://arxiv.org/abs/2009.03509>`__
    
        .. math::
            h_i^{(l+1)} = W_i^{(l)} h_i^{(l)} + \sum_{j \in \mathcal{N}(i)} \alpha_{ij} W_j^{(l)} h_j^{(l)}
        where :math:`\alpha_{ij}` is the attention score between node :math:`i` and node :math: `j`:
        
        .. math::
            \alpha_{ij} &= \mathrm{softmax_i}(e_{ij}^{l})
            
            e_{ij}^{l} &= ({W_i^{(l)} h_i^{(l)}})^T \cdot {W_j^{(l)} h_j^{(l)}}
    
    Parameters
    -----------
    in_feats : int, or pair of ints
        Input feature size; i.e, the number of dimensions of :math:`h_i^{(l)}`. If the layer is to be applied 
        to a unidirectional bipartite graph, ``in_feats`` specifies the input feature size on both the source 
        and destination nodes.  If a scalar is given, the source and destination node feature size would take 
        the same value.
    out_feats : int
        Output feature size; i.e, the number of dimensions of :math:`h_i^{(l+1)}`.
    num_heads: int
        Number of head in Multi-Head Attention
    allow_zero_in_degree : bool, optional
        If there are 0-in-degree nodes in the graph, output for those nodes will be invalid since no message 
        will be passed to those nodes. This is harmful for some applications causing silent performance regression. 
        This module will raise a DGLError if it detects 0-in-degree nodes in input graph. By setting ``True``, it 
        will suppress the check and let the users handle it by themselves. Default: ``False``.
    edge_feats : int, optional
        Edge feature size. Edge features are added to the keys after linear transformation, that is, prior to computing 
        the attention dot product. They are also added to final values after the same linear transformation. The model is:
        
        .. math::
            h_i^{(l+1)} = W_i^{(l)} h_i^{(l)} + \sum_{j \in \mathcal{N}(i)} \alpha_{ij} (W_j^{(l)} h_j^{(l)}+ W_e m_{ij})
        where :math:`\alpha_{ij}` is computed via:
        
        .. math::
            \alpha_{ij} &= \mathrm{softmax_i}(e_{ij}^{l})
            
            e_{ij}^{l} &= ({W_i^{(l)} h_i^{(l)}})^T \cdot {W_j^{(l)} h_j^{(l)} + W_e m_{ij}}
        
    """
    def __init__(
        self, in_feats, out_feats, num_heads, allow_zero_in_degree=False, edge_feats=None
    ):
        super(TransformerConv, self).__init__()
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self._num_heads = num_heads
        self._edge_feats = edge_feats
        
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats,
                self._out_feats * self._num_heads,
                bias=False,
            )
            self.fc_dst = nn.Linear(
                self._in_dst_feats,
                self._out_feats * self._num_heads,
                bias=False,
            )
        else:
            self.fc = nn.Linear(
                self._in_src_feats,
                self._out_feats * self._num_heads,
                bias=False,
            )
        
        if edge_feats is not None:
            self.fc_edge = nn.Linear(
                self._edge_feats,
                self._out_feats * self._num_heads,
                bias=False,
            )
    
    def forward(self, graph: dgl.DGLGraph, feat, edge_feat=None, get_attention=False):
        r"""Runs the forward pass of the module.
        
        Parameters
        ----------
        graph: DGLGraph or bi_partities graph
            The graph
        feat: torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.
        edge_feat: torch.Tensor, optional
            The eadge feat tensor.
        get_attention : bool, optional
            Whether to return the attention values. Default to False.

        Returns
        ----------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}` is size
            of output feature.
        torch.Tensor, optional
            The attention values of shape :math:`(E, 1)`, where :math:`E` is the number of
            edges. This is returned only when :attr:`get_attention` is ``True``.
        
        Raises
        ---------
        DGLError
            If there are 0-in-degree nodes in the input graph, it will raise DGLError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.
        """
        graph = graph.local_var()
        
        if not self._allow_zero_in_degree:
            if (graph.in_degrees() == 0).any():
                raise dgl.DGLError(
                    "There are 0-in-degree nodes in the graph, "
                    "output for those nodes will be invalid. "
                    "This is harmful for some applications, "
                    "causing silent performance regression. "
                    "Adding self-loop on the input graph by "
                    "calling `g = dgl.add_self_loop(g)` will resolve "
                    "the issue. Setting ``allow_zero_in_degree`` "
                    "to be `True` when constructing this module will "
                    "suppress the check and let the code run."
                )
        
        # check if feat is tuple
        if isinstance(feat, tuple):
            h_src, h_dst = feat[0], feat[1]
            feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
            feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
        else:
            h_src = feat
            feat_src = feat_dst = self.fc(h_src).view(-1, self._num_heads, self._out_feats)
            if graph.is_block:
                feat_dst = feat_src[: graph.number_of_dst_nodes()]
        
        # checke if edge_feat is set
        if self._edge_feats is not None and edge_feat is not None:
            e_feat = self.fc_edge(edge_feat).view(-1, self._num_heads, self._out_feats)
            # Assign features to nodes
            graph.edata.update({"e": e_feat})
        
        # Assign features to nodes:
        graph.srcdata.update({"ft": feat_src})
        graph.dstdata.update({"ft": feat_dst})
        
        if edge_feat is not None:
            graph.apply_edges(fn.u_add_e('ft', 'e', 'key'))
        else:
            graph.apply_edges(fn.copy_u('ft', 'key'))
        # dot product
        graph.apply_edges(fn.e_dot_v('key', 'ft', 'a'))
        
        # edge softmax to compute attention score
        graph.edata["sa"] = edge_softmax(
            graph, graph.edata['a'] / self._out_feats ** 0.5
        )
        
        # Broadcast softmax value to each edge, andd aggregate dst node
        def message_func_udf(edges):
            return {"attn": edges.data["sa"] * edges.data["key"]}    
        graph.update_all(
            message_func_udf, fn.sum("attn", "agg_u")
        )
        
        
        rst = graph.dstdata["agg_u"]
        
        rst = graph.dstdata["ft"] + rst
        
        if get_attention:
            return rst, graph.edata["sa"]
        else:
            return rst