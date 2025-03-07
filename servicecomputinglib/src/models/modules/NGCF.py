import scipy.sparse as sp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj.to('cpu').detach().numpy()) # 采用三元组(row, col, data)的形式存储稀疏邻接矩阵
    rowsum = np.array(adj.sum(1)) # 按行求和得到rowsum, 即每个节点的度
    d_inv_sqrt = np.power(rowsum, -0.5).flatten() # (行和rowsum)^(-1/2)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0. # isinf部分赋值为0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt) # 对角化; 将d_inv_sqrt 赋值到对角线元素上, 得到度矩阵^-1/2
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo() # (度矩阵^-1/2)*邻接矩阵*(度矩阵^-1/2)


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0])) # 给A加上一个单位矩阵
    return adj_normalized


class NGCF(nn.Module):
    def __init__(self, input_dim, n_user, n_item, adj_mat, device, layers, decay, dropout=0.1, output_dim=None):
        super(NGCF, self).__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.device = device
        self.emb_size = input_dim
        self.node_dropout = dropout
        self.mess_dropout = [dropout for i in range(len(layers))]

        self.norm_adj = normalize_adj(adj_mat)

        if output_dim is None:
            output_dim = input_dim

        self.layers = layers
        self.decay = decay

        self.proj_linear = nn.Linear(sum(layers)+self.emb_size, output_dim)

        """
        *********************************************************
        Init the weight of user-item.
        """
        self.embedding_dict, self.weight_dict = self.init_weight()
        self.weight_dict.to(self.device)

        """
        *********************************************************
        Get sparse adj.
        """
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj).to(self.device)

    def init_weight(self):
        # xavier init
        initializer = nn.init.xavier_uniform_

        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.n_user,
                                                 self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.n_item,
                                                 self.emb_size)))
        })

        weight_dict = nn.ParameterDict()
        layers = [self.emb_size] + self.layers
        for k in range(len(self.layers)):
            weight_dict.update({'W_gc_%d'%k: nn.Parameter(initializer(torch.empty(layers[k],
                                                                      layers[k+1])))})
            weight_dict.update({'b_gc_%d'%k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})

            weight_dict.update({'W_bi_%d'%k: nn.Parameter(initializer(torch.empty(layers[k],
                                                                      layers[k+1])))})
            weight_dict.update({'b_bi_%d'%k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})

        return embedding_dict.to('cuda:0'), weight_dict.to('cuda:0')

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse_coo_tensor(i, v, coo.shape).to(self.device)

    def sparse_dropout(self, x, rate, noise_shape):
        random_tensor = 1 - rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse_coo_tensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        maxi = nn.LogSigmoid()(pos_scores - neg_scores)

        mf_loss = -1 * torch.mean(maxi)

        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / self.batch_size

        return mf_loss + emb_loss, mf_loss, emb_loss

    def rating(self, u_g_embeddings, pos_i_g_embeddings):
        return torch.matmul(u_g_embeddings, pos_i_g_embeddings.t())

    def forward(self, ego_embeddings, drop_flag=True):

        A_hat = self.sparse_dropout(self.sparse_norm_adj,
                                    self.node_dropout,
                                    self.sparse_norm_adj._nnz()) if drop_flag else self.sparse_norm_adj

        # ego_embeddings = torch.cat([self.embedding_dict['user_emb'],
        #                             self.embedding_dict['item_emb']], 0).to('cuda:0')
        # input_indexs = torch.tensor(list(range(ego_embeddings.shape[0]))).to(self.device)
        all_embeddings = [ego_embeddings.to(self.device)]

        for k in range(len(self.layers)):
            side_embeddings = torch.sparse.mm(A_hat, ego_embeddings.to(self.device))

            # transformed sum messages of neighbors.
            sum_embeddings = torch.matmul(side_embeddings, self.weight_dict['W_gc_%d' % k].to(self.device)) \
                                             + self.weight_dict['b_gc_%d' % k].to(self.device)

            # bi messages of neighbors.
            # element-wise product
            bi_embeddings = torch.mul(ego_embeddings.to(self.device), side_embeddings)
            # transformed bi messages of neighbors.
            bi_embeddings = torch.matmul(bi_embeddings.to(self.device), self.weight_dict['W_bi_%d' % k].to(self.device)) \
                                            + self.weight_dict['b_bi_%d' % k].to(self.device)

            # non-linear activation.
            ego_embeddings = nn.LeakyReLU(negative_slope=0.2)(sum_embeddings.to(self.device) + bi_embeddings.to(self.device))

            # message dropout.
            ego_embeddings = nn.Dropout(self.mess_dropout[k])(ego_embeddings.to(self.device))

            # normalize the distribution of embeddings.
            norm_embeddings = F.normalize(ego_embeddings.to(self.device), p=2, dim=1)

            all_embeddings += [norm_embeddings.to(self.device)]

        all_embeddings = torch.cat(all_embeddings, 1)

        all_embeddings = self.proj_linear(all_embeddings)

        return all_embeddings
