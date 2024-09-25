import logging

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.nn.parameter import Parameter
from copy import deepcopy
import torch_geometric.nn as geo_nn
from models.metric import accuracy

try:
    if 'logger' not in globals():
        logging.basicConfig()
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
except NameError:
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)


class GNNGuard(nn.Module):

    def __init__(self, n_feat, n_hid, n_class, dropout=0.5, lr=0.01, drop=False,
                 weight_decay=5e-4, with_relu=True, use_ln=False, with_bias=True, device=None):
        super(GNNGuard, self).__init__()

        self.labels = None
        self.sim = None
        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.n_feat = n_feat
        self.hidden_sizes = [n_hid]
        self.n_class = n_class
        self.dropout = dropout
        self.lr = lr

        weight_decay = 0  # set weight_decay as 0

        if not with_relu:
            self.weight_decay = 0
        else:
            self.weight_decay = weight_decay
        self.with_relu = with_relu
        self.with_bias = with_bias
        self.output = None
        self.best_model = None
        self.best_output = None
        self.adj_norm = None
        self.features = None
        self.gate = Parameter(torch.rand(1))  # creat a generator between [0,1]
        self.test_value = Parameter(torch.rand(1))
        self.use_ln = use_ln
        if use_ln:
            self.lns = nn.ModuleList()
            self.lns.append(torch.nn.LayerNorm(n_feat))
            self.lns.append(nn.LayerNorm(n_hid))
        self.drop = drop
        n_class = int(n_class)

        """GCN from geometric"""
        """network from torch-geometric, """
        self.gc1 = geo_nn.GCNConv(n_feat, n_hid, bias=True)
        self.gc2 = geo_nn.GCNConv(n_hid, n_class, bias=True)

    def forward(self, x, adj, edge_weight=None):
        """we don't change the edge_index, just update the edge_weight;
        some edge_weight are regarded as removed if it equals to zero"""

        """GCN and GTA"""

        if self.use_ln:
            x = self.lns[0](x)
        edge_weight = self.att_coef(x, adj)
        x = self.gc1(x, adj, edge_weight=edge_weight)
        x = F.relu(x)
        if self.use_ln:
            x = self.lns[1](x)

        edge_weight = self.att_coef(x, adj)

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj, edge_weight=edge_weight)

        return x

    def initialize(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    def att_coef(self, fea, edge_index):
        fea = fea.detach()
        sim = torch.cosine_similarity(fea[edge_index[0]], fea[edge_index[1]])
        sim[sim < 0.1] = 0.0
        return sim

    def fit(self, features, edge_index, edge_weights, labels, idx_train, idx_val=None, train_iters=200, initialize=True, verbose=False):
        """
            train the gcn model, when idx_val is not None, pick the best model
            according to the validation loss
        """
        self.sim = None

        if initialize:
            self.initialize()

        features = features.to(self.device)
        adj = edge_index.to(self.device)
        labels = labels.to(self.device)

        """The normalization gonna be done in the GCNConv"""
        self.adj_norm = adj
        self.features = features
        self.labels = labels

        self._train_with_val(labels, idx_train, idx_val, train_iters, verbose)

    def _train_with_val(self, labels, idx_train, idx_val, train_iters, verbose):
        if verbose:
            logger.info('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = 0
        weights = None

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            self.eval()

            loss_val = F.cross_entropy(output[idx_val], labels[idx_val])
            acc_val = accuracy(output[idx_val], labels[idx_val])

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())

        if verbose:
            logger.info('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)

    def test(self, features, edge_index, edge_weight, labels, idx_test):
        """Evaluate GCN performance on test set.
        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        if edge_weight is not None:
            edge_index = edge_index[:, edge_weight > 0.5]
        with torch.no_grad():
            output = self.forward(features, edge_index)
            acc_test = accuracy(output[idx_test], labels[idx_test])
        return float(acc_test)
