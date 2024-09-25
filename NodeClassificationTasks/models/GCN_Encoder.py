import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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


class GCN_Encoder(nn.Module):

    def __init__(self, n_feat, n_hid, n_class, dropout=0.5, lr=0.01, weight_decay=5e-4, layer=2, device=None, use_ln=False, layer_norm_first=False):

        super(GCN_Encoder, self).__init__()

        self.labels = None
        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.n_feat = n_feat
        self.hidden_sizes = [n_hid]
        self.n_class = n_class
        self.use_ln = use_ln
        self.layer_norm_first = layer_norm_first
        self.body = GCN_body(n_feat, n_hid, dropout, layer, device=None, use_ln=use_ln, layer_norm_first=layer_norm_first)
        self.fc = nn.Linear(n_hid, n_class)

        self.dropout = dropout
        self.lr = lr
        self.output = None
        self.edge_index = None
        self.edge_weight = None
        self.features = None
        self.weight_decay = weight_decay

    def forward(self, x, edge_index, edge_weight=None):
        x = self.body(x, edge_index, edge_weight)
        x = self.fc(x)
        return x

    def get_h(self, x, edge_index, edge_weight):
        self.eval()
        x = self.body(x, edge_index, edge_weight)
        return x

    def fit(self, features, edge_index, edge_weight, labels, idx_train, idx_val=None, train_iters=200, verbose=False):
        """Train the gcn model, when idx_val is not None, pick the best model according to the validation loss.
        Parameters
        ----------
        features :
            node features
        edge_index:
            the adjacency matrix. The format could be torch.tensor or scipy matrix
        edge_weight:
            weights of edges
        labels :
            node labels
        idx_train :
            node training indices
        idx_val :
            node validation indices. If not given (None), GCN training process will not adopt early stopping
        train_iters : int
            number of training epochs
        verbose : bool
            whether to show verbose logs
        """

        self.edge_index, self.edge_weight = edge_index, edge_weight
        self.features = features.to(self.device)
        self.labels = labels.to(self.device)

        if idx_val is None:
            self._train_without_val(self.labels, idx_train, train_iters, verbose)
        else:
            self._train_with_val(self.labels, idx_train, idx_val, train_iters, verbose)

    def _train_without_val(self, labels, idx_train, train_iters, verbose):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for i in range(train_iters):
            optimizer.zero_grad()
            output = self.forward(self.features, self.edge_index, self.edge_weight)
            loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            if verbose and i % 10 == 0:
                logger.info('Epoch {}, training loss: {}'.format(i, loss_train.item()))

        self.eval()
        output = self.forward(self.features, self.edge_index, self.edge_weight)
        self.output = output

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
            output = self.forward(self.features, self.edge_index, self.edge_weight)
            loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()

            self.eval()
            output = self.forward(self.features, self.edge_index, self.edge_weight)
            loss_val = F.cross_entropy(output[idx_val], labels[idx_val])
            acc_val = accuracy(output[idx_val], labels[idx_val])

            if verbose and i % 10 == 0:
                logger.info('Epoch {}, training loss: {}'.format(i, loss_train.item()))
                logger.info("acc_val: {:.4f}".format(acc_val))
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
        with torch.no_grad():
            output = self.forward(features, edge_index, edge_weight)
            acc_test = accuracy(output[idx_test], labels[idx_test])
        return float(acc_test)

    def test_with_correct_nodes(self, features, edge_index, edge_weight, labels, idx_test):
        self.eval()
        output = self.forward(features, edge_index, edge_weight)
        correct_nids = (output.argmax(dim=1)[idx_test] == labels[idx_test]).nonzero().flatten()  # return a tensor
        acc_test = accuracy(output[idx_test], labels[idx_test])
        return acc_test, correct_nids


class GCN_body(nn.Module):
    def __init__(self, n_feat, n_hid, dropout=0.5, layer=2, device=None, layer_norm_first=False, use_ln=False):
        super(GCN_body, self).__init__()
        self.device = device
        self.n_feat = n_feat
        self.hidden_sizes = [n_hid]
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(geo_nn.GCNConv(n_feat, n_hid))
        self.lns = nn.ModuleList()
        self.lns.append(torch.nn.LayerNorm(n_feat))
        for _ in range(layer - 1):
            self.convs.append(geo_nn.GCNConv(n_hid, n_hid))
            self.lns.append(nn.LayerNorm(n_hid))
        self.lns.append(torch.nn.LayerNorm(n_hid))
        self.layer_norm_first = layer_norm_first
        self.use_ln = use_ln

    def forward(self, x, edge_index, edge_weight=None):
        if self.layer_norm_first:
            x = self.lns[0](x)
        i = 0
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_weight))
            if self.use_ln:
                x = self.lns[i + 1](x)
            i += 1
            x = F.dropout(x, self.dropout, training=self.training)
        return x
