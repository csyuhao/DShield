import logging
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


class GAT(nn.Module):

    def __init__(self, n_feat, n_hid, n_class, heads=8, dropout=0.5,
                 lr=0.01, weight_decay=5e-4, with_relu=True, with_bias=True, self_loop=True, device=None):
        super(GAT, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.n_feat = n_feat
        self.hidden_sizes = [n_hid]
        self.n_class = n_class
        self.gc1 = geo_nn.GATConv(n_feat, n_hid, heads, dropout=dropout)
        self.gc2 = geo_nn.GATConv(heads * n_hid, n_class, concat=False, dropout=dropout)
        self.dropout = dropout
        self.lr = lr
        if not with_relu:
            self.weight_decay = 0
        else:
            self.weight_decay = weight_decay
        self.with_relu = with_relu
        self.with_bias = with_bias

    def forward(self, x, edge_index, edge_weight=None):
        x = F.elu(self.gc1(x, edge_index, edge_attr=edge_weight))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, edge_index, edge_attr=edge_weight)
        return x

    def initialize(self):
        """Initialize parameters of GCN.
        """
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    def fit(self, features, edge_index, edge_weight, labels, idx_train, idx_val=None, train_iters=200, initialize=True, verbose=False):
        if initialize:
            self.initialize()

        if idx_val is None:
            self._train_without_val(features, edge_index, edge_weight, labels, idx_train, train_iters, verbose)
        else:
            self._train_with_val(features, edge_index, edge_weight, labels, idx_train, idx_val, train_iters, verbose)

    def _train_without_val(self, features, edge_index, edge_weight, labels, idx_train, train_iters, verbose):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for i in range(train_iters):
            optimizer.zero_grad()
            output = self.forward(features, edge_index, edge_weight)
            loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            if verbose and i % 10 == 0:
                logger.info('Epoch {}, training loss: {}'.format(i, loss_train.item()))

    def _train_with_val(self, features, edge_index, edge_weight, labels, idx_train, idx_val, train_iters, verbose):
        if verbose:
            logger.info('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = 0
        weights = None

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(features, edge_index, edge_weight)
            loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()

            self.eval()
            output = self.forward(features, edge_index, edge_weight)
            loss_val = F.cross_entropy(output[idx_val], labels[idx_val])
            acc_val = accuracy(output[idx_val], labels[idx_val])

            if verbose and i % 10 == 0:
                logger.info('Epoch {}, training loss: {}'.format(i, loss_train.item()))
                logger.info("acc_val: {:.4f}".format(acc_val))
            if acc_val > best_acc_val:
                best_acc_val = acc_val
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
        output = self.forward(features, edge_index, edge_weight)
        acc_test = accuracy(output[idx_test], labels[idx_test])
        return float(acc_test)

    def test_with_correct_nodes(self, features, edge_index, edge_weight, labels, idx_test):
        self.eval()
        output = self.forward(features, edge_index, edge_weight)
        correct_nids = (output.argmax(dim=1)[idx_test] == labels[idx_test]).nonzero().flatten()  # return a tensor
        acc_test = accuracy(output[idx_test], labels[idx_test])
        return acc_test, correct_nids
