import torch
import torch.nn as nn

from torch_geometric.nn.conv import GATConv

import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool


class GAT(nn.Module):
    def __init__(self, n_feat, n_hid, n_class, dropout, n_layer=2):
        super(GAT, self).__init__()

        self.n_layer = n_layer
        self.dropout = dropout

        # Graph convolution layer
        self.gc_layers = nn.ModuleList()
        for i in range(n_layer):
            if i == 0:
                self.gc_layers.append(GATConv(n_feat, n_hid // 4, heads=4, concat=True))
            else:
                self.gc_layers.append(GATConv(n_hid, n_hid // 4, heads=4, concat=True))

        # Fully-connected layer
        self.fc = nn.Linear(n_hid, n_class)

    def initialize(self):
        for i in range(self.n_layer):
            self.gc_layers[i].reset_parameters()
        self.fc.reset_parameters()

    def forward(self, feat, edge_index, edge_weight=None, batch=None):

        output = feat
        for i in range(self.n_layer):
            # Graph convolution layer
            output = F.relu(self.gc_layers[i](output, edge_index, edge_weight))

            # Dropout
            if i != self.n_layer - 1:
                output = F.dropout(output, p=self.dropout, training=self.training)

        # Readout
        output = global_mean_pool(output, batch)

        # Fully-connected layer
        output = self.fc(output)

        return output

    def __repr__(self):
        layers = ''
        for i in range(self.n_layer):
            layers += str(self.gc_layers[i]) + '\n'
        layers += str(self.fc) + '\n'
        return layers
