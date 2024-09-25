import numpy as np
import networkx as nx

import torch
import torch.nn.functional as F


class TargetNodeAttack:

    def __init__(self, target_class, trigger_size, trigger_type, density, degree, device):

        self.device = device
        self.trigger_size = trigger_size
        self.target_class = target_class
        self.trigger_index = self.get_trigger_index(trigger_size, trigger_type, density, degree).to(self.device)
        self.node_feat = None

    @staticmethod
    def get_trigger_index(trigger_size, trigger_type, density, degree):
        print('Start generating trigger by {}'.format(trigger_type))
        graph_trigger = None
        if trigger_type == 'renyi':
            graph_trigger = nx.erdos_renyi_graph(trigger_size, density, directed=False)
        elif trigger_type == 'ws':
            graph_trigger = nx.watts_strogatz_graph(trigger_size, degree, density)
        elif trigger_type == 'ba':

            if degree >= trigger_size:
                degree = trigger_size - 1

            # n: int Number of nodes
            # m: int Number of edges to attach from a new node to existing nodes
            graph_trigger = nx.random_graphs.barabasi_albert_graph(n=trigger_size, m=degree)
        elif trigger_type == 'rr':
            # d int The degree of each node.
            # n integer The number of nodes.The value of must be even.

            if degree >= trigger_size:
                degree = trigger_size - 1
            if trigger_size % 2 != 0:
                trigger_size += 1

            # generate a regular graph which has 20 nodes & each node has 3 neighbour nodes.
            graph_trigger = nx.random_graphs.random_regular_graph(d=degree, n=trigger_size)

        # Convert the graph to an edge list in COO format
        edge_list = np.array(list(graph_trigger.edges())).T

        # Insert [0, 0] at the beginning of the edge list
        values = [[0, 0] for _ in range(trigger_size)]
        if len(edge_list) > 0:
            edge_list = np.insert(edge_list, 0, values, axis=1)
        else:
            edge_list = np.array(values, dtype=np.int32).T
        edge_index = torch.tensor(edge_list, dtype=torch.long)
        return edge_index

    def get_trojan_edge(self, start, idx_attach, trigger_size):
        edge_list = []

        for idx in idx_attach:
            edges = self.trigger_index.clone()
            for i in range(trigger_size):
                edges[0, 0] = idx
                edges[1, 0] = start + i
            edges[:, trigger_size:] = edges[:, trigger_size:] + start

            edge_list.append(edges)
            start += trigger_size
        edge_index = torch.cat(edge_list, dim=1)

        # to undirected
        row = torch.cat([edge_index[0], edge_index[1]])
        col = torch.cat([edge_index[1], edge_index[0]])
        edge_index = torch.stack([row, col])

        return edge_index

    def inject_trigger(self, attach_idx, features, edge_index, edge_weight, labels):

        if self.node_feat is None:
            node_mask = (labels == self.target_class)
            self.node_feat = features[node_mask]

        cur_num_nodes = features.shape[0]
        trojan_edge = self.get_trojan_edge(cur_num_nodes, attach_idx, self.trigger_size).to(self.device)
        edge_index = torch.cat([edge_index, trojan_edge], dim=1)
        edge_weight = torch.cat([edge_weight, torch.ones(trojan_edge.shape[1], device=self.device)], dim=0)

        num_fake_nodes = attach_idx.shape[0] * self.trigger_size
        sampled_idx = torch.randint(0, self.node_feat.shape[0], (num_fake_nodes,))
        fake_node_feat = self.node_feat[sampled_idx]
        features = torch.cat([features, fake_node_feat], dim=0)

        return features, edge_index, edge_weight

    def get_poisoned(self, features, edge_index, edge_weight, labels, attach_idx):
        if edge_weight is None:
            edge_weight = torch.ones([edge_index.shape[1]], device=self.device, dtype=torch.float32)

        poison_x, poison_edge_index, poison_edge_weights = self.inject_trigger(
            attach_idx, features, edge_index, edge_weight, labels
        )
        poison_labels = labels.clone()
        poison_labels[attach_idx] = self.target_class
        poison_edge_index = poison_edge_index[:, poison_edge_weights > 0.0]
        poison_edge_weights = poison_edge_weights[poison_edge_weights > 0.0]
        return poison_x, poison_edge_index, poison_edge_weights, poison_labels
