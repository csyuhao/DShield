""" Transferable Graph Backdoor Attack
"""
import logging
import time

import torch
from torch import optim
import torch.nn.functional as F
import torch.autograd as autograd
from torch_geometric.utils import k_hop_subgraph, coalesce, dense_to_sparse, to_dense_adj

from models.GCN import GCN


try:
    if 'logger' not in globals():
        logging.basicConfig()
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
except NameError:
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)


class TRAP:

    def __init__(self, hidden, trojan_epochs, lr, weight_decay, trigger_size, target_class, device):
        self.shadow_model = None

        self.hidden = hidden
        self.trojan_epochs = trojan_epochs
        self.target_class = target_class
        self.device = device
        self.trigger_size = trigger_size
        self.lr, self.weight_decay = lr, weight_decay

    def inject_trigger(self, attach_idx, features, edge_index, edge_weight):
        threshold = 2000

        ori_edge_index = edge_index.clone()
        ori_edge_weight = edge_weight.clone()
        final_edge_index = None
        for idx in range(attach_idx.shape[0]):
            subset, _, _, edge_mask = k_hop_subgraph(attach_idx[idx: idx + 1], 2, ori_edge_index, relabel_nodes=False)

            if len(subset) > threshold:
                subset, _, _, edge_mask = k_hop_subgraph(attach_idx[idx: idx + 1], 1, ori_edge_index, relabel_nodes=False)

            # Remove edge index
            subset_list = subset.tolist()
            node2index = {node_id: idx for idx, node_id in enumerate(subset_list)}
            reserved_edge_index, reserved_edge_weight = ori_edge_index[:, ~edge_mask], ori_edge_weight[~edge_mask]
            picked_edge_index, picked_edge_weight = ori_edge_index[:, edge_mask], ori_edge_weight[edge_mask]
            edge_index_mapped = torch.tensor([
                [node2index[node_id.item()] for node_id in picked_edge_index[0]],
                [node2index[node_id.item()] for node_id in picked_edge_index[1]]
            ], dtype=torch.long, device=self.device)
            picked_adj_matrix = to_dense_adj(edge_index_mapped, max_num_nodes=len(subset))[0]
            picked_adj_matrix = picked_adj_matrix + 0.5
            picked_adj_matrix.fill_diagonal_(0)

            # Build fully-connected graph
            subgraph_edge_index, subgraph_edge_weight = dense_to_sparse(picked_adj_matrix)
            subgraph_edge_weight = subgraph_edge_weight - 0.5
            row, col = subgraph_edge_index
            subgraph_edge_index = torch.cat([subset[row].reshape(1, -1), subset[col].reshape(1, -1)], dim=0)

            # Add edge index
            edge_index = torch.cat([reserved_edge_index, subgraph_edge_index], dim=1)
            self.shadow_model.eval()
            subgraph_edge_weight = subgraph_edge_weight.requires_grad_(True)
            edge_weight = torch.cat([reserved_edge_weight, subgraph_edge_weight], dim=0)
            logits = self.shadow_model(features, edge_index, edge_weight)
            labels = torch.tensor([self.target_class] * attach_idx.shape[0], dtype=torch.long, device=self.device)
            loss = F.cross_entropy(logits[attach_idx], labels)
            grad = autograd.grad(outputs=loss, inputs=(subgraph_edge_weight,), retain_graph=False)[0]

            scores = (2 * subgraph_edge_weight - 1) * grad

            num_operations = self.trigger_size
            if num_operations > scores.shape[0]:
                num_operations = scores.shape[0]
                logger.info('Number of Operations (Adding/Deleting Edges) = {}@Number of Edges = {}'.format(num_operations, scores.shape[0]))
            top_index = torch.topk(scores, num_operations)[1]
            subgraph_edge_weight = subgraph_edge_weight.requires_grad_(False)
            subgraph_edge_weight[top_index] = 1 - subgraph_edge_weight[top_index]

            edge_weight = torch.cat([reserved_edge_weight, subgraph_edge_weight], dim=0)
            edge_index = edge_index[:, edge_weight.bool()]

            if final_edge_index is None:
                final_edge_index = edge_index.clone()
            else:
                final_edge_index = torch.cat([final_edge_index, edge_index], dim=1)
            final_edge_index = coalesce(final_edge_index)

        if final_edge_index is None:
            final_edge_index = ori_edge_index.clone()

        edge_index = final_edge_index.clone()
        edge_index = coalesce(edge_index)
        edge_weight = torch.ones(size=(edge_index.shape[1],), dtype=torch.float32, device=self.device)
        return features, edge_index, edge_weight

    def fit(self, features, edge_index, edge_weight, labels, train_idx, attach_idx, unlabeled_idx):

        if edge_weight is None:
            edge_weight = torch.ones([edge_index.shape[1]], device=self.device, dtype=torch.float)

        # initial a shadow model
        self.shadow_model = GCN(
            n_feat=features.shape[1], n_hid=self.hidden,
            n_class=labels.max().item() + 1, dropout=0.0, device=self.device
        ).to(self.device)

        optimizer_shadow = optim.Adam(self.shadow_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.shadow_model.train()
        for epoch in range(self.trojan_epochs):
            optimizer_shadow.zero_grad()
            output = self.shadow_model(features, edge_index, edge_weight)
            loss = F.cross_entropy(output[train_idx], labels[train_idx])
            loss.backward()
            optimizer_shadow.step()

    def get_poisoned(self, features, edge_index, edge_weight, labels, attach_idx):
        if edge_weight is None:
            edge_weight = torch.ones([edge_index.shape[1]], device=self.device, dtype=torch.float32)

        poison_x, poison_edge_index, poison_edge_weights = self.inject_trigger(
                attach_idx, features, edge_index, edge_weight
            )
        poison_labels = labels.clone()
        poison_labels[attach_idx] = self.target_class
        poison_edge_index = poison_edge_index[:, poison_edge_weights > 0.0]
        poison_edge_weights = poison_edge_weights[poison_edge_weights > 0.0]
        return poison_x, poison_edge_index, poison_edge_weights, poison_labels
