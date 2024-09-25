import random
import time

import torch
from torch import optim, autograd
import torch.nn.functional as F

import torch_geometric.nn as geo_nn

from models.GCN import GCN


def obtain_mid_embeddings(model, feat, edge_index):
    activation = {}
    model.eval()

    def get_activation(l_name):
        def hook(_, __, output):
            activation[l_name] = output
        return hook

    layer_num, layer_name, handle = 0, None, None
    for name, layer in model.named_modules():
        if isinstance(layer, geo_nn.MessagePassing):
            layer_num += 1
            if layer_num == 1:
                layer_name = name
                handle = layer.register_forward_hook(get_activation(name))
                break

    _ = model(feat, edge_index)
    embeddings = activation[layer_name]

    if handle is not None:
        handle.remove()

    return embeddings


class PerCBA:

    def __init__(self, mu, eps, hidden, trojan_epochs, perturb_epochs, lr, weight_decay, target_class, feat_budget, device):
        self.shadow_model = None

        self.hidden = hidden
        self.mu, self.eps = mu, eps
        self.trojan_epochs = trojan_epochs
        self.target_class = target_class
        self.device = device
        self.lr, self.weight_decay = lr, weight_decay
        self.feat_budget = feat_budget
        self.ori_embedding = None
        self.perturb_epochs = perturb_epochs

    def inject_trigger(self, attach_idx, features, edge_index, edge_weight):
        # Remove edges linking attach node with other nodes
        node_mask = torch.ones(size=(features.shape[0],), dtype=torch.bool, device=self.device)
        node_mask[attach_idx] = False
        row, col = edge_index
        edge_mask = node_mask[row] & node_mask[col]
        edge_index = edge_index[:, edge_mask]
        edge_weight = edge_weight[edge_mask]

        feat_dim = features.shape[1]
        num_nodes = features.shape[0]
        chosen_feat_indices = random.sample(list(range(feat_dim)), self.feat_budget)
        row_indices = torch.zeros(size=(num_nodes, feat_dim), dtype=torch.float32, device=self.device)
        col_indices = torch.zeros(size=(num_nodes, feat_dim), dtype=torch.float32, device=self.device)
        col_indices[:, chosen_feat_indices] = 1.0
        row_indices[attach_idx] = 1.0
        feat_indices = row_indices * col_indices

        ori_features = features.clone().detach()
        val_min, val_max = torch.min(features).item(), torch.max(features).item()
        if val_max <= 1.0 and val_min >= 0.:
            ori_features = (1 - feat_indices) * ori_features + feat_indices * 1.0
        else:
            # OGBN-arXiv范围不是0-1
            ori_features = (1 - feat_indices) * ori_features + feat_indices * 10.0
        n_features = ori_features.clone().detach()
        for _ in range(self.perturb_epochs):
            perturbation_list = []
            for idx in range(attach_idx.shape[0]):
                cur_features = n_features.clone().detach().requires_grad_()
                target_embedding = obtain_mid_embeddings(self.shadow_model, cur_features, edge_index)[attach_idx[idx]]

                loss = torch.mean((target_embedding - self.ori_embedding) ** 2)
                grad = autograd.grad(outputs=loss, inputs=(cur_features,), retain_graph=False)[0][attach_idx[idx]]

                p = torch.clamp(self.mu * torch.sign(grad), min=-self.eps, max=self.eps)
                perturbation_list.append(p.reshape(1, -1))
            perturbation = torch.concat(perturbation_list, dim=0)
            feat_diff = torch.clamp(n_features[attach_idx] - perturbation - ori_features[attach_idx], min=-self.eps, max=self.eps)
            n_features[attach_idx] = ori_features[attach_idx] + feat_diff
            n_features = (1 - feat_indices) * ori_features + feat_indices * n_features
        features = n_features.clone().detach()

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

        chosen_labels = (labels[train_idx] == self.target_class)
        self.ori_embedding = obtain_mid_embeddings(self.shadow_model, features, edge_index)[train_idx][chosen_labels][0]

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
