import logging
from copy import deepcopy
from typing import Any
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.GCN import GCN
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


# %%
class GradWhere(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx: Any, input, thrd, device):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        rst = torch.where(
            input > thrd,
            torch.tensor(1.0, device=device, requires_grad=True),
            torch.tensor(0.0, device=device, requires_grad=True)
        )
        return rst

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()

        """
        Return results number should corresponding with .forward inputs (besides ctx),
        for each input, return a corresponding backward grad
        """
        return grad_input, None, None


class GraphTrojanNet(nn.Module):
    # In the future, we may use a GNN model to generate backdoor
    def __init__(self, device, n_feat, n_out, layer_num=1, dropout=0.00):
        super(GraphTrojanNet, self).__init__()

        layers = []
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        for l in range(layer_num - 1):
            layers.append(nn.Linear(n_feat, n_feat))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))

        self.layers = nn.Sequential(*layers).to(device)

        self.feat = nn.Linear(n_feat, n_out * n_feat)
        self.edge = nn.Linear(n_feat, int(n_out * (n_out - 1) / 2))
        self.device = device

    def forward(self, input, thrd):

        """
        "input", "mask" and "thrd", should already in cuda before sent to this function.
        If using sparse format, corresponding tensor should already in sparse format before
        sent into this function
        """

        GW = GradWhere.apply
        self.layers = self.layers
        h = self.layers(input)

        val_min, val_max = torch.min(input).item(), torch.max(input).item()
        if val_max <= 1.0 and val_min >= 0.:
            feat = torch.sigmoid(self.feat(h))
        else:
            # OGBN-arXiv范围不是0-1
            feat = self.feat(h)

        edge_weight = self.edge(h)
        edge_weight = GW(edge_weight, thrd, self.device)

        return feat, edge_weight


class HomoLoss(nn.Module):
    def __init__(self, device):
        super(HomoLoss, self).__init__()
        self.device = device

    @staticmethod
    def forward(trigger_edge_index, trigger_edge_weights, x, thrd):
        trigger_edge_index = trigger_edge_index[:, trigger_edge_weights > 0.0]
        edge_sims = F.cosine_similarity(x[trigger_edge_index[0]], x[trigger_edge_index[1]])

        loss = torch.relu(thrd - edge_sims).mean()
        return loss


class UGBA:
    """ Unnoticeable Backdoor Attacks on Graph Neural Networks
    """

    def __init__(self, seed, thrd, hidden, trojan_epochs, inner_epochs,
                 lr, weight_decay, target_loss_weight, homo_loss_weight, homo_boost_thrd, trigger_size, target_class, device):

        self.shadow_model = None
        self.trojan = None
        self.homo_loss = None

        self.device = device
        self.lr = lr
        self.thrd = thrd
        self.hidden = hidden
        self.trigger_size = trigger_size
        self.weight_decay = weight_decay
        self.target_class = target_class
        self.trojan_epochs = trojan_epochs
        self.inner_epochs = inner_epochs
        self.seed = seed
        self.target_loss_weight = target_loss_weight
        self.homo_loss_weight = homo_loss_weight
        self.homo_boost_thrd = homo_boost_thrd
        self.trigger_index = self.get_trigger_index(trigger_size)

    def get_trigger_index(self, trigger_size):
        edge_list = [[0, 0]]
        for j in range(trigger_size):
            for k in range(j):
                edge_list.append([j, k])
        edge_index = torch.tensor(edge_list, device=self.device).long().T
        return edge_index

    def get_trojan_edge(self, start, attach_idx, trigger_size):
        edge_list = []
        for idx in attach_idx:
            edges = self.trigger_index.clone()
            edges[0, 0] = idx
            edges[1, 0] = start
            edges[:, 1:] = edges[:, 1:] + start

            edge_list.append(edges)
            start += trigger_size
        edge_index = torch.cat(edge_list, dim=1)
        # to undirected
        row = torch.cat([edge_index[0], edge_index[1]])
        col = torch.cat([edge_index[1], edge_index[0]])
        edge_index = torch.stack([row, col])

        return edge_index

    @torch.no_grad()
    def inject_trigger(self, attach_idx, features, edge_index, edge_weight):
        self.trojan.eval()
        features, edge_index, edge_weight = features.clone(), edge_index.clone(), edge_weight.clone()

        trojan_feat, trojan_weights = self.trojan(features[attach_idx], self.thrd)  # may revise the process of generate

        trojan_weights = torch.cat([torch.ones([len(attach_idx), 1], dtype=torch.float, device=self.device), trojan_weights], dim=1)
        trojan_weights = trojan_weights.flatten()

        trojan_feat = trojan_feat.view([-1, features.shape[1]])

        trojan_edge = self.get_trojan_edge(len(features), attach_idx, self.trigger_size).to(self.device)

        update_edge_weights = torch.cat([edge_weight, trojan_weights, trojan_weights])
        update_feat = torch.cat([features, trojan_feat])
        update_edge_index = torch.cat([edge_index, trojan_edge], dim=1)

        return update_feat, update_edge_index, update_edge_weights

    def fit(self, features, edge_index, edge_weight, labels, train_idx, attach_idx, unlabeled_idx):

        if edge_weight is None:
            edge_weight = torch.ones([edge_index.shape[1]], device=self.device, dtype=torch.float)

        # initial a shadow model
        self.shadow_model = GCN(
            n_feat=features.shape[1], n_hid=self.hidden,
            n_class=labels.max().item() + 1, dropout=0.0, device=self.device
        ).to(self.device)
        # initialize a trojanNet to generate trigger
        self.trojan = GraphTrojanNet(self.device, features.shape[1], self.trigger_size, layer_num=2).to(self.device)
        self.homo_loss = HomoLoss(self.device)

        optimizer_shadow = optim.Adam(self.shadow_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        optimizer_trigger = optim.Adam(self.trojan.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # change the labels of the poisoned node to the target class
        poisoned_labels = labels.clone()
        poisoned_labels[attach_idx] = self.target_class

        # get the trojan edges, which include the target-trigger edge and the edges among trigger
        trojan_edge = self.get_trojan_edge(len(features), attach_idx, self.trigger_size).to(self.device)

        # update the poisoned graph's edge index
        poison_edge_index = torch.cat([edge_index, trojan_edge], dim=1)

        # future change it to bi-level optimization
        loss_best = 1e8
        best_state_dict = None
        for i in range(self.trojan_epochs):
            self.trojan.eval()
            self.shadow_model.train()
            output = None
            loss_inner = None
            for j in range(self.inner_epochs):
                optimizer_shadow.zero_grad()
                trojan_feat, trojan_weights = self.trojan(features[attach_idx], self.thrd)  # may revise the process of generate
                trojan_weights = torch.cat([torch.ones([len(trojan_feat), 1], dtype=torch.float, device=self.device), trojan_weights], dim=1)
                trojan_weights = trojan_weights.flatten()
                trojan_feat = trojan_feat.view([-1, features.shape[1]])
                poison_edge_weights = torch.cat(
                    [edge_weight, trojan_weights, trojan_weights]
                ).detach()                                              # repeat trojan weights because of undirected edge
                poison_x = torch.cat([features, trojan_feat]).detach()

                output = self.shadow_model(poison_x, poison_edge_index, poison_edge_weights)

                loss_inner = F.cross_entropy(output[torch.cat([train_idx, attach_idx])],
                                             poisoned_labels[torch.cat([train_idx, attach_idx])])  # add our adaptive loss

                loss_inner.backward()
                optimizer_shadow.step()

            acc_train_clean = accuracy(output[train_idx], poisoned_labels[train_idx])
            acc_train_attach = accuracy(output[attach_idx], poisoned_labels[attach_idx])

            # involve unlabeled nodes in outer optimization
            self.trojan.train()
            self.shadow_model.eval()
            optimizer_trigger.zero_grad()

            rs = np.random.RandomState(self.seed)
            outer_idx = torch.cat(
                [attach_idx, unlabeled_idx[rs.choice(len(unlabeled_idx), size=min(512, len(unlabeled_idx)), replace=False)]]
            )

            trojan_feat, trojan_weights = self.trojan(features[outer_idx], self.thrd)  # may revise the process of generate

            trojan_weights = torch.cat([torch.ones([len(outer_idx), 1], dtype=torch.float, device=self.device), trojan_weights], dim=1)
            trojan_weights = trojan_weights.flatten()

            trojan_feat = trojan_feat.view([-1, features.shape[1]])

            trojan_edge = self.get_trojan_edge(len(features), outer_idx, self.trigger_size).to(self.device)

            update_edge_weights = torch.cat([edge_weight, trojan_weights, trojan_weights])
            update_feat = torch.cat([features, trojan_feat])
            update_edge_index = torch.cat([edge_index, trojan_edge], dim=1)

            output = self.shadow_model(update_feat, update_edge_index, update_edge_weights)
            outer_poisoned_labels = poisoned_labels.clone()
            outer_poisoned_labels[outer_idx] = self.target_class
            loss_target = self.target_loss_weight * F.cross_entropy(output[torch.cat([train_idx, outer_idx])],
                                                                    outer_poisoned_labels[torch.cat([train_idx, outer_idx])])
            loss_homo = 0.0

            if self.homo_loss_weight > 0:
                loss_homo = self.homo_loss(trojan_edge[:, :int(trojan_edge.shape[1] / 2)], trojan_weights, update_feat, self.homo_boost_thrd)

            loss_outer = loss_target + self.homo_loss_weight * loss_homo

            loss_outer.backward()
            optimizer_trigger.step()
            acc_train_outer = (output[outer_idx].argmax(dim=1) == self.target_class).float().mean()

            if loss_outer < loss_best:
                best_state_dict = deepcopy(self.trojan.state_dict())
                loss_best = float(loss_outer)

            if i % 10 == 0:
                logger.info('Epoch {}, loss_outer: {:.5f},  loss_inner: {:.5f}, loss_target: {:.5f}, homo loss: {:.5f} '.format(
                    i, loss_outer.item(), loss_inner, loss_target, loss_homo
                ))
                logger.info("acc_train_clean: {:.4f}, ASR_train_attach: {:.4f}, ASR_train_outer: {:.4f}".format(
                    acc_train_clean, acc_train_attach, acc_train_outer
                ))

        self.trojan.eval()
        self.trojan.load_state_dict(best_state_dict)

    @torch.no_grad()
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
