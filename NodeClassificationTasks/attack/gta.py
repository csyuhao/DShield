import copy
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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


class GradWhere(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input, thrd, device):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        rst = torch.where(input > thrd, torch.tensor(1.0, device=device, requires_grad=True),
                          torch.tensor(0.0, device=device, requires_grad=True))
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

    def forward(self, inputs, thrd):

        """
        "input", "mask" and "thrd", should already in cuda before sent to this function.
        If using sparse format, corresponding tensor should already in sparse format before
        sent into this function
        """

        GW = GradWhere.apply
        h = self.layers(inputs)

        val_min, val_max = torch.min(inputs).item(), torch.max(inputs).item()
        if val_max <= 1.0 and val_min >= 0.:
            feat = torch.sigmoid(self.feat(h))
        else:
            # OGBN-arXiv范围不是0-1
            feat = self.feat(h)
            # feat = self.feat(h) - 2.0

        edge_weight = self.edge(h)
        edge_weight = GW(edge_weight, thrd, self.device)

        return feat, edge_weight


class HomoLoss(nn.Module):
    def __init__(self, args, device):
        super(HomoLoss, self).__init__()
        self.args = args
        self.device = device

    def forward(self, trigger_edge_index, trigger_edge_weights, x, thrd):
        trigger_edge_index = trigger_edge_index[:, trigger_edge_weights > 0.0]
        edge_sims = F.cosine_similarity(x[trigger_edge_index[0]], x[trigger_edge_index[1]])

        loss = torch.relu(thrd - edge_sims).mean()
        return loss


def calc_euc_dis(h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
    hh11 = (h1 * h1).sum(-1).reshape(-1, 1).repeat(1, h2.shape[0])
    hh22 = (h2 * h2).sum(-1).reshape(1, -1).repeat(h1.shape[0], 1)
    hh11_hh22 = hh11 + hh22
    hh12 = h1 @ h2.T
    distance = hh11_hh22 - 2 * hh12
    return distance


class GTA:
    """ Graph Backdoor
    """

    def __init__(self, thrd, hidden, trojan_epochs, loss_factor,
                 lr, weight_decay, trigger_size, target_class, device):

        self.shadow_model = None
        self.trojan = None
        self.device = device
        self.trigger_size = trigger_size
        self.target_class = target_class
        self.thrd = thrd
        self.hidden = hidden
        self.trojan_epochs = trojan_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_factor = loss_factor
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

    def inject_trigger(self, attach_idx, features, edge_index, edge_weight):
        self.trojan.eval()
        features, edge_index, edge_weight = features.clone(), edge_index.clone(), edge_weight.clone()

        trojan_feat, trojan_weight = self.trojan(features[attach_idx], self.thrd)  # may revise the process of generate

        trojan_weight = torch.cat([torch.ones([len(attach_idx), 1], dtype=torch.float, device=self.device), trojan_weight], dim=1)
        trojan_weight = trojan_weight.flatten()

        trojan_feat = trojan_feat.view([-1, features.shape[1]])

        trojan_edge = self.get_trojan_edge(len(features), attach_idx, self.trigger_size).to(self.device)

        update_edge_weights = torch.cat([edge_weight, trojan_weight, trojan_weight])
        update_feat = torch.cat([features, trojan_feat])
        update_edge_index = torch.cat([edge_index, trojan_edge], dim=1)

        return update_feat, update_edge_index, update_edge_weights

    def fit(self, features, edge_index, edge_weight, labels, train_idx, attach_idx, idx_unlabeled=None):

        if edge_weight is None:
            edge_weight = torch.ones([edge_index.shape[1]], device=self.device, dtype=torch.float)

        # initial a shadow model
        self.shadow_model = GCN(n_feat=features.shape[1],
                                n_hid=self.hidden,
                                n_class=labels.max().item() + 1,
                                dropout=0.0, device=self.device).to(self.device)

        # initialize a trojanNet to generate trigger
        self.trojan = GraphTrojanNet(self.device, features.shape[1], self.trigger_size, layer_num=2).to(self.device)

        optimizer_shadow = optim.Adam(self.shadow_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        optimizer_trigger = optim.Adam(self.trojan.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # change the labels of the poisoned node to the target class
        poisoned_labels = labels.clone()
        poisoned_labels[attach_idx] = self.target_class
        node_idx = torch.arange(labels.shape[0], dtype=torch.long, device=self.device)
        target_cls_node_idx = node_idx[labels == self.target_class]

        # get the trojan edges, which include the target-trigger edge and the edges among trigger
        trojan_edge = self.get_trojan_edge(len(features), attach_idx, self.trigger_size).to(self.device)

        # update the poisoned graph's edge index
        poison_edge_index = torch.cat([edge_index, trojan_edge], dim=1)

        logger.info('Training Model to Craft Poisoned Graphs')

        self.shadow_model.train()
        for i in range(self.trojan_epochs):
            optimizer_shadow.zero_grad()
            output = self.shadow_model(features, edge_index, edge_weight)
            loss_train_cls = F.cross_entropy(output[train_idx], labels[train_idx])
            loss_train_cls.backward()
            optimizer_shadow.step()

        self.trojan.train()
        self.shadow_model.eval()
        for i in range(self.trojan_epochs):
            optimizer_trigger.zero_grad()

            # may revise the process of generate
            trojan_feat, trojan_weight = self.trojan(features[attach_idx], self.thrd)
            trojan_weight = torch.cat([torch.ones([len(trojan_feat), 1], dtype=torch.float, device=self.device), trojan_weight], dim=1)
            trojan_weight = trojan_weight.flatten()
            trojan_feat = trojan_feat.view([-1, features.shape[1]])

            # repeat trojan weights because of undirected edge
            poison_edge_weight = torch.cat([edge_weight, trojan_weight, trojan_weight])
            poison_x = torch.cat([features, trojan_feat])

            output = self.shadow_model(poison_x, poison_edge_index, poison_edge_weight)

            # classification loss
            loss_train_cls = F.cross_entropy(output[target_cls_node_idx], poisoned_labels[target_cls_node_idx])
            loss_attach_cls = F.cross_entropy(output[attach_idx], poisoned_labels[attach_idx])

            # add our adaptive loss
            loss_embedding_dist = calc_euc_dis(features[target_cls_node_idx], trojan_feat).mean()
            if i % 50 == 0:
                logger.info('Loss on Attach Idx = {:.2f}@Loss Embedding = {:.2f}'.format(loss_attach_cls.item(), loss_embedding_dist.item()))

            loss = torch.abs(loss_attach_cls - loss_train_cls) + self.loss_factor * loss_embedding_dist

            loss.backward()
            optimizer_trigger.step()

    @torch.no_grad()
    def get_poisoned(self, features, edge_index, edge_weight, labels, attach_idx):

        if edge_weight is None:
            edge_weight = torch.ones([edge_index.shape[1]], device=self.device, dtype=torch.float)

        poisoned_x, poisoned_edge_index, poisoned_edge_weight = self.inject_trigger(
            attach_idx, features, edge_index, edge_weight
        )

        poisoned_labels = labels.clone()
        poisoned_labels[attach_idx] = self.target_class

        poisoned_edge_index = poisoned_edge_index[:, poisoned_edge_weight > 0.0]
        poisoned_edge_weight = poisoned_edge_weight[poisoned_edge_weight > 0.0]
        return poisoned_x, poisoned_edge_index, poisoned_edge_weight, poisoned_labels
