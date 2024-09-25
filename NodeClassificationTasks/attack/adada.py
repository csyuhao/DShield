# Adaptive Dirty-Label Backdoor Attacks

import logging
import numpy as np
from copy import deepcopy
from abc import ABC, abstractmethod
from typing import Optional, Tuple, NamedTuple, List, Any

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.utils import index_to_mask

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
        inputs, = ctx.saved_tensors
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
        for _ in range(layer_num - 1):
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
        self.layers = self.layers
        h = self.layers(inputs)

        val_min, val_max = torch.min(inputs).item(), torch.max(inputs).item()
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


class Graph(NamedTuple):
    x: torch.FloatTensor
    edge_index: torch.LongTensor
    edge_weight: Optional[torch.FloatTensor]
    node_idx: Optional[torch.LongTensor]

    def unfold(self) -> Tuple[torch.FloatTensor, torch.LongTensor, Optional[torch.FloatTensor], Optional[torch.LongTensor]]:
        return self.x, self.edge_index, self.edge_weight, self.node_idx


class Augmentor(ABC):
    """Base class for graph augmentors."""

    def __init__(self):
        pass

    @abstractmethod
    def augment(self, g: Graph) -> Graph:
        raise NotImplementedError(f"GraphAug.augment should be implemented.")

    def __call__(
            self, x: torch.FloatTensor, edge_index: torch.LongTensor,
            edge_weight: Optional[torch.FloatTensor] = None,
            node_idx: Optional[torch.LongTensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self.augment(Graph(x, edge_index, edge_weight, node_idx)).unfold()


class Compose(Augmentor):
    def __init__(self, augmentors: List[Augmentor]):
        super(Compose, self).__init__()
        self.augmentors = augmentors

    def augment(self, g: Graph) -> Graph:
        for aug in self.augmentors:
            g = aug.augment(g)
        return g


class EdgeRemoving(Augmentor):
    def __init__(self, pe: float):
        super(EdgeRemoving, self).__init__()
        self.pe = pe

    def augment(self, g: Graph) -> Graph:
        x, edge_index, edge_weight, node_idx = g.unfold()
        edge_index, edge_id = dropout_edge(edge_index, node_idx=node_idx, num_nodes=x.shape[0], p=self.pe)
        return Graph(x=x, edge_index=edge_index, edge_weight=edge_weight[edge_id], node_idx=node_idx)


def dropout_edge(edge_index: torch.LongTensor, node_idx: torch.LongTensor = None, num_nodes: int = 0,
                 p: float = 0.5, force_undirected: bool = False, training: bool = True) -> Tuple[torch.LongTensor, torch.LongTensor]:

    def softmax(x):
        """ Softmax """
        x -= np.max(x)
        return np.exp(x) / np.sum(np.exp(x))

    if p < 0. or p > 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 (got {p}')

    if not training or p == 0.0:
        edge_mask = edge_index.new_ones(edge_index.size(1), dtype=torch.bool)
        return edge_index, edge_mask

    # Pick up effect edges
    row, col = edge_index
    device = edge_index.device
    node_mask = index_to_mask(node_idx, size=num_nodes)
    effect_edge_mask = node_mask[row] | node_mask[col]
    effect_num_edges = torch.sum(effect_edge_mask).item()
    list_candidates = np.arange(effect_num_edges, dtype=np.int32)
    list_chosen_candidates = np.random.choice(list_candidates, int(p * effect_num_edges))
    chosen_candidates = torch.tensor(list_chosen_candidates, dtype=torch.long, device=device)

    chosen_mask = torch.tensor([True] * effect_num_edges, dtype=torch.bool, device=device)
    chosen_mask[chosen_candidates] = False

    n_edge_mask = effect_edge_mask.clone().detach()
    n_edge_mask[effect_edge_mask] = chosen_mask
    edge_mask = n_edge_mask.clone().detach()

    if force_undirected:
        edge_mask[row > col] = False
    edge_index = edge_index[:, edge_mask]

    if force_undirected:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_mask = edge_mask.nonzero().repeat((2, 1)).squeeze()

    return edge_index, edge_mask


def calc_similarity(h1: torch.Tensor, h2: torch.Tensor):
    if h1.ndim == 2:
        h1 = F.normalize(h1, p=2, dim=-1)
        h2 = F.normalize(h2, p=2, dim=-1)
        return h1 @ h2.t()
    elif h1.ndim == 3:
        # b x 1 x m * b x m x 1   ===> b
        h1 = F.normalize(h1, p=2, dim=-1)
        h2 = F.normalize(h2, p=2, dim=1)
        return torch.bmm(h1, h2).reshape(-1)


def infonce_loss(anchor, sample, pos_mask, neg_mask, tau):
    # InfoNCE Loss
    sim = calc_similarity(anchor, sample) / tau
    exp_sim = torch.exp(sim) * (pos_mask + neg_mask)
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
    loss = log_prob * pos_mask
    loss = loss.sum(dim=1) / pos_mask.sum(dim=1)
    return - loss.mean()


def calc_euc_dis(h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
    hh11 = (h1 * h1).sum(-1).reshape(-1, 1).repeat(1, h2.shape[0])
    hh22 = (h2 * h2).sum(-1).reshape(1, -1).repeat(h1.shape[0], 1)
    hh11_hh22 = hh11 + hh22
    hh12 = h1 @ h2.T
    distance = hh11_hh22 - 2 * hh12
    return distance


class AdaDA:

    def __init__(self, seed, thrd, hidden, trojan_epochs, inner_epochs,
                 lr, weight_decay, target_loss_weight, homo_loss_weight, reg_loss_weight,
                 homo_boost_thrd, trigger_size, target_class, edge_drop_ratio, ssl_tau, dataset, device):

        self.shadow_model = None
        self.ssl_model = None
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

        self.dataset = dataset
        self.ssl_tau = ssl_tau
        self.edge_drop_ratio = edge_drop_ratio
        self.reg_loss_weight = reg_loss_weight

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

    def ssl_training(self, features, edge_index, edge_weight, idx_train, attach_idx, dataset):

        aug1 = Compose([EdgeRemoving(pe=self.edge_drop_ratio), ])
        aug2 = Compose([EdgeRemoving(pe=self.edge_drop_ratio), ])
        node_idx = torch.cat([idx_train, attach_idx], dim=0).long()
        optimizer = optim.Adam(
            self.ssl_model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        self.ssl_model.train()
        for i in range(self.trojan_epochs):
            optimizer.zero_grad()

            # Sample positive and negative samples according to labels
            cur_node_idx = node_idx
            if dataset == 'Flickr':
                cur_node_idx = node_idx[torch.randperm(node_idx.shape[0])[:10000]]
            elif dataset == 'ogbn-arxiv':
                cur_node_idx = node_idx[torch.randperm(node_idx.shape[0])[:12000]]

            feat1, edge_index1, edge_weight1, _ = aug1(features, edge_index, edge_weight, cur_node_idx)
            feat2, edge_index2, edge_weight2, _ = aug2(features, edge_index, edge_weight, cur_node_idx)

            embedding = self.ssl_model(features, edge_index, edge_weight)
            embedding1 = self.ssl_model(feat1, edge_index1, edge_weight1)
            embedding2 = self.ssl_model(feat2, edge_index2, edge_weight2)

            # Sample positive and negative samples according to labels
            part_num_nodes = cur_node_idx.shape[0]
            pos_mask = torch.eye(part_num_nodes, dtype=torch.float32, device=self.device)
            neg_mask = 1. - pos_mask

            loss1 = infonce_loss(embedding[cur_node_idx], embedding1[cur_node_idx], pos_mask, neg_mask, self.ssl_tau)
            loss2 = infonce_loss(embedding[cur_node_idx], embedding2[cur_node_idx], pos_mask, neg_mask, self.ssl_tau)
            loss = 0.5 * loss1 + 0.5 * loss2

            loss.backward()

            optimizer.step()

            if i == 0 or (i + 1) % 50 == 0:
                logger.info('SSL@Epoch = {}@Loss = {:.4f}'.format(i + 1, loss.item()))

    def fit(self, features, edge_index, edge_weight, labels, train_idx, attach_idx, unlabeled_idx):

        if edge_weight is None:
            edge_weight = torch.ones([edge_index.shape[1]], device=self.device, dtype=torch.float)

        # initial a shadow model
        self.shadow_model = GCN(
            n_feat=features.shape[1], n_hid=self.hidden,
            n_class=labels.max().item() + 1, dropout=0.0, device=self.device
        ).to(self.device)

        self.ssl_model = GCN(
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

            if i % 50 == 0:
                self.ssl_training(features, edge_index, edge_weight, train_idx, attach_idx, dataset=self.dataset)

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
            self.ssl_model.eval()

            optimizer_trigger.zero_grad()

            rs = np.random.RandomState(self.seed)
            outer_idx = torch.cat([attach_idx, unlabeled_idx[rs.choice(len(unlabeled_idx), size=512, replace=False)]])
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

            embedding = self.ssl_model(update_feat, update_edge_index, update_edge_weights)
            target_embedding = embedding[train_idx[labels[train_idx] == self.target_class]]
            origin_embedding = embedding[attach_idx]
            loss_reg = calc_euc_dis(origin_embedding, target_embedding).mean()

            loss_outer = loss_target + self.homo_loss_weight * loss_homo + self.reg_loss_weight * loss_reg

            loss_outer.backward()
            optimizer_trigger.step()
            acc_train_outer = (output[outer_idx].argmax(dim=1) == self.target_class).float().mean()

            if loss_outer < loss_best:
                best_state_dict = deepcopy(self.trojan.state_dict())
                loss_best = float(loss_outer)

            if i % 10 == 0:
                logger.info('Epoch {}, loss_outer: {:.5f},  loss_inner: {:.5f}, loss_target: {:.5f}, homo loss: {:.5f}, reg loss: {:.5f} '.format(
                    i, loss_outer.item(), loss_inner, loss_target, loss_homo, loss_reg
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

