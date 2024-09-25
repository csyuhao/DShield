import logging
from abc import ABC, abstractmethod
from itertools import chain
from typing import Optional, Tuple, NamedTuple, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch_geometric.utils import index_to_mask

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


def sample_node_feat(ori_feat, latent, feat_budget, tau):
    smoothed_latent = F.sigmoid(latent * tau)
    chosen_indices = torch.topk(smoothed_latent, k=feat_budget)[1]
    target_feat = ori_feat.clone()
    target_feat[chosen_indices] = ori_feat[chosen_indices] + (1 - 2 * ori_feat[chosen_indices]) * smoothed_latent[chosen_indices]
    return chosen_indices, target_feat


class GCBA(object):
    """ Graph Contrastive Backdoor Attacks
    """

    def __init__(self, num_feat, num_hidden,
                 num_labels, feat_budget, trojan_epochs,
                 ssl_tau, tau, lr, weight_decay, edge_drop_ratio, target_class, device):

        self.num_feat = num_feat
        self.hidden = num_hidden
        self.num_labels = num_labels
        self.device = device
        self.feat_budget = feat_budget
        self.trojan_epochs = trojan_epochs
        self.tau = tau
        self.ssl_tau = ssl_tau
        self.edge_drop_ratio = edge_drop_ratio
        self.lr = lr
        self.weight_decay = weight_decay

        # initial a shadow model
        self.shadow_model = GCN(n_feat=num_feat,
                                n_hid=num_hidden,
                                n_class=num_labels,
                                dropout=0.0, device=device).to(device)
        self.ori_feat = torch.zeros(size=(num_feat,), dtype=torch.float32, device=device)
        self.latent_vec = torch.randn(
            size=(num_feat,), dtype=torch.float32, device=self.device
        ).requires_grad_()
        self.target_class = target_class

    def fit(self, features, edge_index, edge_weight, labels, idx_train, attach_idx, dataset):

        if edge_weight is None:
            edge_weight = torch.ones([edge_index.shape[1]], device=self.device, dtype=torch.float)

        aug1 = Compose([EdgeRemoving(pe=self.edge_drop_ratio),])
        aug2 = Compose([EdgeRemoving(pe=self.edge_drop_ratio),])
        node_idx = torch.cat([idx_train, attach_idx], dim=0).long()
        optimizer = optim.Adam(
            chain(self.shadow_model.parameters(), [self.latent_vec]),
            lr=self.lr, weight_decay=self.weight_decay
        )
        self.ori_feat = features[attach_idx[0]]

        self.shadow_model.train()
        for i in range(self.trojan_epochs):
            optimizer.zero_grad()
            n_feat = features.clone()
            tau = self.tau * (1 + i / 10)
            chosen_indices, sampled_weights = sample_node_feat(self.ori_feat, self.latent_vec, self.feat_budget, tau)

            feat_mask = torch.zeros(self.ori_feat.shape[0], dtype=torch.float32, device=self.device)
            feat_mask[chosen_indices] = 1.
            feat_mask = feat_mask.reshape(1, -1).repeat(attach_idx.shape[0], 1)
            n_feat[attach_idx] = (1 - feat_mask) * n_feat[attach_idx] + feat_mask * sampled_weights

            # Sample positive and negative samples according to labels
            cur_node_idx = node_idx
            if dataset == 'Flickr':
                cur_node_idx = node_idx[torch.randperm(node_idx.shape[0])[:10000]]
            elif dataset == 'ogbn-arxiv':
                cur_node_idx = node_idx[torch.randperm(node_idx.shape[0])[:12000]]
            elif dataset == 'ogbn-products':
                cur_node_idx = node_idx[torch.randperm(node_idx.shape[0])[:1000]]

            feat1, edge_index1, edge_weight1, _ = aug1(n_feat, edge_index, edge_weight, cur_node_idx)
            feat2, edge_index2, edge_weight2, _ = aug2(n_feat, edge_index, edge_weight, cur_node_idx)

            embedding = self.shadow_model(n_feat, edge_index, edge_weight)
            embedding1 = self.shadow_model(feat1, edge_index1, edge_weight1)
            embedding2 = self.shadow_model(feat2, edge_index2, edge_weight2)

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

        self.tau = self.tau * (1 + self.trojan_epochs / 10)

    @torch.no_grad()
    def get_poisoned(self, features, edge_index, edge_weight, labels, attach_idx):

        if edge_weight is None:
            edge_weight = torch.ones([edge_index.shape[1]], device=self.device, dtype=torch.float)

        chosen_indices, target_feat = sample_node_feat(self.ori_feat, self.latent_vec, self.feat_budget, self.tau)
        poison_x = features.clone()

        feat_mask = torch.zeros(self.ori_feat.shape[0], dtype=torch.float32, device=self.device)
        feat_mask[chosen_indices] = 1.
        feat_mask = feat_mask.reshape(1, -1).repeat(attach_idx.shape[0], 1)
        poison_x[attach_idx] = (1 - feat_mask) * poison_x[attach_idx] + feat_mask * target_feat

        poisoned_labels = labels.clone()
        poisoned_labels[attach_idx] = self.target_class

        return poison_x, edge_index, edge_weight, poisoned_labels, chosen_indices

    def inject_trigger(self, attach_idx, features, edge_index, edge_weight):
        features, edge_index, edge_weight = features.clone(), edge_index.clone(), edge_weight.clone()
        chosen_indices, target_feat = sample_node_feat(self.ori_feat, self.latent_vec, self.feat_budget, self.tau)
        poison_x = features.clone()

        feat_mask = torch.zeros(self.ori_feat.shape[0], dtype=torch.float32, device=self.device)
        feat_mask[chosen_indices] = 1.
        feat_mask = feat_mask.reshape(1, -1).repeat(attach_idx.shape[0], 1)
        poison_x[attach_idx] = (1 - feat_mask) * poison_x[attach_idx] + feat_mask * target_feat

        return poison_x, edge_index, edge_weight
