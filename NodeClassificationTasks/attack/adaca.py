# Adaptive Clean-Label Backdoor Attacks
import logging
from abc import ABC, abstractmethod
from itertools import chain
from typing import Optional, Tuple, NamedTuple, List

from umap.umap_ import fuzzy_simplicial_set, make_epochs_per_sample
from umap.umap_ import find_ab_params
from pynndescent import NNDescent

import numpy as np
import torch
import torch.nn as nn
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


class Encoder(nn.Module):

    def __init__(self, input_dim, output_dim=2):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        return x


class UMAPDataset:

    def __init__(self, data, epochs_per_sample, head, tail, weight, device='cpu', batch_size=1000):

        """
        create dataset for iteration on graph edges

        """
        self.weigh = weight
        self.batch_size = batch_size
        self.data = data
        self.device = device

        self.edges_to_exp, self.edges_from_exp = (
            np.repeat(head, epochs_per_sample.astype("int")),
            np.repeat(tail, epochs_per_sample.astype("int")),
        )
        self.num_edges = len(self.edges_to_exp)

        # shuffle edges
        shuffle_mask = np.random.permutation(range(len(self.edges_to_exp)))
        self.edges_to_exp = self.edges_to_exp[shuffle_mask]
        self.edges_from_exp = self.edges_from_exp[shuffle_mask]

    def get_batches(self):
        batches_per_epoch = int(self.num_edges / self.batch_size / 5)
        for _ in range(batches_per_epoch):
            rand_index = np.random.randint(0, len(self.edges_to_exp) - 1, size=self.batch_size)
            batch_index_to = self.edges_to_exp[rand_index]
            batch_index_from = self.edges_from_exp[rand_index]
            if self.device.startswith('cuda'):
                batch_to = torch.Tensor(self.data[batch_index_to]).cuda()
                batch_from = torch.Tensor(self.data[batch_index_from]).cuda()
            else:
                batch_to = torch.Tensor(self.data[batch_index_to])
                batch_from = torch.Tensor(self.data[batch_index_from])
            yield batch_to, batch_from


class ConstructUMAPGraph:

    def __init__(self, metric='euclidean', n_neighbors=10, batch_size=1000, random_state=42):
        self.batch_size = batch_size
        self.random_state = random_state
        self.metric = metric  # distance metric
        self.n_neighbors = n_neighbors  # number of neighbors for computing k-neighbor graph

        pass

    @staticmethod
    def get_graph_elements(graph_, n_epochs):

        """
        gets elements of graphs, weights, and number of epochs per edge
        Parameters
        ----------
        graph_ : scipy.sparse.csr.csr_matrix
            umap graph of probabilities
        n_epochs : int
            maximum number of epochs per edge
        Returns
        -------
        graph scipy.sparse.csr.csr_matrix
            umap graph
        epochs_per_sample np.array
            number of epochs to train each sample for
        head np.array
            edge head
        tail np.array
            edge tail
        weight np.array
            edge weight
        n_vertices int
            number of vertices in graph
        """

        graph = graph_.tocoo()
        # eliminate duplicate entries by summing them together
        graph.sum_duplicates()
        # number of vertices in dataset
        n_vertices = graph.shape[1]
        # get the number of epochs based on the size of the dataset
        if n_epochs is None:
            # For smaller datasets we can use more epochs
            if graph.shape[0] <= 10000:
                n_epochs = 500
            else:
                n_epochs = 200
        # remove elements with very low probability
        graph.data[graph.data < (graph.data.max() / float(n_epochs))] = 0.0
        graph.eliminate_zeros()
        # get epochs per sample based upon edge probability
        epochs_per_sample = make_epochs_per_sample(graph.data, n_epochs)

        head = graph.row
        tail = graph.col
        weight = graph.data

        return graph, epochs_per_sample, head, tail, weight, n_vertices

    def __call__(self, X):
        # number of trees in random projection forest
        n_trees = 5 + int(round((X.shape[0]) ** 0.5 / 20.0))
        # max number of nearest neighbor iters to perform
        n_iters = max(5, int(round(np.log2(X.shape[0]))))

        # get nearest neighbors
        nnd = NNDescent(
            X.reshape((len(X), np.product(np.shape(X)[1:]))),
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            n_trees=n_trees,
            n_iters=n_iters,
            max_candidates=60,
            verbose=True
        )
        # get indices and distances
        knn_indices, knn_dists = nnd.neighbor_graph

        # build fuzzy_simplicial_set
        umap_graph, sigmas, rhos = fuzzy_simplicial_set(
            X=X,
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            random_state=self.random_state,
            knn_indices=knn_indices,
            knn_dists=knn_dists,
        )

        graph, epochs_per_sample, head, tail, weight, n_vertices = self.get_graph_elements(umap_graph, None)
        return epochs_per_sample, head, tail, weight


class UMAPLoss(nn.Module):

    def __init__(self, device='cpu', min_dist=0.1, batch_size=1000, negative_sample_rate=5,
                 edge_weight=None, repulsion_strength=1.0):

        """
        batch_size : int
        size of mini-batches
        negative_sample_rate : int
          number of negative samples per positive samples to train on
        _a : float
          distance parameter in embedding space
        _b : float float
          distance parameter in embedding space
        edge_weights : array
          weights of all edges from sparse UMAP graph
        parametric_embedding : bool
          whether the embedding is parametric or nonparametric
        repulsion_strength : float, optional
          strength of repulsion vs attraction for cross-entropy, by default 1.0
        """

        super().__init__()
        self.device = device
        self._a, self._b = find_ab_params(1.0, min_dist)
        self.batch_size = batch_size
        self.negative_sample_rate = negative_sample_rate
        self.repulsion_strength = repulsion_strength

    @staticmethod
    def convert_distance_to_probability(distances, a=1.0, b=1.0):
        return 1.0 / (1.0 + a * distances ** (2 * b))

    def compute_cross_entropy(self, probabilities_graph, probabilities_distance, EPS=1e-4, repulsion_strength=1.0):
        # cross entropy
        attraction_term = -probabilities_graph * torch.log(
            torch.clamp(probabilities_distance, EPS, 1.0)
        )

        repellent_term = -(1.0 - probabilities_graph) * torch.log(torch.clamp(
            1.0 - probabilities_distance, EPS, 1.0
        )) * self.repulsion_strength
        CE = attraction_term + repellent_term
        return attraction_term, repellent_term, CE

    def forward(self, embedding_to, embedding_from):
        # get negative samples
        embedding_neg_to = torch.repeat_interleave(embedding_to, self.negative_sample_rate, dim=0)
        repeat_neg = torch.repeat_interleave(embedding_from, self.negative_sample_rate, dim=0)
        if self.device.startswith('cuda'):
            embedding_neg_from = torch.index_select(repeat_neg, 0, torch.randperm(repeat_neg.size(0)).cuda())
        else:
            embedding_neg_from = torch.index_select(repeat_neg, 0, torch.randperm(repeat_neg.size(0)))

        #  distances between samples (and negative samples)
        distance_embedding = torch.cat(
            [
                torch.norm(embedding_to - embedding_from, dim=1),
                torch.norm(embedding_neg_to - embedding_neg_from, dim=1)
            ],
            dim=0)

        # convert probabilities to distances
        probabilities_distance = self.convert_distance_to_probability(
            distance_embedding, self._a, self._b
        )

        # set true probabilities based on negative sampling
        if self.device.startswith('cuda'):
            probabilities_graph = torch.cat(
                [torch.ones(self.batch_size).cuda(), torch.zeros(self.batch_size * self.negative_sample_rate).cuda()],
                dim=0
            )
        else:
            probabilities_graph = torch.cat(
                [torch.ones(self.batch_size), torch.zeros(self.batch_size * self.negative_sample_rate)],
                dim=0
            )

        # compute cross entropy
        (attraction_loss, repellent_loss, ce_loss) = self.compute_cross_entropy(
            probabilities_graph,
            probabilities_distance,
            repulsion_strength=self.repulsion_strength,
        )

        return torch.mean(ce_loss)


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


def calc_euc_dis(h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
    hh11 = (h1 * h1).sum(-1).reshape(-1, 1).repeat(1, h2.shape[0])
    hh22 = (h2 * h2).sum(-1).reshape(1, -1).repeat(h1.shape[0], 1)
    hh11_hh22 = hh11 + hh22
    hh12 = h1 @ h2.T
    distance = hh11_hh22 - 2 * hh12
    return distance


class AdaCA(object):

    def __init__(self, num_feat, num_hidden,
                 num_labels, feat_budget, trojan_epochs, umap_epochs,
                 ssl_tau, tau, lr, weight_decay, reg_weight, edge_drop_ratio, target_class, device):

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
        self.target_class = target_class
        self.reg_weight = reg_weight
        self.umap_epochs = umap_epochs

        # initial a shadow model
        self.shadow_model = GCN(n_feat=num_feat,
                                n_hid=num_hidden,
                                n_class=num_labels,
                                dropout=0.0, device=device).to(device)
        self.ori_feat = torch.zeros(size=(num_feat,), dtype=torch.float32, device=device)
        self.latent_vec = torch.randn(
            size=(num_feat,), dtype=torch.float32, device=self.device
        ).requires_grad_()

    def fit(self, features, edge_index, edge_weight, labels, idx_train, attach_idx, dataset):

        if dataset == 'Cora':
            batch_size = 128
        elif dataset == 'PubMed':
            batch_size = 20480
        elif dataset == 'Flickr':
            batch_size = 20480
        else:
            batch_size = 40960

        if edge_weight is None:
            edge_weight = torch.ones([edge_index.shape[1]], device=self.device, dtype=torch.float)

        aug1 = Compose([EdgeRemoving(pe=self.edge_drop_ratio), ])
        aug2 = Compose([EdgeRemoving(pe=self.edge_drop_ratio), ])
        node_idx = torch.cat([idx_train, attach_idx], dim=0).long()
        optimizer = optim.Adam(
            chain(self.shadow_model.parameters(), [self.latent_vec]),
            lr=self.lr, weight_decay=self.weight_decay
        )
        self.ori_feat = features[attach_idx[0]]
        target_node_idx = node_idx[labels[node_idx] == self.target_class]

        # construct graph of nearest neighbors
        graph_constructor = ConstructUMAPGraph(metric='euclidean', n_neighbors=target_node_idx.shape[0] // 2 + 1, batch_size=batch_size, random_state=42)
        umap_model = Encoder(input_dim=features.shape[1], output_dim=self.hidden).to(self.device)
        criterion = UMAPLoss(device=self.device, min_dist=0.1, batch_size=batch_size, negative_sample_rate=5, edge_weight=None, repulsion_strength=1.0)

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

            if i == 0 or (i + 1) % 50 == 0:
                umap_model.train()
                epochs_per_sample, head, tail, weight = graph_constructor(n_feat[target_node_idx].detach().cpu().numpy())
                dataset = UMAPDataset(
                    n_feat[target_node_idx].detach().cpu().numpy(), epochs_per_sample, head, tail, weight, device=self.device, batch_size=batch_size
                )

                sub_optimizer = optim.Adam(
                    umap_model.parameters(), lr=self.lr, weight_decay=self.weight_decay
                )

                for epoch in range(self.umap_epochs):
                    train_loss, iter_round = 0., 0
                    for batch_to, batch_from in dataset.get_batches():
                        sub_optimizer.zero_grad()
                        batch_to, batch_from = batch_to.to(self.device), batch_from.to(self.device)
                        embedding_to = umap_model(batch_to)
                        embedding_from = umap_model(batch_from)
                        loss = criterion(embedding_to, embedding_from)
                        train_loss += loss.item()
                        loss.backward()
                        sub_optimizer.step()
                        iter_round += 1
                    if epoch % 50 == 0:
                        logger.info('UMAP@epoch: {}, loss: {}'.format(epoch, train_loss / iter_round))

            umap_model.eval()
            lower_dim_embedding = umap_model(n_feat[target_node_idx])
            loss_reg = torch.mean(calc_euc_dis(lower_dim_embedding, lower_dim_embedding))
            loss = 0.5 * loss1 + 0.5 * loss2 + self.reg_weight * loss_reg

            loss.backward()
            optimizer.step()

            if i == 0 or (i + 1) % 50 == 0:
                logger.info('SSL@Epoch = {}@Loss = {:.4f}@Loss Contrastive = {:.4f}@Loss Regularize = {:.4f}'.format(
                    i + 1, loss.item(), 0.5 * loss1.item() + 0.5 * loss2.item(), loss_reg.item()
                ))

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
        return poison_x, edge_index, edge_weight, labels, chosen_indices

    def inject_trigger(self, attach_idx, features, edge_index, edge_weight):
        features, edge_index, edge_weight = features.clone(), edge_index.clone(), edge_weight.clone()
        chosen_indices, target_feat = sample_node_feat(self.ori_feat, self.latent_vec, self.feat_budget, self.tau)
        poison_x = features.clone()

        feat_mask = torch.zeros(self.ori_feat.shape[0], dtype=torch.float32, device=self.device)
        feat_mask[chosen_indices] = 1.
        feat_mask = feat_mask.reshape(1, -1).repeat(attach_idx.shape[0], 1)
        poison_x[attach_idx] = (1 - feat_mask) * poison_x[attach_idx] + feat_mask * target_feat
        return poison_x, edge_index, edge_weight
