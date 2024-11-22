import copy
import logging
from itertools import chain
from abc import ABC, abstractmethod
from typing import Optional, Tuple, NamedTuple, List

import numpy as np
from sklearn.manifold import TSNE
import matplotlib
from matplotlib import pyplot as plt
from scipy.stats import norm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torch_geometric.nn as geo_nn
from torch import autograd
from torch_geometric.utils import index_to_mask, k_hop_subgraph

from models.metric import accuracy

from umap import UMAP
from sklearn.cluster import HDBSCAN


try:
    if 'logger' not in globals():
        logging.basicConfig()
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
except NameError:
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)


class UnsupervisedModel(nn.Module):
    """ Unsupervised Model for contrastive learning
    """

    def __init__(self, input_dim, proj_dim, hidden_dim, num_classes, arch):
        super(UnsupervisedModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.proj_dim = proj_dim
        self.num_classes = num_classes
        self.input_dim = input_dim

        if arch == 'GCN':
            self.encoder = nn.ModuleList([
                geo_nn.GCNConv(in_channels=input_dim, out_channels=proj_dim),
                nn.ReLU(),
                geo_nn.GCNConv(in_channels=proj_dim, out_channels=hidden_dim)
            ])
            self.contra_head = nn.ModuleList([
                geo_nn.GCNConv(in_channels=hidden_dim, out_channels=proj_dim),
                nn.ReLU(),
                geo_nn.GCNConv(in_channels=proj_dim, out_channels=input_dim)
            ])
        elif arch == 'GAT':
            self.encoder = nn.ModuleList([
                geo_nn.GATConv(input_dim, proj_dim // 8, 8),
                nn.ReLU(),
                geo_nn.GATConv(proj_dim, hidden_dim // 8, 8)
            ])
            self.contra_head = nn.ModuleList([
                geo_nn.GATConv(hidden_dim, proj_dim // 8, 8),
                nn.ReLU(),
                geo_nn.GATConv(proj_dim, input_dim, 1, concat=False)
            ])
        elif arch == 'GraphSage':
            self.encoder = nn.ModuleList([
                geo_nn.SAGEConv(in_channels=input_dim, out_channels=proj_dim),
                nn.ReLU(),
                geo_nn.SAGEConv(in_channels=proj_dim, out_channels=hidden_dim)
            ])
            self.contra_head = nn.ModuleList([
                geo_nn.SAGEConv(in_channels=hidden_dim, out_channels=proj_dim),
                nn.ReLU(),
                geo_nn.SAGEConv(in_channels=proj_dim, out_channels=input_dim)
            ])

    def rtn_contra_param(self):
        return self.contra_head.parameters()

    def rtn_encoder_param(self):
        return self.encoder.parameters()

    def encode(self, x, edge_index, edge_weight=None):
        z = x
        for layer in self.encoder:
            if isinstance(layer, geo_nn.GATConv) or isinstance(layer, geo_nn.GCNConv):
                z = layer(z, edge_index, edge_weight)
            elif isinstance(layer, geo_nn.SAGEConv):
                z = layer(z, edge_index)
            else:
                z = layer(z)
        return z

    def contra_pred(self, x, edge_index, edge_weight=None):
        z = x
        for layer in self.contra_head:
            if isinstance(layer, geo_nn.GATConv) or isinstance(layer, geo_nn.GCNConv):
                z = layer(z, edge_index, edge_weight)
            elif isinstance(layer, geo_nn.SAGEConv):
                z = layer(z, edge_index)
            else:
                z = layer(z)
        return z


class SupervisedModel(nn.Module):
    """ Supervised Model for poisoning and cleaning training
    """

    def __init__(self, input_dim, proj_dim, hidden_dim, num_classes, arch):
        super(SupervisedModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.proj_dim = proj_dim
        self.num_classes = num_classes
        self.input_dim = input_dim

        if arch == 'GCN':
            self.encoder = nn.ModuleList([
                geo_nn.GCNConv(in_channels=input_dim, out_channels=proj_dim),
                nn.ReLU(),
                geo_nn.GCNConv(in_channels=proj_dim, out_channels=hidden_dim)
            ])
        elif arch == 'GAT':
            self.encoder = nn.ModuleList([
                geo_nn.GATConv(input_dim, proj_dim // 8, 8),
                nn.ReLU(),
                geo_nn.GATConv(proj_dim, hidden_dim // 8, 8)
            ])
        elif arch == 'GraphSage':
            self.encoder = nn.ModuleList([
                geo_nn.SAGEConv(in_channels=input_dim, out_channels=proj_dim),
                nn.ReLU(),
                geo_nn.SAGEConv(in_channels=proj_dim, out_channels=hidden_dim)
            ])

        self.clf_head = nn.ModuleList([
            nn.Linear(hidden_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, num_classes)
        ])

    def rtn_clf_param(self):
        return self.clf_head.parameters()

    def clf_pred(self, z):
        for layer in self.clf_head:
            z = layer(z)
        return z

    def encode(self, x, edge_index, edge_weight=None):
        z = x
        for layer in self.encoder:
            if isinstance(layer, geo_nn.GATConv) or isinstance(layer, geo_nn.GCNConv):
                z = layer(z, edge_index, edge_weight)
            elif isinstance(layer, geo_nn.SAGEConv):
                z = layer(z, edge_index)
            else:
                z = layer(z)
        return z

    def forward(self, x, edge_index, edge_weight=None):
        embedding = self.encode(x, edge_index, edge_weight)
        z = self.clf_pred(embedding)
        return z


class Graph(NamedTuple):
    x: torch.FloatTensor
    feat_import: torch.FloatTensor
    edge_index: torch.LongTensor
    edge_weights: Optional[torch.FloatTensor]
    edge_sims: Optional[torch.FloatTensor]
    node_idx: Optional[torch.LongTensor]

    def unfold(self):
        return self.x, self.feat_import, self.edge_index, self.edge_weights, self.edge_sims, self.node_idx


class Augmentor(ABC):
    """Base class for graph augmentors."""

    def __init__(self):
        pass

    @abstractmethod
    def augment(self, g: Graph) -> Graph:
        raise NotImplementedError(f"GraphAug.augment should be implemented.")

    def __call__(
        self, x: torch.FloatTensor, feat_import: torch.FloatTensor,
        edge_index: torch.LongTensor, edge_weights: Optional[torch.FloatTensor] = None,
        edge_sims: Optional[torch.FloatTensor] = None, node_idx: Optional[torch.LongTensor] = None
    ):
        return self.augment(Graph(x, feat_import, edge_index, edge_weights, edge_sims, node_idx)).unfold()


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
        x, feat_import, edge_index, edge_weights, edge_sims, node_idx = g.unfold()
        edge_index, edge_id = dropout_edge(edge_index, edge_sims, node_idx=node_idx, num_nodes=x.shape[0], p=self.pe)
        return Graph(
            x=x, feat_import=feat_import, edge_index=edge_index,
            edge_weights=edge_weights[edge_id], edge_sims=edge_sims[edge_id], node_idx=node_idx
        )


class FeatureMasking(Augmentor):
    def __init__(self, pf: float):
        super(FeatureMasking, self).__init__()
        self.pf = pf

    def augment(self, g: Graph) -> Graph:
        x, feat_import, edge_index, edge_weights, edge_sims, node_idx = g.unfold()
        x = drop_feature(x, feat_import, node_idx, p=self.pf)
        return Graph(x=x, feat_import=feat_import, edge_index=edge_index, edge_weights=edge_weights, edge_sims=edge_sims, node_idx=node_idx)


def dropout_edge(edge_index: torch.LongTensor,
                 edge_sims: torch.FloatTensor, node_idx: torch.LongTensor = None, num_nodes: int = 0,
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
    effect_edge_mask = torch.logical_or(node_mask[row], node_mask[col])
    effect_num_edges = torch.sum(effect_edge_mask).item()

    # Generate candidate edges according to probabilities
    list_edge_probs = softmax(-edge_sims[effect_edge_mask].cpu().numpy())
    list_candidates = np.arange(effect_num_edges, dtype=np.int32)
    list_chosen_candidates = np.random.choice(list_candidates, int(p * effect_num_edges), p=list_edge_probs)
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


def drop_feature(feat: torch.FloatTensor, feat_import: torch.FloatTensor, node_idx: torch.LongTensor, p: float) -> torch.FloatTensor:
    num_nodes = feat.shape[0]
    device = feat.device
    train_node_mask = index_to_mask(node_idx, size=num_nodes)

    feat_mask = feat_import.bool()
    feat_mask[torch.logical_not(train_node_mask), :] = False

    drop_mask = torch.empty(size=feat.shape, dtype=torch.float32).uniform_(0, 1) <= p
    drop_mask[torch.logical_not(train_node_mask), :] = False
    drop_mask = drop_mask.to(device)

    mask = torch.logical_or(feat_mask.bool(), drop_mask.bool())
    feat = feat.clone()
    feat[mask] = 0
    return feat


def random_drop_embedding(x: torch.FloatTensor, drop_prob: float) -> torch.FloatTensor:
    device = x.device
    drop_mask = torch.empty((x.size(1),), dtype=torch.float32).uniform_(0, 1) <= drop_prob
    drop_mask = drop_mask.to(device)
    x = x.clone()
    x[:, drop_mask] = 0
    return x


def calc_euc_dis(h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
    hh11 = (h1 * h1).sum(-1).reshape(-1, 1).repeat(1, h2.shape[0])
    hh22 = (h2 * h2).sum(-1).reshape(1, -1).repeat(h1.shape[0], 1)
    hh11_hh22 = hh11 + hh22
    hh12 = h1 @ h2.T
    distance = hh11_hh22 - 2 * hh12
    return distance


def calc_similarity(h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
    if h1.ndim == 2:
        h1 = F.normalize(h1, p=2, dim=-1)
        h2 = F.normalize(h2, p=2, dim=-1)
        return h1 @ h2.t()
    elif h1.ndim == 3:
        # b x 1 x m * b x m x 1   ===> b
        h1 = F.normalize(h1, p=2, dim=-1)
        h2 = F.normalize(h2, p=2, dim=1)
        return torch.bmm(h1, h2).reshape(-1)


@torch.no_grad()
def calc_edge_sims(model, feat, edge_index):
    # Change to eval mode
    model.eval()
    poison_embedding = obtain_mid_embeddings(model, feat, edge_index)

    # Obtain `row` and `col`
    row, col = edge_index
    row_embedding, col_embedding = poison_embedding[row], poison_embedding[col]

    edge_sims = calc_similarity(
        torch.unsqueeze(row_embedding, dim=1), torch.unsqueeze(col_embedding, dim=2)
    )
    return edge_sims


def infonce_loss(anchor, sample, pos_mask, neg_mask, tau):
    # InfoNCE Loss
    sim = calc_similarity(anchor, sample) / tau
    exp_sim = torch.exp(sim) * (pos_mask + neg_mask)
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
    loss = log_prob * pos_mask
    loss = loss.sum(dim=1) / pos_mask.sum(dim=1)
    return - loss.mean()


def pretrain(model: UnsupervisedModel, optimizer, augmentor,
             feat, edge_index, edge_weight, node_idx, feat_import, edge_sims,
             drop_feat_ratio, kappa1, kappa2, tau=0.5, epochs=100, dataset=None):

    # Pretraining without any pseudo-label information
    model.train()

    device = feat.device
    num_nodes, num_feat = feat.shape
    aug1, aug2 = augmentor

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        # Sample positive and negative samples according to labels
        cur_node_idx = node_idx
        if dataset == 'Flickr':
            cur_node_idx = node_idx[torch.randperm(node_idx.shape[0], device=device)[:5000]]
        elif dataset == 'ogbn-arxiv':
            cur_node_idx = node_idx[torch.randperm(node_idx.shape[0], device=device)[:10000]]

        feat1, _, edge_index1, edge_weight1, _, _ = aug1(feat, feat_import, edge_index, edge_weight, edge_sims, cur_node_idx)
        feat2, _, edge_index2, edge_weight2, _, _ = aug2(feat, feat_import, edge_index, edge_weight, edge_sims, cur_node_idx)

        # Obtain embeddings
        embedding = model.encode(feat, edge_index, edge_weight)
        embedding1 = model.encode(feat1, edge_index1, edge_weight1)
        embedding2 = model.encode(feat2, edge_index2, edge_weight2)

        # Reconstruct features
        masked_embedding1 = random_drop_embedding(embedding1, drop_prob=drop_feat_ratio)
        masked_embedding2 = random_drop_embedding(embedding2, drop_prob=drop_feat_ratio)
        rx1 = model.contra_pred(masked_embedding1, edge_index1, edge_weight1)
        rx2 = model.contra_pred(masked_embedding2, edge_index2, edge_weight2)

        part_num_nodes = cur_node_idx.shape[0]
        pos_mask = torch.eye(part_num_nodes, dtype=torch.float32, device=device)
        neg_mask = 1. - pos_mask

        # InfoNCE loss
        loss1 = infonce_loss(embedding[cur_node_idx], embedding1[cur_node_idx], pos_mask, neg_mask, tau)
        loss2 = infonce_loss(embedding[cur_node_idx], embedding2[cur_node_idx], pos_mask, neg_mask, tau)
        loss_c = 0.5 * loss1 + 0.5 * loss2

        # Reconstruction loss
        num_nodes = feat.shape[0]
        reshaped_feat = feat.reshape(num_nodes, 1, num_feat)
        reshaped_rx1 = rx1.reshape(num_nodes, num_feat, 1)
        reshaped_rx2 = rx2.reshape(num_nodes, num_feat, 1)
        loss_r = 0.5 * (1 - calc_similarity(reshaped_feat[cur_node_idx], reshaped_rx1[cur_node_idx])) ** kappa2 + \
            0.5 * (1 - calc_similarity(reshaped_feat[cur_node_idx], reshaped_rx2[cur_node_idx])) ** kappa2
        loss_r = loss_r.mean()

        loss = loss_c + kappa1 * loss_r

        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 50 == 0:
            logger.info('Pretrain@Epoch = {}@Loss = {:.4f}@Loss Contra = {:.4f}@Loss Recon = {:.4f}'.format(
                epoch, loss.item(), loss_c.item(), loss_r.item()
            ))

    return model


def classify(model, optimizer,
             feat, edge_index, edge_weight, labels, kappa3, pos_node_idx, neg_node_idx, val_idx, attach_idx, epochs=100):

    model.initialize()
    model.train()
    device = feat.device

    _, neg_edge_index, _, edge_mask = k_hop_subgraph(neg_node_idx, num_hops=1,
                                                     edge_index=edge_index, num_nodes=feat.shape[0], relabel_nodes=False)
    neg_edge_weight = edge_weight[edge_mask]
    clf_loss_fn = nn.CrossEntropyLoss(reduction='mean')
    best_acc_val, best_asr, best_balance_metric, best_round, best_state_dict = 0., 0., 0., 0, copy.deepcopy(model.state_dict())
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        # Classification
        pos_logits = model(feat, edge_index, edge_weight)
        neg_logits = model(feat, neg_edge_index, neg_edge_weight)
        loss_pos, loss_neg = torch.tensor(0, dtype=torch.float32, device=device), \
            torch.tensor(0, dtype=torch.float32, device=device)
        loss_clf_pos, loss_clf_neg = loss_pos, loss_neg
        if pos_node_idx.shape[0] > 0:
            loss_clf_pos = clf_loss_fn(pos_logits[pos_node_idx], labels[pos_node_idx])
            loss_pos = torch.where(loss_clf_pos < 5, torch.exp(loss_clf_pos) - 1., loss_clf_pos)
        if neg_node_idx.shape[0] > 0:
            loss_clf_neg = clf_loss_fn(neg_logits[neg_node_idx], labels[neg_node_idx]) + 1.0
            loss_neg = torch.where(loss_clf_neg < 1e27, torch.log(loss_clf_neg), loss_clf_neg)
        loss = loss_pos - kappa3 * loss_neg

        loss.backward()
        optimizer.step()

        model.eval()
        output = model(feat, edge_index, edge_weight)
        acc_val = accuracy(output[val_idx], labels[val_idx])
        asr = accuracy(output[attach_idx], labels[attach_idx])
        balance_metric = 2 * (acc_val * (1 - asr)) / (acc_val + (1 - asr) + 1e-6)

        if epoch > int(0.8 * epochs) and best_balance_metric <= balance_metric:
            best_balance_metric = balance_metric
            best_acc_val = acc_val
            best_asr = asr
            best_state_dict = copy.deepcopy(model.state_dict())
            best_round = epoch

        if epoch == 1 or epoch % 50 == 0:
            logger.info(
                'Classify@Epoch = {}@Loss = {:.4f}@Loss Clf[P] = {:.4f}@Loss Clf[N] = {:.4f}'.format(
                    epoch, loss.item(), loss_clf_pos.mean().item(), loss_clf_neg.mean().item()
            ))
    logger.info('Best Round = {}@Best ACC = {:.4f}@Best ASR = {:4f}'.format(best_round, best_acc_val, best_asr))
    model.load_state_dict(best_state_dict)
    return model


@torch.no_grad()
def obtain_mid_embeddings(model, feat, edge_index):
    activation = {}
    model.eval()

    def get_activation(l_name):
        def hook(_, __, output):
            activation[l_name] = output.clone().detach()
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

    return embeddings.clone().detach()


def vis_graph(model, feat, edge_index, edge_weight, labels, attach_idx, num_classes, vis_node_idx=None):
    # Visualizing
    model.eval()
    vis_labels = copy.deepcopy(labels)
    num_nodes = vis_labels.shape[0]
    embedding = model.encode(feat, edge_index, edge_weight)
    clipped_embedding = embedding[:num_nodes].detach().cpu().numpy()
    vis_labels[attach_idx] = num_classes
    vis_labels = vis_labels.detach().cpu().numpy()

    if vis_node_idx is not None:
        vis_node_idx = vis_node_idx.detach().cpu().numpy()
        clipped_embedding = clipped_embedding[vis_node_idx]
        vis_labels = vis_labels[vis_node_idx]

    plot_scatter(clipped_embedding, vis_labels, num_classes + 1)


def vis_feat_importance(feat_importance, labels, attach_idx, num_classes, vis_node_idx=None, dataset_name='Cora', attack_name='GCBA'):
    # Visualizing
    vis_labels = copy.deepcopy(labels)
    num_nodes = vis_labels.shape[0]
    clipped_embedding = feat_importance[:num_nodes].detach().cpu().numpy()
    vis_labels[attach_idx] = num_classes
    vis_labels = vis_labels.detach().cpu().numpy()

    if vis_node_idx is not None:
        vis_node_idx = vis_node_idx.detach().cpu().numpy()
        clipped_embedding = clipped_embedding[vis_node_idx]
        vis_labels = vis_labels[vis_node_idx]

    plot_scatter(clipped_embedding, vis_labels, num_classes + 1)


def plot_scatter(embeds, labels, num_classes, dataset_name='Cora', attack_name='GCBA'):
    tsne_model = TSNE(n_components=2, perplexity=30)
    plot_points = tsne_model.fit_transform(embeds)

    matplotlib.use('Agg')
    fig, ax = plt.subplots()
    all_colors = ['#7fc97f', '#beaed4', '#fdc086', '#ffff99', '#386cb0', '#f0027f', '#bf5b16', '#000000']
    for label in range(0, num_classes):
        points = plot_points[labels == label]
        logger.info('Label = {}@Number of Samples = {}'.format(label + 1, len(points)))
        ax.scatter(
            points[:, 0], points[:, 1],
            s=30, c=all_colors[label % len(all_colors)], label='{}'.format(label + 1), alpha=1.0, edgecolors='face'
        )
    ax.legend(loc='upper left', ncol=4, labelspacing=0.6, prop={'size': 8})
    plt.savefig('{}_Feat_{}.svg'.format(dataset_name, attack_name))


@torch.no_grad()
def model_test(model, features, edge_index, edge_weight, labels, idx_test):
    output = model(features, edge_index, edge_weight)
    acc_test = accuracy(output[idx_test], labels[idx_test])
    return acc_test


def attribute_importance(model, feat, edge_index, edge_weight, labels) -> Tuple[torch.Tensor, torch.Tensor]:

    model.eval()
    device = feat.device

    n_feat = feat.clone().detach().requires_grad_()
    poisoned_logits = model(n_feat, edge_index, edge_weight)

    feat_import = torch.zeros(size=feat.shape, dtype=torch.float32, device=device)
    feat_bin_import = feat_import.clone()
    if feat.shape[0] < 20000:
        # Too slow if the graph is too large
        list_node_idx = np.arange(labels.shape[0], dtype=np.int32).tolist()
        for node_idx in list_node_idx:
            node_logits = poisoned_logits[node_idx]
            feat_grad = autograd.grad(outputs=node_logits[labels[node_idx]], inputs=n_feat, retain_graph=True)[0]
            node_feat_grad = feat_grad[node_idx]
            feat_import[node_idx] = node_feat_grad.clone()
            feat_bin_import[node_idx] = torch.sign(torch.clamp_min(node_feat_grad, min=0.)).clone()
    else:
        # speed up if the graph is too large
        selected_logits = torch.gather(poisoned_logits, 1, labels.unsqueeze(1)).mean()
        feat_import = autograd.grad(outputs=selected_logits, inputs=n_feat, retain_graph=False)[0]
        feat_bin_import = torch.sign(torch.clamp_min(feat_import, min=0.)).clone()

    return feat_import, feat_bin_import


def obtain_sample_inter_relationship(refer_model: nn.Module, feat, poisoned_feat, edge_index,
                                     labels, c_id, train_idx, pos_idx, neg_idx, epochs, lr, weight_decay):
    device = feat.device

    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)
        elif isinstance(m, geo_nn.MessagePassing):
            m.reset_parameters()
        elif isinstance(m, nn.LayerNorm):
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(0.0)

    model = copy.deepcopy(refer_model)
    model.apply(init_weights)

    # Loss function and optimizers
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # K-hop subgraph
    _, pos_reserved_edge_index, _, _ = k_hop_subgraph(train_idx, num_hops=2, edge_index=edge_index, num_nodes=feat.shape[0], relabel_nodes=False)
    pos_reserved_edge_weight = torch.ones(size=(pos_reserved_edge_index.shape[1],), dtype=torch.float32, device=device)
    _, neg_reserved_edge_index, _, _ = k_hop_subgraph(neg_idx, num_hops=1, edge_index=edge_index, num_nodes=feat.shape[0], relabel_nodes=False)
    neg_reserved_edge_weight = torch.ones(size=(neg_reserved_edge_index.shape[1],), dtype=torch.float32, device=device)

    model.train()
    best_state_dict = copy.deepcopy(model.state_dict())
    for idx in range(1, epochs + 1):
        optimizer.zero_grad()

        pos_logits = model(feat, pos_reserved_edge_index, pos_reserved_edge_weight)
        neg_logits = model(poisoned_feat, neg_reserved_edge_index, neg_reserved_edge_weight)

        loss_pos, loss_neg = torch.tensor(0, dtype=torch.float32, device=device), \
            torch.tensor(0, dtype=torch.float32, device=device)
        loss_clf_pos, loss_clf_neg = loss_pos, loss_neg
        if len(train_idx) > 0:
            loss_clf_pos = loss_fn(pos_logits[train_idx], labels[train_idx])
            loss_pos = torch.mean(torch.where(loss_clf_pos < 5, torch.exp(loss_clf_pos) - 1., loss_clf_pos))
        if len(neg_idx) > 0:
            loss_clf_neg = loss_fn(neg_logits[neg_idx], labels[neg_idx]) + 1.0
            loss_neg = torch.mean(torch.where(loss_clf_neg < 1e27, torch.log(loss_clf_neg), loss_clf_neg))
        loss = loss_pos - loss_neg

        if not torch.isnan(loss).any():
            best_state_dict = copy.deepcopy(model.state_dict())
        else:
            model.load_state_dict(best_state_dict)
            break
        if idx == 1 or idx % 50 == 0:
            logger.info(
                'Inter relationship@Label = {}@Idx = {}@Clf Loss = {:.4f}@Clf Loss [Pos] = {:.4f}@Clf Loss [Neg] = {:.4f}'.format(
                    c_id, idx, loss.item(), loss_clf_pos.mean().item(), loss_clf_neg.mean().item()
                )
            )

        loss.backward()
        optimizer.step()

    pos_embedding = obtain_mid_embeddings(model, feat, pos_reserved_edge_index)[pos_idx]
    pos_embedding = torch.mean(pos_embedding, dim=0, keepdim=True)
    neg_embedding = obtain_mid_embeddings(model, poisoned_feat, neg_reserved_edge_index)[neg_idx]
    mean2neg_dists = calc_euc_dis(pos_embedding, neg_embedding)

    logger.info('Label ID = {}@Distance Between Pos And Neg Nodes = [{}]'.format(
        c_id, ', '.join(['{:.4f}'.format(v) for v in mean2neg_dists[0].cpu().tolist()])
    ))
    return mean2neg_dists[0].cpu().numpy()


@torch.no_grad()
def obtain_predict_confidence(model, feat, edge_index, edge_weight, labels):
    model.eval()
    poisoned_logits = model(feat, edge_index, edge_weight)
    poisoned_conf = F.softmax(poisoned_logits, dim=-1)
    labeled_num_samples = labels.shape[0]
    conf = torch.gather(poisoned_conf[:labeled_num_samples], dim=1, index=labels.reshape(-1, 1)).reshape(-1)
    return conf


def mad_based_outlier(points, thresh=3.5):
    if type(points) is list:
        points = np.asarray(points)
    med = np.median(points)
    abs_dev = np.absolute(points - med)
    med_abs_dev = np.median(abs_dev)

    mod_z_score = norm.ppf(0.75) * abs_dev / (med_abs_dev + 1e-6)
    return mod_z_score > thresh


def dshield(poisoned_model, feat, edge_index, edge_weight, labels,
            train_node_idx, attach_idx, val_idx, hidden_dim, lr, weight_decay, kappa1, kappa2, kappa3, thresh,
            edge_drop_ratio, feature_drop_ratio, tau, pretrain_epochs, classify_epochs, finetune_epochs, device,
            balance_factor, classify_rounds, dataset, arch, attack_name, rtn_node_idx=False):
    input_dim = feat.shape[1]
    num_classes = torch.max(labels).item() + 1

    if edge_weight is None:
        edge_weight = torch.ones([edge_index.shape[1]], device=device, dtype=torch.float32)

    # Augment operators
    aug1 = Compose([
        FeatureMasking(pf=feature_drop_ratio),
        EdgeRemoving(pe=edge_drop_ratio),
    ])
    aug2 = Compose([
        FeatureMasking(pf=feature_drop_ratio),
        EdgeRemoving(pe=edge_drop_ratio),
    ])

    # Get attribute importance
    _, attribute_import = attribute_importance(poisoned_model, feat, edge_index, edge_weight, labels)

    # Get edge similarities
    edge_sims = calc_edge_sims(poisoned_model, feat, edge_index)

    # Initialize pretrain model
    pretrain_model = UnsupervisedModel(input_dim=input_dim,  hidden_dim=hidden_dim,
                                       proj_dim=hidden_dim // 2, num_classes=num_classes, arch=arch).to(device)

    # SSL Training
    optimizer = optim.Adam(chain(pretrain_model.rtn_encoder_param(), pretrain_model.rtn_contra_param()), lr=lr, weight_decay=weight_decay)
    pretrain_model = pretrain(pretrain_model, optimizer,
                              (aug1, aug2), feat, edge_index, edge_weight,
                              train_node_idx, attribute_import, edge_sims, feature_drop_ratio, kappa1, kappa2, tau, pretrain_epochs, dataset)

    # Pick up the consistency of similarities among nodes with the same label
    poisoned_embeddings = obtain_mid_embeddings(poisoned_model, feat, edge_index)
    train_labels = labels[train_node_idx]

    # Conduct classify
    classify_model = copy.deepcopy(poisoned_model)
    optimizer = optim.Adam(classify_model.parameters(), lr=lr, weight_decay=weight_decay)

    list_attach_node_idx = attach_idx.cpu().tolist()
    logger.info('Malicious Nodes = [{}]'.format(', '.join('{}'.format(n) for n in list_attach_node_idx)))

    n_round = 0
    while n_round < classify_rounds:
        n_round += 1

        pretrain_model.eval()
        with torch.no_grad():
            ssl_embeddings = pretrain_model.encode(feat, edge_index, edge_weight)

        # Check each class
        poisoned_import_attribute = attribute_import * feat
        class2list_pos_nodes, class2list_neg_nodes, class2sample_relation = {}, {}, {}
        dist_diff_gain_list, feat_diff_gain_list = [], []
        for c_id in range(num_classes):
            # Pick up nodes with `c_id` class
            cls_node_idx = train_node_idx[train_labels == c_id]
            num_cls_nodes = cls_node_idx.shape[0]
            if num_cls_nodes < 4:
                dist_diff_gain_list.append(0.0)
                feat_diff_gain_list.append(0.0)
                class2list_pos_nodes[c_id] = cls_node_idx.cpu().tolist()
                class2list_neg_nodes[c_id] = []
                continue

            # Pick up poisoned and ssl embeddings
            train_ssl_embeddings = ssl_embeddings[cls_node_idx]
            train_poisoned_embeddings = poisoned_embeddings[cls_node_idx]
            indices_matrix = torch.eye(n=train_ssl_embeddings.shape[0], dtype=torch.float32, device=device)

            # Clustering on distance of embeddings (on poison-label backdoor attack)
            train_ssl_dists = calc_euc_dis(train_ssl_embeddings, train_ssl_embeddings)
            train_poisoned_dists = calc_euc_dis(train_poisoned_embeddings, train_poisoned_embeddings)
            dist_diff_matrix = torch.max(train_ssl_dists, train_poisoned_dists)
            dist_diff_info_gain = torch.std(dist_diff_matrix)
            dist_diff_matrix = (dist_diff_matrix - torch.min(dist_diff_matrix)) / (torch.max(dist_diff_matrix) - torch.min(dist_diff_matrix))
            dist_diff_matrix = (1 - indices_matrix) * dist_diff_matrix

            # Clustering on similarities of importance features (on clean-label backdoor attack)
            embed_dim = train_ssl_embeddings.shape[1]
            if embed_dim >= len(cls_node_idx):
                embed_dim = len(cls_node_idx) - 2
            decomposition_model = UMAP(n_components=embed_dim, n_neighbors=num_cls_nodes // 2 + 1)

            train_import_feat = decomposition_model.fit_transform(poisoned_import_attribute[cls_node_idx].detach().cpu().numpy())
            train_import_feat = torch.tensor(train_import_feat).to(device)
            feat_diff_matrix = calc_euc_dis(train_import_feat, train_import_feat)
            feat_diff_info_gain = torch.std(feat_diff_matrix)
            feat_diff_matrix = (feat_diff_matrix - torch.min(feat_diff_matrix)) / (torch.max(feat_diff_matrix) - torch.min(feat_diff_matrix))
            feat_diff_matrix = (1 - indices_matrix) * feat_diff_matrix

            dist_diff_gain_list.append(dist_diff_info_gain.item())
            feat_diff_gain_list.append(feat_diff_info_gain.item())

            dist_matrix = balance_factor * dist_diff_info_gain / (dist_diff_info_gain + feat_diff_info_gain) * dist_diff_matrix + \
                (1 - balance_factor) * feat_diff_info_gain / (dist_diff_info_gain + feat_diff_info_gain) * feat_diff_matrix

            # HDBSCAN for clustering two clusters
            emb_clusterer = HDBSCAN(
                min_cluster_size=num_cls_nodes // 2 + 1, min_samples=1, allow_single_cluster=True, metric='precomputed', n_jobs=-1
            )
            np_dist_matrix = dist_matrix.cpu().numpy()
            np_dist_matrix = (np_dist_matrix + np_dist_matrix.T) / 2
            clu_labels = emb_clusterer.fit_predict(np_dist_matrix) + 1
            clu_labels = torch.tensor(clu_labels, dtype=torch.int64, device=device)

            clu2cls_node_idx = {}
            num_clu_labels = torch.max(clu_labels).item() + 1
            for clu_id in range(num_clu_labels):
                cls_clu_node_idx = cls_node_idx[clu_labels == clu_id]
                clu2cls_node_idx[clu_id] = cls_clu_node_idx.cpu().tolist()

            for clu_id in range(num_clu_labels):
                logger.info('Label = {}@Cluster ID = {}@Clu Node Idx = [{}]'.format(
                    c_id, clu_id, ', '.join('{}'.format(n) for n in clu2cls_node_idx[clu_id])
                ))

                if c_id not in class2list_pos_nodes:
                    class2list_pos_nodes[c_id] = []
                if c_id not in class2list_neg_nodes:
                    class2list_neg_nodes[c_id] = []

                if clu_id != 0:
                    class2list_pos_nodes[c_id].extend(clu2cls_node_idx[clu_id])
                    continue
                class2list_neg_nodes[c_id].extend(clu2cls_node_idx[clu_id])

        list_candidate_neg_nodes, list_malicious_dist = [], []
        for c_id in range(num_classes):
            all_pos_nodes = []
            for i in range(num_classes):
                all_pos_nodes.extend(class2list_pos_nodes[i])
                if i != c_id:
                    all_pos_nodes.extend(class2list_neg_nodes[i])

            if len(class2list_neg_nodes[c_id]) == 0:
                continue

            # Calculate the distance between positive and negative nodes
            poisoned_feat = feat
            if dist_diff_gain_list[c_id] < feat_diff_gain_list[c_id]:
                poisoned_feat = attribute_import * feat
            inter_dist = obtain_sample_inter_relationship(
                poisoned_model, feat, poisoned_feat, edge_index, labels, c_id, all_pos_nodes, class2list_pos_nodes[c_id],
                class2list_neg_nodes[c_id], epochs=finetune_epochs, lr=lr, weight_decay=weight_decay
            )

            list_candidate_neg_nodes.extend(class2list_neg_nodes[c_id])
            list_malicious_dist.extend(inter_dist.tolist())

        list_pos_nodes, list_neg_nodes = [], []
        malicious_dist = np.array(list_malicious_dist)
        malicious_node_mask = mad_based_outlier(malicious_dist, thresh=thresh)
        malicious_nodes = np.array(list_candidate_neg_nodes)[malicious_node_mask]

        for c_id in range(num_classes):
            cur_list_pos_nodes = class2list_pos_nodes[c_id]
            cur_list_pos_nodes.extend(list(set(class2list_neg_nodes[c_id]) - set(malicious_nodes)))
            cur_list_neg_nodes = list(set(class2list_neg_nodes[c_id]) & set(malicious_nodes))

            logger.info('Label = {}@Number of Nodes = {}@Number of Malicious Nodes = {}@Number of Malicious Nodes = [{}]'.format(
                c_id, len(cur_list_pos_nodes), len(cur_list_neg_nodes), ', '.join([str(i) for i in cur_list_neg_nodes])
            ))
            list_pos_nodes.extend(cur_list_pos_nodes)
            list_neg_nodes.extend(cur_list_neg_nodes)

        pos_node_idx = torch.tensor(list_pos_nodes, dtype=torch.long, device=device)
        neg_node_idx = torch.tensor(list_neg_nodes, dtype=torch.long, device=device)

        logger.info(
            'Number of Total Nodes = {}@Number of Reserved Nodes = {}@Number of Abandoned Nodes = {}@Number of Reserved Poisoned Nodes = {}@Number of Obtained Poisoned Nodes = {}'.format(
                train_node_idx.shape[0],
                pos_node_idx.shape[0],
                neg_node_idx.shape[0],
                len(set(list_pos_nodes) & set(attach_idx.cpu().tolist())),
                len(set(list_neg_nodes) & set(attach_idx.cpu().tolist())))
        )

        train_idx = pos_node_idx
        classify_model = classify(
            classify_model, optimizer,
            feat, edge_index, edge_weight, labels, kappa3, train_idx, neg_node_idx, val_idx, neg_node_idx, classify_epochs
        )

    if rtn_node_idx is True:
        return classify_model, neg_node_idx
    return classify_model
