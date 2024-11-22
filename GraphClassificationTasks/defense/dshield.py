import copy
import logging
from abc import ABC, abstractmethod
from typing import Tuple, NamedTuple, List

import numpy as np
from scipy.stats import norm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torch_geometric.nn as geo_nn
from torch import autograd
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import unbatch
from torch_geometric.loader import DataLoader

from models.utils import model_test

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


class Graph(NamedTuple):
    x: torch.FloatTensor
    edge_index: torch.LongTensor

    def unfold(self):
        return self.x, self.edge_index


class Augmentor(ABC):
    """Base class for graph augmentors."""

    def __init__(self):
        pass

    @abstractmethod
    def augment(self, g: Graph) -> Graph:
        raise NotImplementedError(f"GraphAug.augment should be implemented.")

    def __call__(self, x: torch.FloatTensor, edge_index: torch.LongTensor):
        return self.augment(Graph(x, edge_index)).unfold()


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
        x, edge_index = g.unfold()
        edge_index, _ = dropout_edge(edge_index, p=self.pe)
        return Graph(x=x, edge_index=edge_index)


class FeatureMasking(Augmentor):
    def __init__(self, pf: float):
        super(FeatureMasking, self).__init__()
        self.pf = pf

    def augment(self, g: Graph) -> Graph:
        x, edge_index = g.unfold()
        x = drop_feature(x, p=self.pf)
        return Graph(x=x, edge_index=edge_index)


class UnsupervisedModel(nn.Module):
    """ Unsupervised Model for contrastive learning
    """

    def __init__(self, input_dim, proj_dim, hidden_dim, arch):
        super(UnsupervisedModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.proj_dim = proj_dim
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
        # Fully-connected layer
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def rtn_encoder_param(self):
        return self.encoder.parameters()

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        z = x
        for layer in self.encoder:
            if isinstance(layer, geo_nn.GATConv) or isinstance(layer, geo_nn.GCNConv):
                z = layer(z, edge_index, edge_weight)
            elif isinstance(layer, geo_nn.SAGEConv):
                z = layer(z, edge_index)
            else:
                z = layer(z)

        # Readout
        output = global_mean_pool(z, batch)

        return self.fc(output)


def dropout_edge(edge_index: torch.LongTensor,
                 p: float = 0.5, force_undirected: bool = False, training: bool = True) -> Tuple[torch.LongTensor, torch.LongTensor]:

    if p < 0. or p > 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 (got {p}')

    if not training or p == 0.0:
        edge_mask = edge_index.new_ones(edge_index.size(1), dtype=torch.bool)
        return edge_index, edge_mask

    # Generate candidate edges according to probabilities
    row, col = edge_index
    device = edge_index.device
    list_candidates = np.arange(edge_index.shape[1], dtype=np.int32)
    list_chosen_candidates = np.random.choice(list_candidates, int(p * edge_index.shape[1]))
    chosen_candidates = torch.tensor(list_chosen_candidates, dtype=torch.long, device=device)

    edge_mask = torch.tensor([True] * edge_index.shape[1], dtype=torch.bool, device=device)
    edge_mask[chosen_candidates] = False

    if force_undirected:
        edge_mask[row > col] = False
    edge_index = edge_index[:, edge_mask]

    if force_undirected:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_mask = edge_mask.nonzero().repeat((2, 1)).squeeze()

    return edge_index, edge_mask


def drop_feature(feat: torch.FloatTensor, p: float) -> torch.FloatTensor:
    device = feat.device

    mask = torch.empty(size=feat.shape, dtype=torch.float32).uniform_(0, 1) <= p
    mask = mask.to(device)

    feat = feat.clone()
    feat[mask] = 0
    return feat


def attribute_importance(model, datasets, batch_size, device):
    model.eval()

    feat_import_list, feat_bin_import_list = [], []
    loader = DataLoader(datasets, batch_size=batch_size, shuffle=False, drop_last=False)
    for _, data in enumerate(loader):
        data = data.to(device)

        feat = data.x.requires_grad_()
        poisoned_logits = model(feat, data.edge_index, batch=data.batch)
        selected_logits = torch.gather(poisoned_logits, dim=1, index=data.y.view(-1, 1)).mean()
        batch_feat_import = autograd.grad(outputs=selected_logits, inputs=feat, retain_graph=False)[0]
        batch_feat_bin_import = torch.sign(torch.clamp_min(batch_feat_import, min=0.)).clone()

        batch_feat_import = unbatch(batch_feat_import, data.batch)
        batch_feat_bin_import = unbatch(batch_feat_bin_import, data.batch)
        feat_import_list.extend(batch_feat_import)
        feat_bin_import_list.extend(batch_feat_bin_import)
    return feat_import_list, feat_bin_import_list


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


def infonce_loss(anchor, sample, pos_mask, neg_mask, tau):
    # InfoNCE Loss
    sim = calc_similarity(anchor, sample) / tau
    exp_sim = torch.exp(sim) * (pos_mask + neg_mask)
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
    loss = log_prob * pos_mask
    loss = loss.sum(dim=1) / pos_mask.sum(dim=1)
    return - loss.mean()


def pretrain(model: UnsupervisedModel, optimizer, augmentor,
             datasets, batch_size, tau=0.5, epochs=100, device='cuda'):

    # Pretraining without any pseudo-label information
    model.train()
    train_loader = DataLoader(datasets, batch_size=batch_size, shuffle=True, drop_last=False)

    aug1, aug2 = augmentor
    for epoch in range(1, epochs + 1):
        mean_loss = 0.
        for idx, data in enumerate(train_loader):
            optimizer.zero_grad()

            data = data.to(device)

            # Sample positive and negative samples according to labels
            feat, edge_index = aug1(data.x, data.edge_index)
            pos_graph = Graph(x=feat, edge_index=edge_index)
            feat, edge_index = aug2(data.x, data.edge_index)
            neg_graph = Graph(x=feat, edge_index=edge_index)

            # Obtain embeddings
            embedding = model(data.x, data.edge_index, batch=data.batch)
            pos_embedding = model(pos_graph.x, pos_graph.edge_index, batch=data.batch)
            neg_embedding = model(neg_graph.x, neg_graph.edge_index, batch=data.batch)

            batch_size = len(data)
            pos_mask = torch.eye(batch_size, dtype=torch.float32, device=device)
            neg_mask = 1. - pos_mask

            # InfoNCE loss
            loss1 = infonce_loss(embedding, pos_embedding, pos_mask, neg_mask, tau)
            loss2 = infonce_loss(embedding, neg_embedding, pos_mask, neg_mask, tau)
            loss_c = 0.5 * loss1 + 0.5 * loss2

            loss = loss_c

            loss.backward()
            optimizer.step()
            mean_loss += loss.item()
        mean_loss /= len(train_loader)

        if epoch == 1 or epoch % 50 == 0:
            logger.info('Pretrain@Epoch = {}@Loss = {:.4f}'.format(
                epoch, mean_loss
            ))

    return model


@torch.no_grad()
def obtain_mid_embeddings(model, datasets, batch_size, device):
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

    all_embeddings = None
    train_loader = DataLoader(datasets, batch_size=batch_size, shuffle=False, drop_last=False)
    for idx, data in enumerate(train_loader):
        data = data.to(device)
        _ = model(data.x, data.edge_index, batch=data.batch)
        batch_embeddings = global_mean_pool(activation[layer_name], data.batch)
        all_embeddings = torch.cat([all_embeddings, batch_embeddings], dim=0) if all_embeddings is not None else batch_embeddings

    if handle is not None:
        handle.remove()

    return all_embeddings.clone().detach()


@torch.no_grad()
def obtain_ssl_embeddings(model, datasets, batch_size, device):
    model.eval()
    all_embeddings = None
    train_loader = DataLoader(datasets, batch_size=batch_size, shuffle=False, drop_last=False)
    for idx, data in enumerate(train_loader):
        data = data.to(device)
        batch_embeddings = model(data.x, data.edge_index, batch=data.batch)
        all_embeddings = torch.cat([all_embeddings, batch_embeddings], dim=0) if all_embeddings is not None else batch_embeddings
    return all_embeddings.clone().detach()


def calc_euc_dis(h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
    hh11 = (h1 * h1).sum(-1).reshape(-1, 1).repeat(1, h2.shape[0])
    hh22 = (h2 * h2).sum(-1).reshape(1, -1).repeat(h1.shape[0], 1)
    hh11_hh22 = hh11 + hh22
    hh12 = h1 @ h2.T
    distance = hh11_hh22 - 2 * hh12
    return distance


def obtain_sample_inter_relationship(refer_model: nn.Module, dataset, c_id, train_idx,
                                     pos_idx, neg_idx, epochs, batch_size, lr, weight_decay, device):

    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
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

    normal_train_loader = DataLoader([dataset[i] for i in train_idx], batch_size=batch_size, shuffle=True)
    poisoned_train_loader = DataLoader([dataset[i] for i in neg_idx], batch_size=batch_size, shuffle=True)

    model.train()
    best_state_dict = copy.deepcopy(model.state_dict())
    round_iter = 0
    for idx in range(1, epochs + 1):
        for _, data in enumerate(normal_train_loader):
            round_iter += 1
            data = data.to(device)
            poisoned_data = next(iter(poisoned_train_loader)).to(device)

            optimizer.zero_grad()

            pos_logits = model(data.x, data.edge_index, batch=data.batch)
            neg_logits = model(poisoned_data.x, poisoned_data.edge_index, batch=poisoned_data.batch)

            loss_pos, loss_neg = torch.tensor(0, dtype=torch.float32, device=device), \
                torch.tensor(0, dtype=torch.float32, device=device)
            loss_clf_pos, loss_clf_neg = loss_pos, loss_neg
            if len(train_idx) > 0:
                loss_clf_pos = loss_fn(pos_logits, data.y)
                loss_pos = torch.mean(torch.where(loss_clf_pos < 5, torch.exp(loss_clf_pos) - 1., loss_clf_pos))
            if len(neg_idx) > 0:
                loss_clf_neg = loss_fn(neg_logits, poisoned_data.y) + 1.0
                loss_neg = torch.mean(torch.where(loss_clf_neg < 1e27, torch.log(loss_clf_neg), loss_clf_neg))
            loss = loss_pos - loss_neg

            if not torch.isnan(loss).any():
                best_state_dict = copy.deepcopy(model.state_dict())
            else:
                model.load_state_dict(best_state_dict)
                break
            if round_iter == 1 or round_iter % 500 == 0:
                logger.info(
                    'Inter relationship@Label = {}@Idx = {}@Clf Loss = {:.4f}@Clf Loss [Pos] = {:.4f}@Clf Loss [Neg] = {:.4f}'.format(
                        c_id, idx, loss.item(), loss_clf_pos.mean().item(), loss_clf_neg.mean().item()
                    )
                )

            loss.backward()
            optimizer.step()

    pos_embedding = obtain_mid_embeddings(model, [dataset[i] for i in pos_idx], batch_size, device)
    pos_embedding = torch.mean(pos_embedding, dim=0, keepdim=True)
    neg_embedding = obtain_mid_embeddings(model, [dataset[i] for i in neg_idx], batch_size, device)
    mean2neg_dists = calc_euc_dis(pos_embedding, neg_embedding)

    logger.info('Label ID = {}@Distance Between Pos And Neg Nodes = [{}]'.format(
        c_id, ', '.join(['{:.4f}'.format(v) for v in mean2neg_dists[0].cpu().tolist()])
    ))
    return mean2neg_dists[0].cpu().numpy()


def mad_based_outlier(points, thresh=3.5):
    if type(points) is list:
        points = np.asarray(points)
    med = np.median(points)
    abs_dev = np.absolute(points - med)
    med_abs_dev = np.median(abs_dev)

    mod_z_score = norm.ppf(0.75) * abs_dev / med_abs_dev
    return mod_z_score > thresh


def classify(model, optimizer, datasets, kappa1,
             pos_node_idx, neg_node_idx, test_loader, epochs=100, batch_size=16, device='cuda'):

    model.initialize()
    model.train()

    clf_loss_fn = nn.CrossEntropyLoss(reduction='mean')

    normal_train_loader = DataLoader([datasets[i] for i in pos_node_idx], batch_size=batch_size, shuffle=True)
    poisoned_train_loader = DataLoader([datasets[i] for i in neg_node_idx], batch_size=batch_size, shuffle=True)

    best_acc_val, best_asr, best_balance_metric, best_round, best_state_dict = 0., 0., 0., 0, copy.deepcopy(model.state_dict())
    for epoch in range(1, epochs + 1):
        model.train()

        mean_loss, mean_loss_pos, mean_loss_neg = 0., 0., 0.
        for _, data in enumerate(normal_train_loader):
            optimizer.zero_grad()
            data = data.to(device)
            poisoned_data = next(iter(poisoned_train_loader)).to(device)

            # Classification
            pos_logits = model(data.x, data.edge_index, batch=data.batch)
            neg_logits = model(poisoned_data.x, poisoned_data.edge_index, batch=poisoned_data.batch)
            loss_pos, loss_neg = torch.tensor(0, dtype=torch.float32, device=device), torch.tensor(0, dtype=torch.float32, device=device)
            loss_clf_pos, loss_clf_neg = loss_pos, loss_neg
            if len(pos_node_idx) > 0:
                loss_clf_pos = clf_loss_fn(pos_logits, data.y)
                loss_pos = torch.where(loss_clf_pos < 5, torch.exp(loss_clf_pos) - 1., loss_clf_pos)
            if len(neg_node_idx) > 0:
                loss_clf_neg = clf_loss_fn(neg_logits, poisoned_data.y) + 1.0
                loss_neg = torch.where(loss_clf_neg < 1e27, torch.log(loss_clf_neg), loss_clf_neg)
            loss = loss_pos - kappa1 * loss_neg

            loss.backward()
            optimizer.step()
            mean_loss, mean_loss_pos, mean_loss_neg = mean_loss + loss.item(), mean_loss_pos + loss_pos.item(), mean_loss_neg + loss_neg.item()

        model.eval()
        _, acc_val = model_test(model, test_loader, clf_loss_fn, device)
        _, asr = model_test(model, poisoned_train_loader, clf_loss_fn, device)
        balance_metric = 2 * (acc_val * (1 - asr)) / (acc_val + (1 - asr) + 1e-6)

        if epoch > int(0.8 * epochs) and best_balance_metric <= balance_metric:
            best_balance_metric = balance_metric
            best_acc_val = acc_val
            best_asr = asr
            best_state_dict = copy.deepcopy(model.state_dict())
            best_round = epoch
        mean_loss, mean_loss_pos, mean_loss_neg = mean_loss / len(normal_train_loader), \
            mean_loss_pos / len(normal_train_loader), mean_loss_neg / len(normal_train_loader)

        if epoch == 1 or epoch % 50 == 0:
            logger.info(
                'Classify@Epoch = {}@Loss = {:.4f}@Loss Clf[P] = {:.4f}@Loss Clf[N] = {:.4f}'.format(
                    epoch, mean_loss, mean_loss_pos, mean_loss_neg
            ))

    logger.info('Best Round = {}@Best ACC = {:.4f}@Best ASR = {:4f}'.format(best_round, best_acc_val, best_asr))
    model.load_state_dict(best_state_dict)
    return model


def dshield(poisoned_model, datasets, attach_idx, feat_dim, hidden_dim, num_classes, target_class,
            test_loader, batch_size, lr, weight_decay, kappa1, thresh, edge_drop_ratio, feature_drop_ratio,
            tau, pretrain_epochs, classify_epochs, finetune_epochs, device, balance_factor, classify_rounds):

    # Augment operators
    aug1 = Compose([
        FeatureMasking(pf=feature_drop_ratio),
        EdgeRemoving(pe=edge_drop_ratio),
    ])
    aug2 = Compose([
        FeatureMasking(pf=feature_drop_ratio),
        EdgeRemoving(pe=edge_drop_ratio),
    ])

    # Initialize pretrain model
    pretrain_model = UnsupervisedModel(input_dim=feat_dim, hidden_dim=hidden_dim,
                                       proj_dim=hidden_dim // 2, arch='GCN').to(device)

    # SSL Training
    optimizer = optim.Adam(pretrain_model.rtn_encoder_param(), lr=lr, weight_decay=weight_decay)
    pretrain_model = pretrain(pretrain_model, optimizer,
                              (aug1, aug2), datasets, batch_size, tau, pretrain_epochs, device)

    # Pick up the consistency of similarities among nodes with the same label
    poisoned_embeddings = obtain_mid_embeddings(poisoned_model, datasets, batch_size, device)

    # Conduct classify
    classify_model = copy.deepcopy(poisoned_model)
    optimizer = optim.Adam(classify_model.parameters(), lr=lr, weight_decay=weight_decay)

    list_attach_node_idx = attach_idx.tolist()
    logger.info('Malicious Nodes = [{}]'.format(', '.join('{}'.format(n) for n in list_attach_node_idx)))

    # Get attribute importance
    _, attribute_import_list = attribute_importance(poisoned_model, datasets, batch_size, device)

    n_round = 0
    while n_round < classify_rounds:
        n_round += 1

        pretrain_model.eval()
        with torch.no_grad():
            ssl_embeddings = obtain_ssl_embeddings(pretrain_model, datasets, batch_size, device)

        # check each class
        poisoned_import_attribute = []
        for idx in range(len(datasets)):
            poisoned_import_attribute.append(
                torch.reshape(attribute_import_list[idx] * datasets[idx].x, shape=(1, -1))[:, :feat_dim]
            )
        poisoned_import_attribute = torch.cat(poisoned_import_attribute, dim=0)

        class2list_pos_nodes, class2list_neg_nodes = {}, {}
        dist_diff_gain_list, feat_diff_gain_list = [], []
        for c_id in range(num_classes):

            # Pick up nodes with `c_id` class
            cls_node_idx = torch.tensor(
                [idx for idx in range(len(datasets)) if datasets[idx].y.item() == c_id],
                dtype=torch.long, device=device
            )
            num_cls_nodes = cls_node_idx.shape[0]
            if num_cls_nodes < 3:
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
            embed_dim = poisoned_import_attribute.shape[1]
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

                metric_val = torch.std(dist_matrix[clu_labels == clu_id, :][:, clu_labels == clu_id]).item()
                if metric_val < 1e-2 and clu_id > 0:
                    clu2cls_node_idx[0].extend(cls_clu_node_idx.cpu().tolist())

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
            inter_dist = obtain_sample_inter_relationship(
                poisoned_model, datasets, c_id, all_pos_nodes, class2list_pos_nodes[c_id],
                class2list_neg_nodes[c_id], epochs=finetune_epochs, batch_size=batch_size, lr=lr, weight_decay=weight_decay, device=device
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

        train_idx = list_pos_nodes
        neg_node_idx = list_neg_nodes

        logger.info(
            'Number of Reserved Nodes = {}@Number of Abandoned Nodes = {}@Number of Reserved Poisoned Nodes = {}@Number of Obtained Poisoned Nodes = {}'.format(
                len(list_pos_nodes),
                len(list_neg_nodes),
                len(set(list_pos_nodes) & set(attach_idx.tolist())),
                len(set(list_neg_nodes) & set(attach_idx.tolist())))
        )

        classify_model = classify(
            classify_model, optimizer, datasets, kappa1, train_idx, neg_node_idx, test_loader, classify_epochs, batch_size, device
        )

    return classify_model
