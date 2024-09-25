import logging

import torch

import numpy as np
from sklearn.linear_model import LassoLars

import torch_geometric.nn as geo_nn
from torch_geometric.utils import k_hop_subgraph

from models.GCN import GCN
from models.metric import accuracy

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)

try:
    if 'logger' not in globals():
        logging.basicConfig()
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
except NameError:
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)


class GraphLIME:

    def __init__(self, model, hop=2, rho=0.1, cached=True):
        self.hop = hop
        self.rho = rho
        self.model = model
        self.cached = cached
        self.cached_result = None

        self.model.eval()

    def __flow__(self):
        for module in self.model.modules():
            if isinstance(module, geo_nn.MessagePassing):
                return module.flow

        return 'source_to_target'

    def __subgraph__(self, node_idx, x, y, edge_index, **kwargs):
        num_nodes, num_edges = x.size(0), edge_index.size(1)

        subset, edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx, self.hop, edge_index, relabel_nodes=True, num_nodes=num_nodes, flow=self.__flow__()
        )

        if len(subset) > 500:
            subset = subset[:500]
            edge_mask = edge_mask[:500]
            edge_index = edge_index[:, :500]

        x = x[subset]
        y = y[subset]

        for key, item in kwargs.items():
            if torch.is_tensor(item) and item.size(0) == num_nodes:
                item = item[subset]
            elif torch.is_tensor(item) and item.size(0) == num_edges:
                item = item[edge_mask]
            kwargs[key] = item

        return x, y, edge_index, mapping, edge_mask, kwargs

    def __init_predict__(self, x, edge_index, **kwargs):
        if self.cached and self.cached_result is not None:
            if x.size(0) != self.cached_result.size(0):
                raise RuntimeError('Cached {} number of nodes, but found {}.'.format(x.size(0), self.cached_result.size(0)))

        # get the initial prediction
        if not self.cached or self.cached_result is None:
            with torch.no_grad():
                logits = self.model(x, edge_index, **kwargs)
                probs = torch.softmax(logits, dim=-1)

            self.cached_result = probs

        return self.cached_result

    @staticmethod
    def __compute_kernel__(x, reduce):
        assert x.ndim == 2, x.shape

        n, d = x.shape

        dist = x.reshape(1, n, d) - x.reshape(n, 1, d)  # (n, n, d)
        dist = dist ** 2

        if reduce:
            dist = np.sum(dist, axis=-1, keepdims=True)  # (n, n, 1)

        std = np.sqrt(d)

        K = np.exp(-dist / (2 * std ** 2 * 0.1 + 1e-6))  # (n, n, 1) or (n, n, d)

        return K

    @staticmethod
    def __compute_gram_matrix__(x):

        # more stable and accurate implementation
        G = x - np.mean(x, axis=0, keepdims=True)
        G = G - np.mean(G, axis=1, keepdims=True)

        G = G / (np.linalg.norm(G, ord='fro', axis=(0, 1), keepdims=True) + 1e-6)

        return G

    def explain_node(self, node_idx, x, edge_index, **kwargs):
        probs = self.__init_predict__(x, edge_index, **kwargs)

        x, probs, _, _, _, _ = self.__subgraph__(
            node_idx, x, probs, edge_index, **kwargs
        )

        x = x.detach().cpu().numpy()  # (n, d)
        y = probs.detach().cpu().numpy()  # (n, classes)

        n, d = x.shape

        K = self.__compute_kernel__(x, reduce=False)  # (n, n, d)
        L = self.__compute_kernel__(y, reduce=True)  # (n, n, 1)

        K_bar = self.__compute_gram_matrix__(K)  # (n, n, d)
        L_bar = self.__compute_gram_matrix__(L)  # (n, n, 1)

        K_bar = K_bar.reshape(n ** 2, d)  # (n ** 2, d)
        L_bar = L_bar.reshape(n ** 2, )  # (n ** 2,)

        solver = LassoLars(self.rho, fit_intercept=False, positive=True, eps=1e-6)
        solver.fit(K_bar * n, L_bar * n)
        return solver.coef_


class ExplainBackdoor:
    """ Explainability-based Backdoor Attacks Against Graph Neural Networks
    """

    def __init__(self, trig_feat_val, trig_feat_wid, hidden, epochs, target_class, device):

        self.device = device
        self.shadow_model = None
        self.explainer = None
        self.hidden = hidden
        self.epochs = epochs
        self.target_class = target_class
        self.trig_feat_val = trig_feat_val
        self.trig_feat_wid = trig_feat_wid

    def fit(self, features, edge_index, edge_weight, labels, train_idx, attach_idx, unlabeled_idx):

        if edge_weight is None:
            edge_weight = torch.ones([edge_index.shape[1]], device=self.device, dtype=torch.float)

        # Initialize the shadow model
        self.shadow_model = GCN(
            n_feat=features.shape[1], n_hid=self.hidden,
            n_class=labels.max().item() + 1, dropout=0.0, device=self.device
        ).to(self.device)

        # Train models
        self.shadow_model.train()
        self.shadow_model.fit(features, edge_index, edge_weight, labels, train_idx, train_iters=self.epochs)
        with torch.no_grad():
            self.shadow_model.eval()
            output = self.shadow_model(features, edge_index, edge_weight)
            final_test_acc = accuracy(output[train_idx], labels[train_idx])
        logger.info('Shadow Model Accuracy = {:.2f}'.format(final_test_acc))

        # Initialize GNNExplainer
        self.explainer = GraphLIME(self.shadow_model, hop=2, rho=0.1, cached=True)

    def get_poisoned(self, features, edge_index, labels, attach_idx):

        poison_labels = labels.clone()
        poison_x, poison_edge_index = features.clone(), edge_index.clone()
        poison_labels[attach_idx] = self.target_class

        coefs = []
        self.explainer = GraphLIME(self.shadow_model, hop=2, rho=0.1, cached=True)
        for n_id in attach_idx:
            coef = self.explainer.explain_node(n_id.item(), features, edge_index)
            sorted_coef = coef.argsort()
            coefs.append(sorted_coef)

        for idx, n_id in enumerate(attach_idx):
            poison_x[n_id, coefs[idx][:self.trig_feat_wid]] = self.trig_feat_val

        poison_edge_weights = torch.ones([poison_edge_index.shape[1]], dtype=torch.float, device=self.device)

        return poison_x, poison_edge_index, poison_edge_weights, poison_labels

    def inject_trigger(self, attach_idx, features, edge_index, edge_weight):
        coefs = []
        features, edge_index, edge_weight = features.clone(), edge_index.clone(), edge_weight.clone()
        self.explainer = GraphLIME(self.shadow_model, hop=2, rho=0.1, cached=True)
        for n_id in attach_idx:
            coef = self.explainer.explain_node(n_id.item(), features, edge_index)
            sorted_coef = coef.argsort()
            coefs.append(sorted_coef)

        for idx, n_id in enumerate(attach_idx):
            features[n_id, coefs[idx][:self.trig_feat_wid]] = self.trig_feat_val

        return features, edge_index, edge_weight
