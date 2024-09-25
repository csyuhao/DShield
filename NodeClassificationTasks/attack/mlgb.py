""" Multi-target label backdoor attacks on graph neural networks
"""
import logging
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch_geometric.explain import Explainer, GNNExplainer

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


class GraphTrojanNet(nn.Module):
    # In the future, we may use a GNN model to generate backdoor
    def __init__(self, device, feat_dim, num_classes, hidden, layer_num=1, dropout=0.0):
        super(GraphTrojanNet, self).__init__()

        layers = []
        for l in range(layer_num - 1):
            if l == 0:
                layers.append(nn.Linear(feat_dim + num_classes, hidden))
            elif l < layer_num - 1:
                layers.append(nn.Linear(hidden, hidden))

            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(hidden, feat_dim))

        self.device = device
        self.num_classes = num_classes
        self.layers = nn.Sequential(*layers).to(device)

    def forward(self, feat, labels):
        one_hot_label = F.one_hot(labels, self.num_classes).float()
        feat = torch.cat([feat, one_hot_label], dim=-1)
        feat = self.layers(feat)
        return feat


class MLGB:
    """ Explainability-based Backdoor Attacks Against Graph Neural Networks
    """

    def __init__(self, seed, trigger_dim, trojan_epochs, inner_epochs,
                 lr, weight_decay, num_classes, hidden, epochs, target_class, device):

        self.seed = seed
        self.device = device
        self.shadow_model = None
        self.explainer = None
        self.hidden = hidden
        self.epochs = epochs
        self.trigger_dim = trigger_dim
        self.target_class = target_class
        self.lr, self.weight_decay = lr, weight_decay
        self.trojan_epochs, self.inner_epochs = trojan_epochs, inner_epochs
        self.trojan = GraphTrojanNet(device, feat_dim=trigger_dim,
                                     num_classes=num_classes, hidden=hidden, layer_num=2, dropout=0.0).to(device)

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
        self.explainer = Explainer(
            model=self.shadow_model,
            algorithm=GNNExplainer(epochs=200),
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=dict(
                mode='multiclass_classification',
                task_level='node',
                return_type='log_probs',
            ),
        )

        poison_labels = labels.clone()
        poison_x, poison_edge_index = features.clone(), edge_index.clone()
        poison_labels[attach_idx] = self.target_class

        explanation = self.explainer(poison_x, poison_edge_index)
        node_feat_mask = explanation.node_mask
        node_feat_import_idx = torch.topk(node_feat_mask, k=self.trigger_dim, dim=-1, largest=False)[1]

        optimizer_shadow = optim.Adam(self.shadow_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        optimizer_trigger = optim.Adam(self.trojan.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        loss_best = 1e8
        best_state_dict = None
        for i in range(self.trojan_epochs):
            self.trojan.eval()
            self.shadow_model.train()
            output, loss_inner = None, None
            for j in range(self.inner_epochs):
                optimizer_shadow.zero_grad()

                attach_node_feat = poison_x[attach_idx]
                effect_feat = torch.gather(attach_node_feat, dim=1, index=node_feat_import_idx[attach_idx])
                trojan_feat = self.trojan(effect_feat, poison_labels[attach_idx])  # may revise the process of generate
                poison_x[attach_idx] = poison_x[attach_idx].scatter_(1, node_feat_import_idx[attach_idx], trojan_feat.detach())
                output = self.shadow_model(poison_x, poison_edge_index)

                loss_inner = F.cross_entropy(output[torch.cat([train_idx, attach_idx])],
                                             poison_labels[torch.cat([train_idx, attach_idx])])  # add our adaptive loss

                loss_inner.backward()
                optimizer_shadow.step()

            acc_train_clean = accuracy(output[train_idx], poison_labels[train_idx])
            acc_train_attach = accuracy(output[attach_idx], poison_labels[attach_idx])

            # involve unlabeled nodes in outer optimization
            self.trojan.train()
            self.shadow_model.eval()
            optimizer_trigger.zero_grad()

            rs = np.random.RandomState(self.seed)
            sampled_unlabeled_idx = unlabeled_idx[rs.choice(len(unlabeled_idx), size=min(512, len(unlabeled_idx)), replace=False)]
            outer_idx = torch.cat([attach_idx, sampled_unlabeled_idx])

            update_feat = features.clone()
            update_edge_index = edge_index.clone()

            attach_node_feat = features[outer_idx]
            effect_feat = torch.gather(attach_node_feat, 1, node_feat_import_idx[outer_idx])
            trojan_feat = self.trojan(effect_feat, poison_labels[outer_idx])  # may revise the process of generate
            update_feat[outer_idx] = update_feat[outer_idx].scatter_(1, node_feat_import_idx[outer_idx], trojan_feat)

            output = self.shadow_model(update_feat, update_edge_index)
            outer_poisoned_labels = poison_labels.clone()
            outer_poisoned_labels[outer_idx] = self.target_class
            loss_target = F.cross_entropy(output[torch.cat([train_idx, outer_idx])],
                                          outer_poisoned_labels[torch.cat([train_idx, outer_idx])])

            loss_target.backward()
            optimizer_trigger.step()
            acc_train_outer = (output[outer_idx].argmax(dim=1) == self.target_class).float().mean()

            if loss_target < loss_best:
                best_state_dict = deepcopy(self.trojan.state_dict())
                loss_best = float(loss_target)

            if i % 10 == 0:
                logger.info('Epoch {}, loss_outer: {:.5f},  loss_inner: {:.5f}, loss_target: {:.5f}, homo loss: {:.5f} '.format(
                    i, loss_target.item(), loss_inner, loss_target, loss_target.item()
                ))
                logger.info("acc_train_clean: {:.4f}, ASR_train_attach: {:.4f}, ASR_train_outer: {:.4f}".format(
                    acc_train_clean, acc_train_attach, acc_train_outer
                ))

        self.trojan.eval()
        self.trojan.load_state_dict(best_state_dict)
        return poison_x, poison_edge_index, poison_labels

    def inject_trigger(self, attach_idx, features, edge_index, edge_weight, poison_labels=None):
        self.trojan.eval()
        features, edge_index, edge_weight = features.clone(), edge_index.clone(), edge_weight.clone()

        if poison_labels is None:
            poison_labels = torch.ones(size=(features.shape[0],), dtype=torch.long, device=self.device) * self.target_class

        explanation = self.explainer(features, edge_index)
        node_feat_mask = explanation.node_mask
        node_feat_import_idx = torch.topk(node_feat_mask, k=self.trigger_dim, dim=-1, largest=False)[1]

        attach_node_feat = features[attach_idx]
        effect_feat = torch.gather(attach_node_feat, 1, node_feat_import_idx[attach_idx])
        with torch.no_grad():
            trojan_feat = self.trojan(effect_feat, poison_labels[attach_idx])  # may revise the process of generate
        features[attach_idx] = features[attach_idx].scatter_(1, node_feat_import_idx[attach_idx], trojan_feat)
        return features, edge_index, edge_weight

    def get_poisoned(self, features, edge_index, edge_weight, labels, attach_idx):

        if edge_weight is None:
            edge_weight = torch.ones([edge_index.shape[1]], device=self.device, dtype=torch.float32)

        features, edge_index, edge_weight = features.clone(), edge_index.clone(), edge_weight.clone()
        poison_labels = labels.clone()
        poison_labels[attach_idx] = self.target_class
        features, edge_index, edge_weight = self.inject_trigger(attach_idx, features, edge_index, edge_weight, poison_labels)
        return features, edge_index, edge_weight, poison_labels
