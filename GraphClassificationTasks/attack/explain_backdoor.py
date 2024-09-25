import logging

import torch
import torch.nn as nn

import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.explain import Explainer, GNNExplainer

from models.GCN import GCN

import warnings
from sklearn.exceptions import ConvergenceWarning

from models.utils import model_test

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


class ExplainBackdoor:
    """ Explainability-based Backdoor Attacks Against Graph Neural Networks
    """

    def __init__(self, batch_size, trigger_size, trig_feat_val, hidden, epochs, target_class, device):

        self.device = device
        self.explainer = None
        self.hidden = hidden
        self.epochs = epochs
        self.batch_size = batch_size
        self.target_class = target_class
        self.trig_feat_val = trig_feat_val
        self.shadow_model = None
        self.trigger_size = trigger_size

    def fit(self, datasets, lr, weight_decay, train_iters, num_features, num_classes):

        # initialize shadow model
        self.shadow_model = GCN(n_feat=num_features,
                                n_hid=self.hidden, n_class=num_classes, dropout=0.5).to(self.device)

        # Train models
        self.shadow_model.train()
        criterion = nn.CrossEntropyLoss()
        train_loader = DataLoader(datasets, batch_size=self.batch_size, shuffle=True, drop_last=False)
        optimizer = optim.Adam(self.shadow_model.parameters(), lr=lr, weight_decay=weight_decay)

        for epoch in range(train_iters):
            for idx, data in enumerate(train_loader):
                data = data.to(self.device)
                optimizer.zero_grad()
                output = self.shadow_model(data.x, data.edge_index, data.edge_weight, data.batch)
                loss = criterion(output, data.y)
                loss.backward()
                optimizer.step()

        _, final_test_acc = model_test(self.shadow_model, datasets, criterion, self.device)
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
                task_level='graph',
                return_type='raw',  # Model returns log probabilities.
            ),
        )

    def get_poisoned(self, dataset, attach_idx):
        for idx in attach_idx:
            explanation = self.explainer(dataset[idx].x, dataset[idx].edge_index)
            node_mask = torch.mean(explanation.node_mask, dim=1, keepdim=False)
            node_mask = node_mask.argsort()[:self.trigger_size]
            dataset[idx].x[node_mask] = self.trig_feat_val
            dataset[idx].y = torch.tensor([self.target_class], device=self.device)
        return dataset

    def inject_trigger(self, dataset, attach_idx):
        return self.get_poisoned(dataset, attach_idx)
