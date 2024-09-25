""" https://github.com/21721677/SemanticBackdoor
"""
import logging
import random
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
from pyexpat import features

from torch_geometric.loader import DataLoader

from models.GCN import GCN
from models.utils import model_test

try:
    if 'logger' not in globals():
        logging.basicConfig()
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
except NameError:
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)


def predict_sample(model, data, device):
    model.eval()

    data = data.to(device)
    out = model(data.x, data.edge_index)
    pred_score = out[0][data.y.item()].item()
    return pred_score


# modify features of the target node to zeros([0, 0, ..., 0])
def modify_features(graph, num_attributes, nodeLabel: int):
    n = graph.x.shape[1]
    new_graph = graph
    new_feature = np.zeros(n)
    for i, feature in enumerate(new_graph.x):
        if feature[num_attributes + nodeLabel] == 1:
            new_graph.x[i, :] = torch.from_numpy(new_feature)
    return new_graph


def has_node(graph, num_attributes, nodeLabel: int):
    sum_array = graph.x.sum(axis=0).cpu().numpy().astype(int)
    return sum_array[num_attributes + nodeLabel] > 0


class SBAG:
    """ Backdoor attacks to graph neural networks.
    """

    def __init__(self, batch_size, trigger_size, hidden, epochs, target_class,
                 trigger_node, num_node_attributes, poisoning_num, t, device):
        self.device = device
        self.batch_size = batch_size
        self.trigger_size = trigger_size
        self.target_class = target_class
        self.hidden, self.epochs = hidden, epochs
        self.trigger_node = trigger_node
        self.num_node_attributes = num_node_attributes
        self.poisoning_num = poisoning_num
        self.t = int(t)
        self.feature_list = []

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

    def get_poisoned(self, dataset, attach_idx):
        attach_idx = attach_idx.tolist()

        candidate_data_list = []
        for idx in range(len(dataset)):
            if idx in attach_idx and has_node(dataset[idx], self.num_node_attributes, self.trigger_node):
                score_one = predict_sample(
                    self.shadow_model, deepcopy(dataset[idx]), self.device)
                new_graph = modify_features(deepcopy(dataset[idx]), self.num_node_attributes, self.trigger_node)
                score_two = predict_sample(self.shadow_model, new_graph, self.device)
                diff = abs(score_one - score_two)
                candidate_data_list.append((idx, diff))

        # sort the array by diff and select top-k samples for poisoning
        candidate_data_list.sort(key=lambda x: x[1], reverse=True)
        poisoning_data_idx = [x[0] for x in candidate_data_list]
        poisoning_num = min(self.poisoning_num, len(candidate_data_list))

        # record all possible features of the target node
        feature_list = []
        for graph in dataset:
            for line in graph.x:
                if line[self.num_node_attributes + self.trigger_node]:
                    feature_list.append(line[:])
        self.feature_list = feature_list

        # relabel top-k samples
        for i in poisoning_data_idx:
            # print(f"original class: {poisoning_data[i].y.item()}")
            dataset[i].y = torch.tensor([self.target_class], dtype=torch.long, device=self.device)
        return dataset

    def inject_trigger(self, dataset, attach_idx):
        if not isinstance(attach_idx, list):
            attach_idx = attach_idx.tolist()
        poisoned_dataset = deepcopy(dataset)
        for i, graph in enumerate(poisoned_dataset):
            if i not in attach_idx:
                continue

            node_num = graph.x.shape[0]

            new_graph = graph
            node_sample = random.sample(range(0, node_num), min(node_num, self.t))
            for node_idx in node_sample:
                new_graph.x[node_idx, :] = self.feature_list[0]
            new_graph.y = torch.tensor([self.target_class], dtype=torch.long, device=self.device)
            poisoned_dataset[i] = new_graph

        return poisoned_dataset
