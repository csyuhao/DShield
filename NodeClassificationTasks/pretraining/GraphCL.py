import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torch_geometric.nn as geo_nn

import GCL.losses as L
import GCL.augmentors as A
from GCL.models import DualBranchContrast
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

from itertools import chain
import numpy as np
import matplotlib

from sklearn.manifold import TSNE


class GConv(nn.Module):

    def __init__(self, input_dim, hidden_dim, activation, num_layers):
        super(GConv, self).__init__()
        self.activation = activation()
        self.layers = nn.ModuleList()
        self.layers.append(geo_nn.GCNConv(input_dim, hidden_dim, cached=False))
        self.num_layers = num_layers
        for _ in range(num_layers - 1):
            self.layers.append(geo_nn.GCNConv(hidden_dim, hidden_dim, cached=False))

    def forward(self, x, edge_index, edge_weight=None):
        z = x
        for i, conv in enumerate(self.layers):
            z = conv(z, edge_index, edge_weight)
            if i != self.num_layers - 1:
                z = self.activation(z)
            else:
                z = F.normalize(z, p=2, dim=-1)
        return z


class Encoder(nn.Module):
    def __init__(self, encoder, augmentor, hidden_dim, proj_dim):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor

        self.fc1 = nn.Linear(hidden_dim, proj_dim)
        self.fc2 = nn.Linear(proj_dim, hidden_dim)

    def forward(self, x, edge_index, edge_weight=None):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)
        z = self.encoder(x, edge_index, edge_weight)
        z1 = self.encoder(x1, edge_index1, edge_weight1)
        z2 = self.encoder(x2, edge_index2, edge_weight2)
        return z, z1, z2

    def project(self, z):
        z = F.elu(self.fc1(z))
        return self.fc2(z)


def train(encoder_model, contrast_model, features, edge_index, edge_weights, optimizer):
    encoder_model.train()
    optimizer.zero_grad()
    z, z1, z2 = encoder_model(features, edge_index, edge_weights)
    h1, h2 = [encoder_model.project(x) for x in [z1, z2]]
    if h1.shape[0] > 5000:
        picked_idx = torch.randperm(h1.shape[0])[:5000]
        h1, h2 = h1[picked_idx], h2[picked_idx]
    loss = contrast_model(h1, h2)
    loss.backward()
    optimizer.step()
    return loss.item()


def dis_fun(x, c, diag=False):
    xx = (x * x).sum(-1).reshape(-1, 1).repeat(1, c.shape[0])
    cc = (c * c).sum(-1).reshape(1, -1).repeat(x.shape[0], 1)
    xx_cc = xx + cc
    xc = x @ c.T
    if diag is True:
        diag_tensor = torch.eye(n=x.shape[0], dtype=torch.float32, device=x.device) * 1e7
        distance = torch.min(xx_cc - 2 * xc + diag_tensor, dim=-1)[0]
    else:
        distance = torch.min(xx_cc - 2 * xc, dim=-1)[0]
    return distance


def finetune(encoder_model, contrast_model, features, edge_index, edge_weights, optimizer, cluster_centers):
    encoder_model.train()
    optimizer.zero_grad()
    z, z1, z2 = encoder_model(features, edge_index, edge_weights)
    h1, h2 = [encoder_model.project(x) for x in [z1, z2]]
    if h1.shape[0] > 5000:
        picked_idx = torch.randperm(h1.shape[0])[:5000]
        h1, h2 = h1[picked_idx], h2[picked_idx]
    disc_loss = contrast_model(h1, h2)

    sample_center_distance = dis_fun(z, cluster_centers)
    clu_loss = sample_center_distance.mean()
    loss = disc_loss + 0.4 * clu_loss

    loss.backward()
    optimizer.step()
    return loss.item()


def pretrain(features, edge_index, labels,
             train_node_idx, attach_idx, effect_idx, train_iters, lr, weight_decay, device, dataset_name, attack_name):

    num_features = features.shape[1]
    num_cluster = torch.max(labels).item() + 1

    # Augment
    aug1 = A.Compose([A.EdgeRemoving(pe=0.2), A.FeatureMasking(pf=0.2)])
    aug2 = A.Compose([A.EdgeRemoving(pe=0.2), A.FeatureMasking(pf=0.2)])
    edge_weights = torch.ones([edge_index.shape[1]], device=device, dtype=torch.float)

    g_conv = GConv(input_dim=num_features, hidden_dim=32, activation=nn.ReLU, num_layers=2).to(device)
    encoder_model = Encoder(encoder=g_conv, augmentor=(aug1, aug2), hidden_dim=32, proj_dim=16).to(device)
    contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='L2L', intraview_negs=True).to(device)

    # Pre-Training
    optimizer = optim.Adam(encoder_model.parameters(), lr=0.01)
    for epoch in range(1, 1001):
        loss = train(encoder_model, contrast_model, features, edge_index, edge_weights, optimizer)
        if epoch % 100 == 0:
            print('Epoch = {}@Loss = {:.4f}'.format(epoch, loss))

    # Fine-Tuning
    feat_embedding = encoder_model.encoder(features, edge_index, edge_weights)
    km = KMeans(n_clusters=num_cluster, n_init=30).fit(feat_embedding[effect_idx].cpu().detach().numpy())
    cluster_centers = torch.randn(size=(num_cluster, 32), requires_grad=True, device=device)
    cluster_centers.data = torch.tensor(km.cluster_centers_).to(device)
    optimizer = optim.Adam(encoder_model.parameters(), lr=0.001)
    for epoch in range(1, 501):
        loss = finetune(encoder_model, contrast_model, features, edge_index, edge_weights, optimizer, cluster_centers)
        if epoch % 100 == 0:
            print('Epoch = {}@Loss = {:.4f}'.format(epoch, loss))

    encoder_model.eval()
    num_nodes = labels.shape[0]
    embeddings = encoder_model.encoder(features, edge_index, edge_weights).detach().cpu().numpy()

    num_classes = torch.max(labels).item() + 1
    labels[attach_idx] = torch.tensor(num_classes, dtype=torch.int64).to(features.device)
    labels = labels.cpu().numpy()

    tsne = TSNE(n_components=2, perplexity=30, random_state=1027)
    vis_points = tsne.fit_transform(embeddings[effect_idx])

    matplotlib.use('Agg')
    fig, ax = plt.subplots(figsize=(5, 4))
    all_colors = [
        '#7fc97f', '#beaed4', '#fdc086', '#ffff99', '#386cb0', '#f0027f', '#bf5b16', '#000000'
    ]
    markers = ['o', 'v', '^', '<', '>', 'd', 's', 'p']

    for label in range(0, num_classes + 1):
        points = vis_points[labels[effect_idx] == label]
        print('Label = {}@Number of Samples = {}'.format(label + 1, len(points)))
        color = all_colors[label % 7 if label != num_classes else -1]
        ax.scatter(
            points[:, 0], points[:, 1], s=25, c=color,
            label='{}'.format(label + 1 if label != num_classes else 'poisoned'),
            marker=markers[label // 7 if label != num_classes else 0], alpha=1.0, edgecolors='face'
        )
    ax.legend(loc='upper left', ncol=4, labelspacing=0.6, prop={'size': 8})
    plt.savefig('{}_SSL_{}.svg'.format(dataset_name, attack_name))
