import copy
import numpy as np

import torch
import torch.nn.functional as F

import matplotlib
import matplotlib.pyplot as plt

import torch_geometric as torch_geo
from sklearn.manifold import TSNE


def get_colors(color_name, lut):
    return plt.get_cmap(color_name, lut)([i for i in range(lut)])


@torch.no_grad()
def visualize_embedding(model, features, edge_index, labels, train_node_idx, attach_idx, target_label, dataset_name, attack_name):
    model = copy.deepcopy(model)
    features = copy.deepcopy(features)
    edge_index = copy.deepcopy(edge_index)
    labels = copy.deepcopy(labels)

    num_nodes = labels.shape[0]
    num_classes = torch.max(labels).item() + 1
    labels[attach_idx] = torch.tensor(num_classes, dtype=torch.int64).to(features.device)
    labels = labels.cpu().numpy()
    activation = {}

    def get_activation(l_name):
        def hook(_, __, output):
            activation[l_name] = output.clone().detach()
        return hook

    layer_num, layer_name = 0, None
    for name, layer in model.named_modules():
        if isinstance(layer, torch_geo.nn.MessagePassing):
            layer_num += 1
            if layer_num == 2:
                layer_name = name
                layer.register_forward_hook(get_activation(name))
                break

    logits = model(features, edge_index)
    preds = torch.argmax(logits, dim=-1)[:num_nodes]

    preds[attach_idx] = torch.where(preds[attach_idx] == target_label, num_classes, preds[attach_idx])
    pred_correct_index = np.where(preds.cpu().numpy() == labels)[0]
    labels = labels[pred_correct_index]

    embeddings = torch.softmax(activation[layer_name][pred_correct_index], dim=-1).cpu().numpy()

    # only display small part of nodes
    num_nodes_per_cls, all_index = 0, []
    if num_classes == 18:
        num_nodes_per_cls = 40
    elif num_classes == 7:
        num_nodes_per_cls = 100
    elif num_classes == 2:
        num_nodes_per_cls = 400
    else:
        num_nodes_per_cls = 100
    for label in range(0, num_classes + 1):
        index = np.random.choice(np.where(labels == label)[0], min(num_nodes_per_cls, np.sum(labels == label)), replace=False)
        all_index.extend(index.tolist())

    tsne = TSNE(n_components=2, perplexity=30, random_state=1027)
    vis_points = tsne.fit_transform(embeddings[all_index])

    # import matplotlib
    matplotlib.use('Agg')
    fig, ax = plt.subplots(figsize=(5, 4))
    all_colors = [
        '#7fc97f', '#beaed4', '#fdc086', '#ffff99', '#386cb0', '#f0027f', '#bf5b16', '#000000'
    ]
    markers = ['o', 'v', '^', '<', '>', 'd', 's', 'p']

    for label in range(0, num_classes + 1):
        points = vis_points[labels[all_index] == label]
        print('Label = {}@Number of Samples = {}'.format(label + 1, len(points)))
        color = all_colors[label % 7 if label != num_classes else -1]
        ax.scatter(
            points[:, 0], points[:, 1], s=25, c=color,
            label='{}'.format(label + 1 if label != num_classes else 'poisoned'),
            marker=markers[label // 7 if label != num_classes else 0], alpha=1.0, edgecolors='face'
        )
    ax.legend(loc='upper left', ncol=4, labelspacing=0.6, prop={'size': 8})
    plt.savefig('{}_Supervised_{}.svg'.format(dataset_name, attack_name))
    effected_idx = pred_correct_index[all_index]
    return effected_idx
