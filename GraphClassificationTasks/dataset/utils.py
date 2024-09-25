import os
import numpy as np
from collections import defaultdict

from sklearn.model_selection import train_test_split
from torch_geometric.datasets import TUDataset, MNISTSuperpixels


def get_dataset(dataset_name, use_node_attr=False):
    dataset_dir = 'data'
    if dataset_name in ['MNIST']:
        dataset_dir = os.path.join('data', dataset_name)
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

    if dataset_name in ['ENZYMES', 'PROTEINS']:
        dataset = TUDataset(root=dataset_dir, name=dataset_name, use_node_attr=use_node_attr)
    elif dataset_name in ['MNIST']:
        dataset = MNISTSuperpixels(root=dataset_dir)
    else:
        raise Exception('Dataset not found')

    num_classes = dataset.num_classes
    num_features = dataset.num_features
    return dataset, num_features, num_classes


def get_split(dataset, test_size=0.25, seed=1027):
    train_dataset, all_test_dataset = train_test_split(dataset, test_size=test_size, random_state=seed)
    clean_test_dataset, atk_test_dataset = train_test_split(all_test_dataset, test_size=0.5, random_state=seed)
    return train_dataset, clean_test_dataset, atk_test_dataset


def has_node(graph, num_attributes, nodeLabel: int):
    sum_array = graph.x.sum(axis=0).cpu().numpy().astype(int)
    return sum_array[num_attributes + nodeLabel] > 0


def get_attach_idx(dataset, vs_ratio=0.1, target_label=1,
                   clean_label=False, chosen_method=None,
                   num_node_attributes=None, num_node_labels=None, num_classes=None):

    label_list = [data.y.item() for data in dataset]
    label_np = np.array(label_list)

    num_groundth_label = np.sum(label_np == target_label)
    if clean_label:
        attach_idx = np.where(label_np == target_label)[0]
    else:
        attach_idx = np.where(label_np != target_label)[0]

    if chosen_method is None:
        attach_idx = np.random.permutation(attach_idx)
        num_attach = int(min(len(attach_idx) * vs_ratio, vs_ratio * num_groundth_label))
        attach_idx = attach_idx[:num_attach]
    else:
        occ_num = np.zeros(num_node_labels, dtype=int)
        nodes_table = defaultdict(
            lambda: {"occ": 0, **{x: 0 for x in range(0, num_classes)}})
        poisoning_num = int(len(attach_idx) * vs_ratio)

        for graph in dataset:
            # count the occurrence number of each node
            sum_array = graph.x.sum(axis=0).cpu().numpy().astype(int)
            occ_num += sum_array[num_node_attributes:]

            # count the corresponding graph labels
            for node, num in enumerate(sum_array[num_node_attributes:]):
                if num > 0:
                    nodes_table[node][graph.y.item()] += 1

        for node, num in enumerate(occ_num):
            nodes_table[node]["occ"] = num

        trigger_node, min_diff = -1, float("inf")
        for node in nodes_table:
            ava_num = sum(nodes_table[node][label] for label in range(0, num_classes)) - nodes_table[node][target_label]
            diff = abs(ava_num - poisoning_num)
            if diff < min_diff:
                trigger_node, min_diff = node, diff

        # count the average occurrence number of the trigger node
        t = nodes_table[trigger_node]["occ"] / (
                sum(nodes_table[trigger_node].values()) - nodes_table[trigger_node]["occ"])

        attach_idx = []
        for idx in range(len(dataset)):
            if has_node(dataset[idx], num_node_attributes, trigger_node) and dataset[idx].y.item() != target_label:
                attach_idx.append(idx)
        attach_idx = np.array(attach_idx, dtype=np.int32)

    if chosen_method is None:
        return attach_idx
    return attach_idx, trigger_node, t
