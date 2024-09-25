from collections import defaultdict

import torch
import logging
import os
import random
import numpy as np

from typing import List, Union
import torch_scatter
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import degree, to_undirected
from torch_geometric.utils import homophily

try:
    if 'logger' not in globals():
        logging.basicConfig()
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
except NameError:
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)


def seed_experiment(seed=0):
    import torch.backends.cudnn as cudnn
    # seed = 1234
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # TODO: Do we need deterministic in cudnn ? Double check
    cudnn.deterministic = True
    cudnn.benchmark = False
    logger.info('Seeded everything')


def get_split(data, device, node_idx=None):
    rs = np.random.RandomState(10)
    if node_idx is None:
        perm = rs.permutation(data.num_nodes)
    else:
        perm = node_idx[rs.permutation(len(node_idx))].cpu().numpy()
    train_number = int(0.2 * len(perm))

    num_classes = int(torch.max(data.y)) + 1
    if num_classes > 10:
        class2num_nodes = [0] * num_classes
        for i in range(num_classes):
            class2num_nodes[i] = train_number // num_classes
        for i in range(train_number % num_classes):
            class2num_nodes[i] += 1

        idx_train = None
        train_mask = torch.zeros_like(data.train_mask).to(device)
        for i in range(num_classes):
            _idx_train = torch.tensor(perm).to(device)
            _idx_train = _idx_train[data.y[perm] == i][:class2num_nodes[i]]
            idx_train = torch.cat([idx_train, _idx_train], dim=0) if idx_train is not None else _idx_train
            train_mask[_idx_train] = True
        idx_train = torch.sort(idx_train)[0]
        data.train_mask = train_mask.clone()
    else:
        idx_train = torch.tensor(sorted(perm[:train_number])).to(device)
        data.train_mask = torch.zeros_like(data.train_mask)
        data.train_mask[idx_train] = True

    val_number = int(0.1 * len(perm))
    idx_val = torch.tensor(sorted(perm[train_number: train_number + val_number])).to(device)
    data.val_mask = torch.zeros_like(data.val_mask)
    data.val_mask[idx_val] = True

    test_number = int(0.2 * len(perm))
    idx_test = torch.tensor(sorted(perm[train_number + val_number: train_number + val_number + test_number])).to(device)
    data.test_mask = torch.zeros_like(data.test_mask)
    data.test_mask[idx_test] = True

    idx_clean_test = idx_test[:int(len(idx_test) / 2)]
    idx_atk = idx_test[int(len(idx_test) / 2):]

    return data, idx_train, idx_val, idx_clean_test, idx_atk


def subgraph(subset, edge_index, edge_attr=None, relabel_nodes: bool = False):
    """Returns the induced subgraph of :obj:`(edge_index, edge_attr)`
    containing the nodes in :obj:`subset`.

    Args:
        subset (LongTensor, BoolTensor or [int]): The nodes to keep.
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)
        relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
            :obj:`edge_index` will be relabeled to hold consecutive indices
            starting from zero. (default: :obj:`False`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`, :class:`Tensor`)
    """

    node_mask = subset
    edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
    edge_index = edge_index[:, edge_mask]
    edge_attr = edge_attr[edge_mask] if edge_attr is not None else None
    return edge_index, edge_attr, edge_mask


@functional_transform('column_normalize_features')
class ColumnNormalizeFeatures(BaseTransform):
    r"""Column-normalizes the attributes given in :obj:`attrs` to sum-up to one
    (functional name: :obj:`normalize_features`).

    Args:
        attrs (List[str]): The names of attributes to normalize.
            (default: :obj:`["x"]`)
    """
    def __init__(self, attrs: List[str] = ["x"]):
        super(BaseTransform, self).__init__()
        self.attrs = attrs

    def forward(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        for store in data.stores:
            for key, value in store.items(*self.attrs):
                if value.numel() > 0:
                    value = value - value.min(dim=0, keepdim=True)[0]
                    value.div_(value.max(dim=0, keepdim=True)[0] + 1e-6)
                    store[key] = value
        return data

    def __call__(self, data: Union[Data, HeteroData]) -> Union[Data, HeteroData]:
        return self.forward(data)


def calc_adjusted_homophily(edge_index, labels):

    edge_index = to_undirected(edge_index)

    num_labels = torch.max(labels).item() + 1
    label_degree_cnt = torch.zeros(num_labels, dtype=torch.float)

    node_degrees = degree(edge_index[0])
    label_degree_cnt = torch_scatter.scatter(node_degrees, labels, reduce='sum')

    num_edges = edge_index.shape[1]
    total = torch.sum(label_degree_cnt ** 2) / num_edges ** 2

    edge_hm = homophily(edge_index, labels, method='edge')
    adjusted_homophily = (edge_hm - total) / (1.0 - total)
    return adjusted_homophily
