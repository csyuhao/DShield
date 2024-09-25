""" Neighboring Backdoor Attacks on Graph Convolutional Network
"""
from typing import Optional, Tuple, Union
from tqdm import tqdm

import torch
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F

from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_geometric.utils import (
    degree,
    scatter,
    sort_edge_index,
    to_dense_batch,
)
from torch_geometric.typing import OptTensor
from torch_sparse import SparseTensor, matmul

from models.GCN import GCN


class Surrogate(Module):
    """Base class for attacker or defenders that require
    a surrogate model for estimating labels or computing
    gradient information.

    Parameters
    ----------
    device : str, optional
        the device of a model to use for, by default "cpu"

    """
    _is_setup = False  # flags to denote the surrogate model is properly set

    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.tau = None
        self.surrogate = None
        self.device = torch.device(device)

    def setup_surrogate(
            self, surrogate: Module, *, tau: float = 1.0, freeze: bool = True,
            required: Union[Module, Tuple[Module]] = None) -> "Surrogate":
        """Method used to initialize the (trained) surrogate model.

        Parameters
        ----------
        surrogate : Module
            the input surrogate module
        tau : float, optional
            temperature used for softmax activation, by default 1.0
        freeze : bool, optional
            whether to freeze the model's parameters to save time,
            by default True
        required : Union[Module, Tuple[Module]], optional
            which class(es) of the surrogate model are required,
            by default None

        Returns
        -------
        Surrogate
            the class itself

        Raises
        ------
        RuntimeError
            if the surrogate model is not an instance of
            :class:`torch.nn.Module`
        RuntimeError
            if the surrogate model is not an instance of :obj:`required`
        """

        if not isinstance(surrogate, Module):
            raise RuntimeError(
                "The surrogate model must be an instance of `torch.nn.Module`."
            )

        if required is not None and not isinstance(surrogate, required):
            raise RuntimeError(
                f"The surrogate model is required to be `{required}`, "
                f"but got `{surrogate.__class__.__name__}`.")

        surrogate.eval()
        if hasattr(surrogate, 'cache_clear'):
            surrogate.cache_clear()

        for layer in surrogate.modules():
            if hasattr(layer, 'cached'):
                layer.cached = False

        self.surrogate = surrogate.to(self.device)
        self.tau = tau

        if freeze:
            self.freeze_surrogate()

        self._is_setup = True

        return self

    def clip_grad(
        self,
        grad: Tensor,
        grad_clip: Optional[float],
    ) -> Tensor:
        """Gradient clipping function

        Parameters
        ----------
        grad : Tensor
            the input gradients to clip
        grad_clip : Optional[float]
            the clipping number of the gradients

        Returns
        -------
        Tensor
            the clipped gradients
        """
        if grad_clip is not None:
            grad_len_sq = grad.square().sum()
            if grad_len_sq > grad_clip * grad_clip:
                grad *= grad_clip / grad_len_sq.sqrt()
        return grad

    def estimate_self_training_labels(
            self, nodes: Optional[Tensor] = None) -> Tensor:
        """Estimate the labels of nodes using the trained surrogate model.

        Parameters
        ----------
        nodes : Optional[Tensor], optional
            the input nodes, if None, it would be all nodes in the graph,
            by default None

        Returns
        -------
        Tensor
            the labels of the input nodes.
        """
        self_training_labels = self.surrogate(self.feat, self.edge_index,
                                              self.edge_weight)
        if nodes is not None:
            self_training_labels = self_training_labels[nodes]
        return self_training_labels.argmax(-1)

    def freeze_surrogate(self) -> "Surrogate":
        """Freeze the parameters of the surrogate model.

        Returns
        -------
        Surrogate
            the class itself
        """
        for para in self.surrogate.parameters():
            para.requires_grad_(False)
        return self

    def defrozen_surrogate(self) -> "Surrogate":
        """Defrozen the parameters of the surrogate model

        Returns
        -------
        Surrogate
            the class itself
        """
        for para in self.surrogate.parameters():
            para.requires_grad_(True)
        return self


class LGCBackdoor:
    """ Implementation of `LGCB` attack from the:
    `"Neighboring Backdoor Attacks on Graph Convolutional Network"
    <https://arxiv.org/abs/2201.06202>`_ paper (arXiv'22)
    """

    def __init__(self, num_budgets, hidden, epochs, target_class, device):
        self.W, self.num_classes = None, None
        self.target_class = target_class
        self.device = device
        self.num_budgets = num_budgets
        self.hidden = hidden
        self.epochs = epochs

    @torch.no_grad()
    def setup_shadow_model(self, shallow_model):
        W = None
        for param in shallow_model.parameters():
            if param.ndim == 1:
                continue
            if W is None:
                W = param.detach()
            else:
                W = param.detach() @ W
        self.W = W.t()
        self.num_classes = self.W.size(-1)

    @torch.no_grad()
    def get_feat_perturbations(self, W, target_class, num_budgets):
        D = W - W[:, target_class].view(-1, 1)
        D = D.sum(1)
        _, indices = torch.topk(D, k=num_budgets, largest=False)
        return indices

    def build_shadow_model(self, features, edge_index, edge_weight, labels, idx_train):

        if edge_weight is None:
            edge_weight = torch.ones([edge_index.shape[1]], device=self.device, dtype=torch.float)

        shadow_model = GCN(
            n_feat=features.shape[1], n_hid=self.hidden,
            n_class=labels.max().item() + 1, dropout=0.0, device=self.device
        ).to(self.device)

        # Train models
        shadow_model.train()
        shadow_model.fit(features, edge_index, edge_weight, labels, idx_train, train_iters=self.epochs)
        self.setup_shadow_model(shadow_model)

    def get_poisoned(self, features, edge_index, edge_weight, labels, attach_idx):

        if edge_weight is None:
            edge_weight = torch.ones([edge_index.shape[1]], device=self.device, dtype=torch.float)

        feat_perturbations = self.get_feat_perturbations(self.W, self.target_class, self.num_budgets)

        num_target_nodes = attach_idx.shape[0]
        inject_feats = torch.zeros(size=(num_target_nodes, features.shape[1]), dtype=torch.float32, device=self.device)
        target_labels = torch.LongTensor([self.target_class]).to(self.device).repeat(num_target_nodes)

        val_min, val_max = torch.min(features).item(), torch.max(features).item()
        if val_max <= 1.0 and val_min >= 0.:
            inject_feats[:, feat_perturbations] = 1.0
        else:
            # OGBN-arXiv范围不是0-1
            inject_feats[:, feat_perturbations] = 10.0

        num_nodes = features.shape[0]
        inject_node_idx = torch.arange(start=num_nodes, end=num_nodes + num_target_nodes, dtype=torch.int64, device=self.device)
        inject_edge_index = torch.stack(
            (attach_idx.reshape(-1), inject_node_idx.reshape(-1)),
            dim=0
        )
        inject_diag_index = torch.arange(num_nodes, num_nodes + num_target_nodes, device=self.device).repeat(2, 1)
        inject_edge_index = torch.cat((inject_edge_index, inject_edge_index[[1, 0]], inject_diag_index), dim=1)
        inject_edge_weight = torch.ones([inject_edge_index.shape[1]], device=self.device, dtype=torch.float)

        features = torch.cat((features, inject_feats), dim=0)
        edge_index = torch.cat((edge_index, inject_edge_index), dim=1)
        edge_weight = torch.cat((edge_weight, inject_edge_weight), dim=0)
        if labels is not None:
            labels[attach_idx] = target_labels
            return features, edge_index, edge_weight, labels
        return features, edge_index, edge_weight

    def inject_trigger(self, attach_idx, features, edge_index, edge_weight):
        features, edge_index, edge_weight = features.clone(), edge_index.clone(), edge_weight.clone()
        return self.get_poisoned(features, edge_index, edge_weight, None, attach_idx)


class FGBackdoor(Surrogate):
    """ Implementation of `GB-FGSM` attack from the:
    `"Neighboring Backdoor Attacks on Graph Convolutional Network"
    <https://arxiv.org/abs/2201.06202>`_ paper (arXiv'22)
    """

    def __init__(self, num_budgets, hidden, epochs, target_class, device):
        super().__init__(device)
        self.num_classes = None
        self.w1, self.w2 = None, None
        self.target_class = target_class
        self.device = device
        self.num_budgets = num_budgets
        self.hidden = hidden
        self.epochs = epochs

    def setup_shadow_model(self, shallow_model, tau: float = 1.0):
        Surrogate.setup_surrogate(self, surrogate=shallow_model, tau=tau, freeze=True)

        W = []
        for param in shallow_model.parameters():
            if param.ndim == 1:
                continue
            else:
                W.append(param.detach().t())
        self.w1, self.w2 = W
        self.num_classes = W[-1].size(-1)

    def build_shadow_model(self, features, edge_index, edge_weight, labels, idx_train, tau):

        if edge_weight is None:
            edge_weight = torch.ones([edge_index.shape[1]], device=self.device, dtype=torch.float)

        shadow_model = GCN(
            n_feat=features.shape[1], n_hid=self.hidden,
            n_class=labels.max().item() + 1, dropout=0.0, device=self.device
        ).to(self.device)

        # Train models
        shadow_model.train()
        shadow_model.fit(features, edge_index, edge_weight, labels, idx_train, train_iters=self.epochs)
        self.setup_shadow_model(shadow_model, tau)

    def get_poisoned(self, features, edge_index, edge_weight, labels, attach_idx, disable=False):

        num_nodes = features.shape[0]
        num_target_nodes = attach_idx.shape[0]
        inject_feats = torch.zeros(size=(num_target_nodes, features.shape[1]), dtype=torch.float32, device=self.device).requires_grad_()
        target_labels = torch.LongTensor([self.target_class]).to(self.device).repeat(num_target_nodes)

        (edge_index, edge_weight_with_trigger, edge_index_with_self_loop, edge_weight,
            trigger_edge_index, trigger_edge_weight, augmented_edge_index, augmented_edge_weight) = get_backdoor_edges(
            edge_index, attach_idx, num_nodes
        )

        for _ in tqdm(range(self.num_budgets), desc="Updating trigger using gradients...", disable=disable):
            inject_feats.grad = None
            aug_feat = torch.cat([features, inject_feats], dim=0)
            feat1 = aug_feat @ self.w1
            h1 = spmm(feat1, edge_index_with_self_loop, edge_weight)
            h1_aug = spmm(feat1, augmented_edge_index, augmented_edge_weight).relu()
            h = spmm(h1_aug @ self.w2, trigger_edge_index, trigger_edge_weight)
            h += spmm(h1 @ self.w2, edge_index, edge_weight_with_trigger)
            h = h[attach_idx] / self.tau
            loss = F.cross_entropy(h, target_labels)
            gradients = torch.autograd.grad(-loss, inject_feats, retain_graph=False)[0] * (1. - inject_feats)
            inject_feats.data.scatter_(dim=1, index=gradients.argmax(dim=1, keepdim=True), value=1.0)

        inject_edge_index = trigger_edge_index.clone()
        features = torch.cat((features, inject_feats), dim=0).detach()
        edge_index = torch.cat((edge_index, inject_edge_index), dim=1)
        edge_weight = torch.ones([edge_index.shape[1]], device=self.device, dtype=torch.float)
        if labels is not None:
            labels[attach_idx] = target_labels
            return features, edge_index, edge_weight, labels
        return features, edge_index, edge_weight

    def inject_trigger(self, attach_idx, features, edge_index, edge_weight):
        return self.get_poisoned(features, edge_index, edge_weight, None, attach_idx)


def spmm(x: Tensor, edge_index: Union[Tensor, SparseTensor],
         edge_weight: OptTensor = None, reduce: str = 'sum') -> Tensor:
    r"""Sparse-dense matrix multiplication.

    Parameters
    ----------
    x : torch.Tensor
        the input dense 2D-matrix
    edge_index : torch.Tensor
        the location of the non-zeros elements in the sparse matrix,
        denoted as :obj:`edge_index` with shape [2, M]
    edge_weight : Optional[Tensor], optional
        the edge weight of the sparse matrix, by default None
    reduce : str, optional
        reduction of the sparse matrix multiplication, including:
        (:obj:`'mean'`, :obj:`'sum'`, :obj:`'add'`,
        :obj:`'max'`, :obj:`'min'`, :obj:`'median'`,
        :obj:`'sample_median'`)
        by default :obj:`'sum'`

    Returns
    -------
    Tensor
        the output result of the matrix multiplication.

    See also
    --------
    :class:`~torch_geometric.utils.spmm` (>=2.2.0)
    """

    # Case 1: `torch_sparse.SparseTensor`
    if isinstance(edge_index, SparseTensor):
        assert reduce in ['sum', 'add', 'mean', 'min', 'max']
        return matmul(edge_index, x, reduce)

    # Case 2: `torch.sparse.Tensor` (Sparse) and `torch.FloatTensor` (Dense)
    if isinstance(edge_index, Tensor) and (edge_index.is_sparse
                                           or edge_index.dtype == torch.float):
        assert reduce in ['sum', 'add']
        return torch.sparse.mm(edge_index, x)

    # Case 3: `torch.LongTensor` (Sparse)
    if reduce == 'median':
        return scatter_median(x, edge_index, edge_weight)
    elif reduce == 'sample_median':
        return scatter_sample_median(x, edge_index, edge_weight)

    row, col = edge_index
    x = x if x.dim() > 1 else x.unsqueeze(-1)

    out = x[row]
    if edge_weight is not None:
        out = out * edge_weight.unsqueeze(-1)
    out = scatter(out, col, dim=0, dim_size=x.size(0), reduce=reduce)
    return out


def scatter_median(x: Tensor, edge_index: Tensor,
                   edge_weight: OptTensor = None) -> Tensor:
    # NOTE: `to_dense_batch` requires the `index` is sorted by column
    ix = torch.argsort(edge_index[1])
    edge_index = edge_index[:, ix]
    row, col = edge_index
    x_j = x[row]

    if edge_weight is not None:
        x_j = x_j * edge_weight[ix].unsqueeze(-1)

    dense_x, mask = to_dense_batch(x_j, col, batch_size=x.size(0))
    h = x_j.new_zeros(dense_x.size(0), dense_x.size(-1))
    deg = mask.sum(dim=1)
    for i in deg.unique():
        if i == 0:
            continue
        deg_mask = deg == i
        h[deg_mask] = dense_x[deg_mask, :i].median(dim=1).values
    return h


def scatter_sample_median(x: Tensor, edge_index: Tensor,
                          edge_weight: OptTensor = None) -> Tensor:
    """Approximating the median aggregation with fixed set of
    neighborhood sampling."""

    try:
        from glcore import neighbor_sampler_cpu  # noqa
    except (ImportError, ModuleNotFoundError):
        raise ModuleNotFoundError(
            "`scatter_sample_median` requires glcore which "
            "is not installed, please refer to "
            "'https://github.com/EdisonLeeeee/glcore' "
            "for more information.")

    if edge_weight is not None:
        edge_index, edge_weight = sort_edge_index(edge_index, edge_weight,
                                                  sort_by_row=False)
    else:
        edge_index = sort_edge_index(edge_index, sort_by_row=False)

    row, col = edge_index
    num_nodes = x.size(0)
    deg = degree(col, dtype=torch.long, num_nodes=num_nodes)
    colptr = torch.cat([deg.new_zeros(1), deg.cumsum(dim=0)], dim=0)
    replace = True
    size = int(deg.float().mean().item())
    nodes = torch.arange(num_nodes)
    targets, neighbors, e_id = neighbor_sampler_cpu(colptr.cpu(), row.cpu(),
                                                    nodes, size, replace)

    x_j = x[neighbors]

    if edge_weight is not None:
        x_j = x_j * edge_weight[e_id].unsqueeze(-1)

    return x_j.view(num_nodes, size, -1).median(dim=1).values


def get_backdoor_edges(edge_index, attach_idx, num_nodes) -> Tuple:
    device = edge_index.device
    num_target_nodes = attach_idx.shape[0]
    influence_nodes = attach_idx.clone()

    num_all_nodes = num_nodes + num_target_nodes
    trigger_nodes = torch.arange(num_nodes, num_all_nodes, device=device)

    # 1. edge index of original graph (without selfloops)
    edge_index, _ = remove_self_loops(edge_index)

    # 2. edge index of original graph (with selfloops)
    edge_index_with_self_loop, _ = add_self_loops(edge_index)

    # 3. edge index of trigger nodes connected to victim nodes with selfloops
    trigger_edge_index = torch.stack([trigger_nodes, influence_nodes], dim=0)
    diag_index = torch.arange(num_nodes, num_all_nodes, device=device).repeat(2, 1)
    trigger_edge_index = torch.cat([trigger_edge_index, trigger_edge_index[[1, 0]], diag_index], dim=1)

    # 4. all edge index with trigger nodes
    augmented_edge_index = torch.cat([edge_index, trigger_edge_index], dim=1)

    d = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.float)
    d_augmented = d.clone()
    d_augmented[influence_nodes] += 1.
    d_augmented = torch.cat([d_augmented, torch.full(trigger_nodes.size(), 2, device=device)])

    d_pow = d.pow(-0.5)
    d_augmented_pow = d_augmented.pow(-0.5)

    edge_weight = d_pow[edge_index_with_self_loop[0]] * d_pow[edge_index_with_self_loop[1]]
    edge_weight_with_trigger = d_augmented_pow[edge_index[0]] * d_pow[edge_index[1]]
    trigger_edge_weight = d_augmented_pow[trigger_edge_index[0]] * d_augmented_pow[trigger_edge_index[1]]
    augmented_edge_weight = torch.cat([edge_weight_with_trigger, trigger_edge_weight], dim=0)

    return edge_index, edge_weight_with_trigger, edge_index_with_self_loop, edge_weight, \
        trigger_edge_index, trigger_edge_weight, augmented_edge_index, augmented_edge_weight
