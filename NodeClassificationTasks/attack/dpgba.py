import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.functional import cosine_similarity

from copy import deepcopy
import logging
import numpy as np

from models.GCN import GCN
from models.MLP import MLP
from models.reconstruct import MLPAE
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


class GradWhere(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input, thrd, device):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        rst = torch.where(input > thrd, torch.tensor(1.0, device=device, requires_grad=True),
                          torch.tensor(0.0, device=device, requires_grad=True))
        return rst

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()

        """
        Return results number should corresponding with .forward inputs (besides ctx),
        for each input, return a corresponding backward grad
        """
        return grad_input, None, None


class GraphTrojanNet(nn.Module):
    # In the future, we may use a GNN model to generate backdoor
    def __init__(self, device, nfeat, nout, layernum=1, dropout=0.00):
        super(GraphTrojanNet, self).__init__()

        layers = []
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        for l in range(layernum - 1):
            layers.append(nn.Linear(nfeat, nfeat))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))

        self.layers = nn.Sequential(*layers).to(device)

        self.feat = nn.Linear(nfeat, nout * (nfeat))
        # self.pattern = nn.Linear(nfeat,40)
        self.edge = nn.Linear(nfeat, int(nout * (nout - 1) / 2))
        self.device = device

    def forward(self, input, thrd):

        """
        "input", "mask" and "thrd", should already in cuda before sent to this function.
        If using sparse format, corresponding tensor should already in sparse format before
        sent into this function
        """

        GW = GradWhere.apply
        self.layers = self.layers
        h = self.layers(input)

        feat = self.feat(h)

        edge_weight = self.edge(h)

        edge_weight = GW(edge_weight, thrd, self.device)

        return feat, edge_weight


class HomoLoss(nn.Module):
    def __init__(self, device):
        super(HomoLoss, self).__init__()
        self.device = device

    def forward(self, trigger_edge_index, trigger_edge_weights, x, thrd):
        trigger_edge_index = trigger_edge_index[:, trigger_edge_weights > 0.0]
        edge_sims = F.cosine_similarity(x[trigger_edge_index[0]], x[trigger_edge_index[1]])

        loss = torch.relu(thrd - edge_sims).mean()
        return loss


class DPGBA:

    def __init__(self, seed, thrd, hidden, trojan_epochs, inner_epochs, rec_epochs, k, outter_size,
                 lr, weight_decay, target_weight, ood_weight, target_class_weight, trigger_size, target_class, dataset_name, device):
        self.device = device
        self.dataset = dataset_name
        self.trigger_size = trigger_size
        self.target_class = target_class
        self.lr, self.weight_decay = lr, weight_decay
        self.seed, self.thrd, self.hidden = seed, thrd, hidden
        self.trojan_epochs, self.inner_epochs = trojan_epochs, inner_epochs
        self.trigger_index = self.get_trigger_index(self.trigger_size)
        self.rec_epochs, self.k, self.outter_size = rec_epochs, k, outter_size
        self.target_weight, self.ood_weight, self.target_class_weight = target_weight, ood_weight, target_class_weight

    def get_trigger_index(self, trigger_size):
        edge_list = [[0, 0]]
        for j in range(trigger_size):
            for k in range(j):
                edge_list.append([j, k])
        edge_index = torch.tensor(edge_list, device=self.device).long().T
        return edge_index

    def calculate_mean_cosine_similarity(self, trojan_feat):
        n = trojan_feat.size(0)  # Number of samples

        # Initialize a tensor to store cosine similarities
        similarities = torch.zeros((n, n))

        # Calculate cosine similarity for each pair of rows
        for i in range(n):
            for j in range(i + 1, n):
                sim = cosine_similarity(trojan_feat[i].unsqueeze(0), trojan_feat[j].unsqueeze(0))
                similarities[i, j] = sim
                similarities[j, i] = sim            # The similarity matrix is symmetric

        # Exclude self-similarities and calculate mean
        similarities.fill_diagonal_(0)
        mean_similarity = similarities.sum() / (n * (n - 1))
        return mean_similarity.item()

    def get_trojan_edge(self, start, idx_attach, trigger_size):
        edge_list = []
        for idx in idx_attach:
            edges = self.trigger_index.clone()
            edges[0, 0] = idx
            edges[1, 0] = start
            edges[:, 1:] = edges[:, 1:] + start

            edge_list.append(edges)
            start += trigger_size
        edge_index = torch.cat(edge_list, dim=1)
        row = torch.cat([edge_index[0], edge_index[1]])
        col = torch.cat([edge_index[1], edge_index[0]])
        edge_index = torch.stack([row, col])
        return edge_index

    def get_trojan_edge_arxiv(self, start, idx_attach, trigger_size):
        edge_list = []
        for idx in idx_attach:
            edges = self.trigger_index.clone()
            edges = edges[:, 1:]
            edges += start
            edge_list.append(edges)
            for i in range(trigger_size):
                edge_list.append(torch.tensor([[idx], [start + i]], device=self.device))
            start += trigger_size
        edge_index = torch.cat(edge_list, dim=1)
        row = torch.cat([edge_index[0], edge_index[1]])
        col = torch.cat([edge_index[1], edge_index[0]])
        edge_index = torch.stack([row, col])
        return edge_index

    def inject_trigger(self, attach_idx, features, edge_index, edge_weight):
        self.trojan.eval()
        features, edge_index, edge_weight = features.clone(), edge_index.clone(), edge_weight.clone()

        trojan_feat, trojan_weights = self.trojan(features[attach_idx], self.thrd)  # may revise the process of generate
        trojan_weights = torch.cat([torch.ones([len(attach_idx), 1], dtype=torch.float, device=self.device), trojan_weights], dim=1)
        trojan_weights = trojan_weights.flatten()
        trojan_feat = trojan_feat.view([-1, features.shape[1]])
        trojan_edge = self.get_trojan_edge(len(features), attach_idx, self.trigger_size).to(self.device)

        update_edge_weights = torch.cat([edge_weight, trojan_weights, trojan_weights])
        update_feat = torch.cat([features, trojan_feat])
        update_edge_index = torch.cat([edge_index, trojan_edge], dim=1)
        return update_feat, update_edge_index, update_edge_weights

    def inject_trigger_arxiv(self, attach_idx, features, edge_index, edge_weight):
        self.trojan.eval()
        features, edge_index, edge_weight = features.clone(), edge_index.clone(), edge_weight.clone()

        trojan_feat, trojan_weights = self.trojan(features[attach_idx], self.thrd)  # may revise the process of generate
        trojan_weights = torch.cat(
            [trojan_weights, torch.ones([len(trojan_feat), 1 * self.trigger_size], dtype=torch.float, device=self.device)],
            dim=1
        )
        trojan_weights = trojan_weights.flatten()
        trojan_feat = trojan_feat.view([-1, features.shape[1]])
        trojan_edge = self.get_trojan_edge_arxiv(len(features), attach_idx, self.trigger_size).to(self.device)

        update_edge_weights = torch.cat([edge_weight, trojan_weights, trojan_weights])
        update_feat = torch.cat([features, trojan_feat])
        update_edge_index = torch.cat([edge_index, trojan_edge], dim=1)

        return update_feat, update_edge_index, update_edge_weights

    def simi(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        s = torch.mm(z1, z2.t())
        return s

    def con_loss(self, z1, z2, labels, output):
        f = lambda x: torch.exp(x / 1.0)
        loss = 0
        z2_target = z2[labels == self.target_class]
        output = output[:, self.target_class]
        output_target = torch.exp(output[labels == self.target_class])
        z2_non_target = z2[labels != self.target_class]
        _, sorted_indices = torch.sort(output_target, descending=True)
        sorted_z2_target = z2_target[sorted_indices]
        # Here we randomly select some samples from target class to reduce computation cost
        N = 10
        top_embeddings = sorted_z2_target[:N]
        intra = f(self.simi(z1, top_embeddings))
        inter = f(self.simi(z1, z2_non_target))
        one_to_all_inter = inter.sum(1, keepdim=True)
        one_to_all_inter = one_to_all_inter.repeat(1, len(intra[0]))
        denomitor = one_to_all_inter + intra
        loss += -torch.log(intra / denomitor).mean()
        return loss

    def fit(self, features, edge_index, edge_weight, labels, train_idx, attach_idx, unlabeled_idx):

        if edge_weight is None:
            edge_weight = torch.ones([edge_index.shape[1]], device=self.device, dtype=torch.float)

        # initial a shadow model
        self.shadow_model = GCN(n_feat=features.shape[1],
                                n_hid=self.hidden,
                                n_class=labels.max().item() + 1,
                                dropout=0.0, device=self.device).to(self.device)
        # initial an ood detector
        self.ood_detector = MLP(n_feat=features.shape[1], n_hid=self.hidden,
                                n_class=2, dropout=0.0, device=self.device).to(self.device)

        # initialize a trojanNet to generate trigger
        self.trojan = GraphTrojanNet(self.device, features.shape[1], self.trigger_size, layernum=2).to(self.device)
        self.homo_loss = HomoLoss(self.device)
        optimizer_shadow = optim.Adam(self.shadow_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        optimizer_detector = optim.Adam(self.ood_detector.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        optimizer_trigger = optim.Adam(self.trojan.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # find representative nodes in idx_unlabeled
        features_select = features[torch.cat([train_idx, attach_idx, unlabeled_idx])]
        AE = MLPAE(features_select, features_select, self.device, self.rec_epochs)
        AE.fit()
        rec_score_ori = AE.inference(features_select)
        mean_ori = torch.mean(rec_score_ori)
        std_ori = torch.std(rec_score_ori)
        condition = torch.abs(rec_score_ori - mean_ori) < std_ori
        selected_features = features_select[condition]
        labels = labels.clone()
        labels[attach_idx] = self.target_class
        if self.dataset == 'ogbn-arxiv':
            trojan_edge = self.get_trojan_edge_arxiv(len(features), attach_idx, self.trigger_size).to(self.device)
        else:
            trojan_edge = self.get_trojan_edge(len(features), attach_idx, self.trigger_size).to(self.device)
        poison_edge_index = torch.cat([edge_index, trojan_edge], dim=1)

        # begin training
        loss_best, best_state_dict = 1e8, None
        for i in range(self.trojan_epochs):
            for j in range(self.inner_epochs):
                # optimize surrogate model
                optimizer_shadow.zero_grad()
                trojan_feat, trojan_weights = self.trojan(features[attach_idx], self.thrd)  # may revise the process of generate
                if self.dataset == 'ogbn-arxiv':
                    trojan_weights = torch.cat([
                        trojan_weights,
                        torch.ones([len(trojan_feat), 1 * self.trigger_size], dtype=torch.float, device=self.device)], dim=1)
                else:
                    trojan_weights = torch.cat([torch.ones([len(trojan_feat), 1], dtype=torch.float, device=self.device), trojan_weights], dim=1)
                trojan_weights = trojan_weights.flatten()
                trojan_feat = trojan_feat.view([-1, features.shape[1]])
                poison_edge_weights = torch.cat(
                    [edge_weight, trojan_weights, trojan_weights]).detach()  # repeat trojan weights because of undirected edge
                poison_x = torch.cat([features, trojan_feat]).detach()
                output, all_features = self.shadow_model(poison_x, poison_edge_index, poison_edge_weights, rtn_mid=True)
                loss_inner = F.cross_entropy(
                    output[torch.cat([train_idx, attach_idx])], labels[torch.cat([train_idx, attach_idx])]
                )  # add our adaptive loss
                loss_inner.backward()
                optimizer_shadow.step()
                ood_x = torch.cat([selected_features, trojan_feat]).detach()

                # optimize ood detector
                for k in range(self.k):
                    optimizer_detector.zero_grad()
                    output_detector = self.ood_detector(ood_x)
                    ood_labels = torch.cat([torch.ones(len(trojan_feat), device=self.device), torch.zeros(len(trojan_feat), device=self.device)])
                    num_to_select = len(trojan_feat)
                    random_indices = torch.randperm(len(output_detector) - len(trojan_feat))[:num_to_select]
                    concatenated_tensors = torch.cat(
                        (output_detector[:-len(trojan_feat)][random_indices], output_detector[-len(trojan_feat):]), dim=0
                    )
                    loss_detector = F.cross_entropy(concatenated_tensors, ood_labels.long())
                    loss_detector.backward()
                    optimizer_detector.step()

            # optimize trigger generator
            acc_train_clean = accuracy(output[train_idx], labels[train_idx])
            acc_train_attach = accuracy(output[attach_idx], labels[attach_idx])
            optimizer_trigger.zero_grad()
            rs = np.random.RandomState()
            # select target nodes for triggers
            if self.outter_size <= len(unlabeled_idx):
                idx_outter = torch.cat([attach_idx, unlabeled_idx[rs.choice(len(unlabeled_idx), size=4096, replace=False)]])
            else:
                idx_outter = torch.cat([attach_idx, unlabeled_idx])
            trojan_feat, trojan_weights = self.trojan(features[idx_outter], self.thrd)  # may revise the process of generate
            if self.dataset == 'ogbn-arxiv':
                trojan_weights = torch.cat(
                    [trojan_weights, torch.ones([len(trojan_feat), 1 * self.trigger_size], dtype=torch.float, device=self.device)], dim=1)
            else:
                trojan_weights = torch.cat([torch.ones([len(idx_outter), 1], dtype=torch.float, device=self.device), trojan_weights], dim=1)
            trojan_weights = trojan_weights.flatten()
            trojan_feat = trojan_feat.view([-1, features.shape[1]])
            if self.dataset == 'ogbn-arxiv':
                trojan_edge = self.get_trojan_edge_arxiv(len(features), idx_outter, self.trigger_size).to(self.device)
            else:
                trojan_edge = self.get_trojan_edge(len(features), idx_outter, self.trigger_size).to(self.device)
            update_edge_weights = torch.cat([edge_weight, trojan_weights, trojan_weights])
            update_feat = torch.cat([features, trojan_feat])
            update_edge_index = torch.cat([edge_index, trojan_edge], dim=1)

            output, all_features = self.shadow_model(update_feat, update_edge_index, update_edge_weights, rtn_mid=True)
            output_detector = self.ood_detector(update_feat)

            labels_outter = labels.clone()
            labels_outter[idx_outter] = self.target_class

            probabilities = torch.exp(output[idx_outter])
            probabilities_target = probabilities[:, self.target_class]
            weights = torch.exp(-probabilities_target)
            weights = weights.detach()

            # This will be a tensor of the same shape as your batch
            losses = F.cross_entropy(output[idx_outter], labels_outter[idx_outter], reduction='none')
            loss_target = losses * (weights + 1)
            loss_target = loss_target.mean()
            sim = self.con_loss(all_features[idx_outter], all_features[train_idx], labels[train_idx], output[train_idx])
            loss_target_class = sim - self.simi(all_features[idx_outter], all_features[idx_outter]).fill_diagonal_(0).mean()
            loss_dis = F.cross_entropy(output_detector[-len(trojan_feat):], torch.ones(len(trojan_feat), device=self.device).long())

            loss_outter = loss_target * self.target_weight
            loss_outter += loss_dis * self.ood_weight
            loss_outter += loss_target_class * self.target_class_weight
            loss_outter.backward()
            optimizer_trigger.step()
            acc_train_outter = (output[idx_outter].argmax(dim=1) == self.target_class).float().mean()

            if loss_outter < loss_best:
                best_state_dict = deepcopy(self.trojan.state_dict())
                loss_best = float(loss_outter)

            if i % 50 == 0:
                logger.info(
                    'Epoch {}, loss_inner: {:.5f}, loss_target: {:.5f},  loss_dis: {:.5f}, loss_diff: {:.5f}, loss_adv: {:.5f}, loss_target_class: {:.5f}, ood_score: {:.5f} ' \
                        .format(i, loss_inner, loss_target, loss_dis, loss_dis, loss_target, loss_target_class,
                                torch.exp(output_detector[-len(trojan_feat):][:, -1:]).mean()))
                logger.info("acc_train_clean: {:.4f}, ASR_train_attach: {:.4f}, ASR_train_outter: {:.4f}" \
                      .format(acc_train_clean, acc_train_attach, acc_train_outter))
        self.trojan.eval()
        self.trojan.load_state_dict(best_state_dict)

    @torch.no_grad()
    def get_poisoned(self, features, edge_index, edge_weight, labels, attach_idx):

        if edge_weight is None:
            edge_weight = torch.ones([edge_index.shape[1]], device=self.device, dtype=torch.float32)

        if self.dataset == 'ogbn-arxiv':
            poison_x, poison_edge_index, poison_edge_weights = self.inject_trigger_arxiv(attach_idx, features,
                                                                                         edge_index, edge_weight)
        else:
            poison_x, poison_edge_index, poison_edge_weights = self.inject_trigger(attach_idx, features, edge_index, edge_weight)

        poison_labels = labels.clone()
        poison_labels[attach_idx] = self.target_class
        poison_edge_index = poison_edge_index[:, poison_edge_weights > 0.0]
        poison_edge_weights = poison_edge_weights[poison_edge_weights > 0.0]

        return poison_x, poison_edge_index, poison_edge_weights, poison_labels
