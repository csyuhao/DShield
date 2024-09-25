import time
import copy
import logging
import torch
import argparse
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import Planetoid, Flickr
from torch_geometric.utils import to_undirected, k_hop_subgraph

from attack.explain_backdoor import ExplainBackdoor
from attack.gcba import GCBA
from attack.gta import GTA
from attack.nba import LGCBackdoor, FGBackdoor
from attack.sba import SBA
from attack.adada import AdaDA
from attack.adaca import AdaCA
from attack.percba import PerCBA
from attack.target_node_attack import TargetNodeAttack
from attack.trap import TRAP
from attack.ugba import UGBA
from attack.dpgba import DPGBA
from attack.mlgb import MLGB

from defense.dshield import dshield, model_test

import heuristic_selection as hs
from models.construct import model_construct

from analysis.study import visualize_embedding
from pretraining.GraphCL import pretrain

from utils import get_split, subgraph, seed_experiment, ColumnNormalizeFeatures, calc_adjusted_homophily


logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def get_dataset(dataset_name):
    logger.info('Loading {} Dataset'.format(dataset_name))
    dataset = None
    if dataset_name == 'Cora' or dataset_name == 'Citeseer' or dataset_name == 'Pubmed':
        dataset = Planetoid(root='D:/Project/AttackOfGNNs/Dataset/', name=dataset_name)
    elif dataset_name == 'Flickr':
        transform = T.Compose([ColumnNormalizeFeatures(['x'])])
        dataset = Flickr(root='D:/Project/AttackOfGNNs/Dataset/Flickr/', transform=transform)
    elif dataset_name == 'ogbn-arxiv':
        # Download and process data at './dataset/ogbn_molhiv/'
        transform = T.Compose([ColumnNormalizeFeatures(['x'])])
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='D:/Project/AttackOfGNNs/Dataset/')

    graph = dataset[0].to(device)

    if args.dataset == 'ogbn-arxiv':
        num_nodes = graph.x.shape[0]
        setattr(graph, 'train_mask', torch.zeros(num_nodes, dtype=torch.bool).to(device))
        graph.val_mask = torch.zeros(num_nodes, dtype=torch.bool).to(device)
        graph.test_mask = torch.zeros(num_nodes, dtype=torch.bool).to(device)
        graph.y = graph.y.squeeze(1)
    return graph, graph.y.max().item() + 1


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1027, help='Random seed.')

    # GPU setting
    parser.add_argument('--device_id', type=int, default=0, help="Threshold of pruning edges")
    parser.add_argument('--instance', type=str, default='Attack', help='the instance name of wandb')
    parser.add_argument('--wandb_group', type=str, default='GraphBackdoor', help='the group name of wandb')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training.')

    # benign settings
    parser.add_argument('--model', type=str, default='GCN', help='model', choices=['GCN', 'GAT', 'GraphSage', 'GIN'])
    parser.add_argument('--dataset', type=str, default='ogbn-arxiv', help='Dataset',
                        choices=['Cora', 'Citeseer', 'Pubmed', 'Flickr', 'ogbn-arxiv'])
    parser.add_argument('--benign_lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--benign_weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--benign_hidden', type=int, default=32, help='Number of hidden units.')
    parser.add_argument('--benign_dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--benign_epochs', type=int, default=200, help='Number of epochs to train benign and backdoor model.')

    # backdoor settings
    parser.add_argument('--target_class', type=str, default=0)
    parser.add_argument('--trigger_size', type=int, default=3, help='trigger_size')
    parser.add_argument('--vs_ratio', type=float, default=0, help="ratio of poisoning nodes relative to the full graph")
    parser.add_argument('--vs_number', type=int, default=0, help="number of poisoning nodes relative to the full graph")
    parser.add_argument('--use_vs_number', action='store_true', default=False, help="if use detailed number to decide Vs")
    parser.add_argument('--attack_method', type=str, default="none",
                        choices=['none', 'UGBA', 'SBA', 'GTA', 'ExplainBackdoor',
                                 'LGCB', 'GB-FGSM', 'GCBA', 'TRAP', 'PerCBA', 'AdaDA',
                                 'AdaCA', 'UGBA-LGCB', 'UGBA-GCBA', 'GCBA-PerCBA', 'TargetNodeAttack', 'DPGBA', 'MLGB'], help="defense method")
    parser.add_argument('--dis_weight', type=float, default=1, help="Weight of cluster distance")  # cluster degree to pick up nodes
    parser.add_argument('--selection_method', type=str, default='none',
                        choices=['loss', 'conf', 'cluster', 'none', 'cluster_degree', 'clean_label', 'mixture'],
                        help='Method to select idx_attach for training trojan model (none means randomly select)')

    # UGBA attack
    parser.add_argument('--ugba_thrd', type=float, default=0.5)
    parser.add_argument('--ugba_lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--ugba_trojan_epochs', type=int, default=400, help='Number of epochs to train trigger generator.')
    parser.add_argument('--ugba_inner_epochs', type=int, default=1, help='Number of inner')
    parser.add_argument('--ugba_target_loss_weight', type=float, default=1, help="Weight of optimize outer trigger generator")
    parser.add_argument('--ugba_homo_loss_weight', type=float, default=100, help="Weight of optimize similarity loss")
    parser.add_argument('--ugba_homo_boost_thrd', type=float, default=0.8, help="Threshold of increase similarity")

    # SBA attack
    parser.add_argument('--sba_attack_method', type=str, default='Rand_Gene', choices=['Rand_Gene', 'Rand_Samp', 'Basic', 'None'],
                        help='Method to select idx_attach for training trojan model (none means randomly select)')
    parser.add_argument('--sba_trigger_prob', type=float, default=0.5,
                        help="The probability to generate the trigger's edges in random method")
    parser.add_argument('--sba_selection_method', type=str, default='none',
                        choices=['loss', 'conf', 'cluster', 'none', 'cluster_degree'],
                        help='Method to select idx_attach for training trojan model (none means randomly select)')

    # GTA attack
    parser.add_argument('--gta_thrd', type=float, default=0.5)
    parser.add_argument('--gta_lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--gta_trojan_epochs', type=int, default=400, help='Number of epochs to train trigger generator.')
    parser.add_argument('--gta_loss_factor', type=float, default=1, help='loss balance factor')
    parser.add_argument('--gta_selection_method', type=str, default='none',
                        choices=['loss', 'conf', 'cluster', 'none', 'cluster_degree'],
                        help='Method to select idx_attach for training trojan model (none means randomly select)')

    # GCBA attack
    parser.add_argument('--gcba_num_hidden', type=int, default=512, help='Dimension of hidden vectors')
    parser.add_argument('--gcba_feat_budget', type=int, default=512, help='Feature budget')
    parser.add_argument('--gcba_trojan_epochs', type=int, default=300, help='Epochs')
    parser.add_argument('--gcba_ssl_tau', type=float, default=0.8, help='InfoNCE Loss')
    parser.add_argument('--gcba_tau', type=float, default=0.2, help='Sampling features')
    parser.add_argument('--gcba_edge_drop_ratio', type=float, default=0.5, help='Feature drop rate')

    # ExplainBackdoor
    parser.add_argument('--eb_trig_feat_val', type=float, default=0.0)
    parser.add_argument('--eb_trig_feat_wid', type=int, default=10)
    parser.add_argument('--eb_selection_method', type=str, default='none',
                        choices=['loss', 'conf', 'cluster', 'none', 'cluster_degree'],
                        help='Method to select idx_attach for training trojan model (none means randomly select)')

    # LGCB
    parser.add_argument('--lgcb_num_budgets', type=int, default=10)

    # GB-FGSM
    parser.add_argument('--fg_num_budgets', type=int, default=10)
    parser.add_argument('--fg_tau', type=float, default=1.0)

    # TRAP attack
    parser.add_argument('--trap_lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--trap_trojan_epochs', type=int, default=300, help='Epochs')

    # MLGB attack
    parser.add_argument('--mlgb_trigger_dim', type=int, default=100, help='Trigger dimension')
    parser.add_argument('--mlgb_trojan_epochs', type=int, default=400, help='Epochs')
    parser.add_argument('--mlgb_inner_epochs', type=int, default=1, help='Number of inner')
    parser.add_argument('--mlgb_lr', type=float, default=0.01, help='Initial learning rate.')

    # PerCBA attack
    parser.add_argument('--percba_lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--percba_trojan_epochs', type=int, default=300, help='Epochs')
    parser.add_argument('--percba_mu', type=float, default=0.01, help='Step size')
    parser.add_argument('--percba_eps', type=float, default=0.1, help='Threshold of Perturbation')
    parser.add_argument('--percba_feat_budget', type=int, default=50, help='Feature budget')
    parser.add_argument('--percba_perturb_epochs', type=int, default=50, help='Perturbation rounds')

    # DPGBA attack
    parser.add_argument('--dpgba_thrd', type=float, default=0.5)
    parser.add_argument('--dpgba_lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--dpgba_trojan_epochs', type=int, default=400, help='Number of epochs to train trigger generator.')
    parser.add_argument('--dpgba_inner_epochs', type=int, default=1, help='Number of inner')
    parser.add_argument('--dpgba_rec_epochs', type=int, default=100, help='Number of epochs to train benign and backdoor model.')
    parser.add_argument('--dpgba_k', type=int, default=100, help='Number of reconstruction')
    parser.add_argument('--dpgba_outter_size', type=int, default=4096, help='Weight of optimize outter trigger generator')
    parser.add_argument('--dpgba_target_weight', type=float, default=1, help="Weight of attack loss")
    parser.add_argument('--dpgba_ood_weight', type=float, default=1, help="Weight of ood constraint")
    parser.add_argument('--dpgba_target_class_weight', type=float, default=1, help="Weight of enhancing attack loss")

    # AdaDA attack
    parser.add_argument('--adaba_thrd', type=float, default=0.5)
    parser.add_argument('--adaba_lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--adaba_trojan_epochs', type=int, default=400, help='Number of epochs to train trigger generator.')
    parser.add_argument('--adaba_inner_epochs', type=int, default=1, help='Number of inner')
    parser.add_argument('--adaba_target_loss_weight', type=float, default=1, help="Weight of optimize outer trigger generator")
    parser.add_argument('--adaba_homo_loss_weight', type=float, default=100, help="Weight of optimize similarity loss")
    parser.add_argument('--adaba_homo_boost_thrd', type=float, default=0.8, help="Threshold of increase similarity")
    parser.add_argument('--adaba_reg_loss_weight', type=float, default=0.8, help="Threshold of increase similarity in hidden space")
    parser.add_argument('--adaba_ssl_tau', type=float, default=0.2, help='InfoNCE Loss')
    parser.add_argument('--adaba_edge_drop_ratio', type=float, default=0.5, help='Feature drop rate')
    parser.add_argument('--adaba_non_training', action='store_true', default=False, help='Disable training')

    # AdaCA attack
    parser.add_argument('--adaca_num_hidden', type=int, default=512, help='Dimension of hidden vectors')
    parser.add_argument('--adaca_feat_budget', type=int, default=512, help='Feature budget')
    parser.add_argument('--adaca_trojan_epochs', type=int, default=300, help='Epochs')
    parser.add_argument('--adaca_ssl_tau', type=float, default=0.8, help='InfoNCE Loss')
    parser.add_argument('--adaca_tau', type=float, default=0.2, help='Sampling features')
    parser.add_argument('--adaca_edge_drop_ratio', type=float, default=0.5, help='Feature drop rate')
    parser.add_argument('--adaca_reg_loss_weight', type=float, default=0.8, help="Threshold of increase similarity in hidden space")
    parser.add_argument('--adaca_umap_epochs', type=int, default=10, help="UMAP epochs")
    parser.add_argument('--adaca_non_training', action='store_true', default=False, help='Disable training')

    parser.add_argument('--target_node_trigger_type', type=str, default='renyi', choices=['renyi', 'ws', 'ba'],
                        help='generate the trigger methods')
    parser.add_argument('--target_node_density', type=float, default=0.8, help='density of the edge in the generated trigger')
    parser.add_argument('--target_node_degree', type=int, default=3, help='The degree of trigger type')

    # defense setting
    parser.add_argument('--defense_method', type=str, default="prune", choices=['none', 'DShield'], help="defense method")

    # ModelCleanse
    parser.add_argument('--dshield_pretrain_epochs', type=int, default=1000, help='SSL pretrain epochs')
    parser.add_argument('--dshield_finetune_epochs', type=int, default=500, help='SSL finetune epochs')
    parser.add_argument('--dshield_classify_epochs', type=int, default=500, help='Classify epochs')
    parser.add_argument('--dshield_kappa1', type=float, default=5, help='Loss balance factor')
    parser.add_argument('--dshield_kappa2', type=float, default=5, help='Loss balance factor')
    parser.add_argument('--dshield_kappa3', type=float, default=0.5, help='Loss balance factor')
    parser.add_argument('--dshield_edge_drop_ratio', type=float, default=0.30, help='probability to drop edges')
    parser.add_argument('--dshield_feature_drop_ratio', type=float, default=0.30, help='probability to drop attributes')
    parser.add_argument('--dshield_tau', type=float, default=0.40, help='Temperature factor')
    parser.add_argument('--dshield_balance_factor', type=float, default=0.50, help='Balance factor')
    parser.add_argument('--dshield_classify_rounds', type=int, default=100, help='Number of rounds')
    parser.add_argument('--dshield_thresh', type=float, default=3.5, help='MAD threshold')

    args = parser.parse_known_args()[0]

    # GPU Settings
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = ('cuda:{}' if torch.cuda.is_available() and args.cuda else 'cpu').format(args.device_id)

    # Deterministic Running
    seed_experiment(args.seed)

    # Dataset Settings
    data, num_class = get_dataset(args.dataset)

    # Calculate homogeneity degree
    homophily = calc_adjusted_homophily(data.edge_index, data.y)
    logger.info('{} Dataset homophily = {:.4f}'.format(args.dataset, homophily.item()))

    feat_dim = data.x.shape[1]
    data, train_idx, val_idx, clean_test_idx, atk_idx = get_split(data, device)
    data.edge_index = to_undirected(data.edge_index)
    train_edge_index, _, edge_mask = subgraph(
        subset=torch.bitwise_not(data.test_mask), edge_index=data.edge_index, edge_attr=None, relabel_nodes=False
    )
    mask_edge_index = data.edge_index[:, torch.bitwise_not(edge_mask)]
    # filter out the unlabeled nodes except from training nodes and testing nodes, nonzero() is to get index, flatten is to get 1-d tensor
    unlabeled_idx = (torch.bitwise_not(data.test_mask) & torch.bitwise_not(data.train_mask)).nonzero().flatten()
    if args.use_vs_number:
        size = args.vs_number
    else:
        size = int((len(data.test_mask) - data.test_mask.sum()) * args.vs_ratio)
    logger.info("# Attach Nodes:{}".format(size))

    if '-' not in args.target_class:
        args.target_class = int(args.target_class)
    attach_idx = None
    # here is randomly select poison nodes from unlabeled nodes
    if args.selection_method == 'none':
        attach_idx = hs.obtain_attach_nodes(args, unlabeled_idx, size)
    elif args.selection_method == 'cluster':
        attach_idx = hs.cluster_distance_selection(
            args, data, train_idx, val_idx, clean_test_idx, unlabeled_idx, train_edge_index, size, device
        )
        attach_idx = torch.LongTensor(attach_idx).to(device)
    elif args.selection_method == 'cluster_degree':
        if args.dataset == 'Pubmed':
            attach_idx = hs.cluster_degree_selection_separate_fixed(
                args, data, train_idx, val_idx, clean_test_idx, unlabeled_idx, train_edge_index, size, device
            )
        else:
            attach_idx = hs.cluster_degree_selection(
                args, data, train_idx, val_idx, clean_test_idx, unlabeled_idx, train_edge_index, size, device
            )
        attach_idx = torch.LongTensor(attach_idx).to(device)
    elif args.selection_method == 'clean_label':
        attach_idx = hs.obtain_clean_label_attach_nodes(args, unlabeled_idx, data.y, size, args.target_class)
    elif args.selection_method == 'mixture':
        attack_methods = list(args.attack_method.split('-'))
        target_classes = [int(i) for i in args.target_class.split('-')]
        num_attack_methods = len(attack_methods)
        list_attach_idx = []
        for idx, attack_method in enumerate(attack_methods):
            clean_label = False
            if attack_method in ('GCBA', 'PerCBA'):
                clean_label = True

            if clean_label is True:
                cur_attach_idx = hs.obtain_clean_label_attach_nodes(args, unlabeled_idx, data.y, size // num_attack_methods, target_classes[idx])
            else:
                args.target_class = target_classes[idx]
                if args.dataset == 'Pubmed':
                    cur_attach_idx = hs.cluster_degree_selection_separate_fixed(
                        args, data, train_idx, val_idx, clean_test_idx, unlabeled_idx, train_edge_index, size // num_attack_methods, device
                    )
                else:
                    cur_attach_idx = hs.cluster_degree_selection(
                        args, data, train_idx, val_idx, clean_test_idx, unlabeled_idx, train_edge_index, size // num_attack_methods, device
                    )
                cur_attach_idx = torch.LongTensor(cur_attach_idx).to(device)
            list_attach_idx.append(cur_attach_idx)
        args.target_class = '-'.join([str(i) for i in target_classes])
        attach_idx = torch.cat(list_attach_idx, dim=0)
    unlabeled_idx = torch.tensor(list(set(unlabeled_idx.cpu().numpy()) - set(attach_idx.cpu().numpy()))).to(device)

    # Train Trigger Generator
    backdoor_gen_model = None
    feat, train_edge_weight, labels, train_node_idx, = None, None, None, None
    if args.attack_method == 'none':
        feat, train_edge_index, labels = data.x, train_edge_index, data.y
        train_edge_weight = torch.ones([train_edge_index.shape[1]], device=device, dtype=torch.float)
        train_node_idx = train_idx
    elif args.attack_method == 'UGBA':
        backdoor_gen_model = UGBA(
            seed=args.seed, thrd=args.ugba_thrd, hidden=args.benign_hidden, trojan_epochs=args.ugba_trojan_epochs,
            inner_epochs=args.ugba_inner_epochs, lr=args.ugba_lr, weight_decay=args.benign_weight_decay,
            target_loss_weight=args.ugba_target_loss_weight, homo_loss_weight=args.ugba_homo_loss_weight,
            homo_boost_thrd=args.ugba_homo_boost_thrd, trigger_size=args.trigger_size, target_class=args.target_class, device=device
        )
        backdoor_gen_model.fit(data.x, train_edge_index, None, data.y, train_idx, attach_idx, unlabeled_idx)
        feat, train_edge_index, train_edge_weight, labels = backdoor_gen_model.get_poisoned(
            data.x, train_edge_index, None, data.y, attach_idx
        )
        train_node_idx = torch.cat([train_idx, attach_idx]).to(device)
    elif args.attack_method == 'SBA':
        backdoor_gen_model = SBA(
            features=data.x, seed=args.seed, attack_method=args.sba_attack_method,
            trigger_prob=args.sba_trigger_prob, trigger_size=args.trigger_size, target_class=args.target_class, dataset=args.dataset, device=device
        )

        feat, train_edge_index, train_edge_weight, labels = backdoor_gen_model.get_poisoned_rand(
            data.x, train_edge_index, data.y, attach_idx
        )
        train_node_idx = torch.cat([train_idx, attach_idx]).to(device)
    elif args.attack_method == 'GTA':
        backdoor_gen_model = GTA(
            thrd=args.gta_thrd, hidden=args.benign_hidden, trojan_epochs=args.gta_trojan_epochs,
            loss_factor=args.gta_loss_factor, lr=args.gta_lr, weight_decay=args.benign_weight_decay,
            trigger_size=args.trigger_size, target_class=args.target_class, device=device
        )
        backdoor_gen_model.fit(data.x, train_edge_index, None, data.y, train_idx, attach_idx, unlabeled_idx)
        feat, train_edge_index, train_edge_weight, labels = backdoor_gen_model.get_poisoned(
            data.x, train_edge_index, None, data.y, attach_idx
        )
        train_node_idx = torch.cat([train_idx, attach_idx]).to(device)
    elif args.attack_method == 'ExplainBackdoor':
        backdoor_gen_model = ExplainBackdoor(
            trig_feat_val=args.eb_trig_feat_val, trig_feat_wid=args.eb_trig_feat_wid,
            hidden=args.benign_hidden, epochs=args.benign_epochs, target_class=args.target_class, device=device
        )
        backdoor_gen_model.fit(data.x, train_edge_index, None, data.y, train_idx, attach_idx, unlabeled_idx)
        feat, train_edge_index, train_edge_weight, labels = backdoor_gen_model.get_poisoned(
            data.x, train_edge_index, data.y, attach_idx
        )
        train_node_idx = torch.cat([train_idx, attach_idx]).to(device)
    elif args.attack_method == 'LGCB':
        # Train models
        backdoor_gen_model = LGCBackdoor(
            num_budgets=args.lgcb_num_budgets, hidden=args.benign_hidden,
            epochs=args.benign_epochs, target_class=args.target_class, device=device
        )
        backdoor_gen_model.build_shadow_model(data.x, train_edge_index, None, data.y, train_idx)
        feat, train_edge_index, train_edge_weight, labels = backdoor_gen_model.get_poisoned(
            data.x, train_edge_index, None, data.y, attach_idx
        )
        train_node_idx = torch.cat([train_idx, attach_idx]).to(device)
    elif args.attack_method == 'GB-FGSM':
        # Train models
        backdoor_gen_model = FGBackdoor(
            num_budgets=args.fg_num_budgets, hidden=args.benign_hidden,
            epochs=args.benign_epochs, target_class=args.target_class, device=device
        )
        backdoor_gen_model.build_shadow_model(data.x, train_edge_index, None, data.y, train_idx, args.fg_tau)
        feat, train_edge_index, train_edge_weight, labels = backdoor_gen_model.get_poisoned(
            data.x, train_edge_index, None, data.y, attach_idx, device
        )
        train_node_idx = torch.cat([train_idx, attach_idx]).to(device)
    elif args.attack_method == 'GCBA':
        backdoor_gen_model = GCBA(
            num_feat=data.x.shape[1], num_hidden=args.gcba_num_hidden, num_labels=torch.max(data.y).item() + 1,
            feat_budget=args.gcba_feat_budget, trojan_epochs=args.gcba_trojan_epochs, ssl_tau=args.gcba_ssl_tau, target_class=args.target_class,
            tau=args.gcba_tau, lr=args.benign_lr, weight_decay=args.benign_weight_decay, edge_drop_ratio=args.gcba_edge_drop_ratio, device=device
        )
        backdoor_gen_model.fit(data.x, train_edge_index, None, data.y, train_idx, attach_idx, dataset=args.dataset)
        feat, train_edge_index, train_edge_weight, labels, _ = backdoor_gen_model.get_poisoned(
            data.x, train_edge_index, None, data.y, attach_idx
        )
        train_node_idx = torch.cat([train_idx, attach_idx]).to(device)
    elif args.attack_method == 'TRAP':
        backdoor_gen_model = TRAP(
            hidden=args.benign_hidden, trojan_epochs=args.trap_trojan_epochs, lr=args.trap_lr,
            weight_decay=args.benign_weight_decay, trigger_size=args.trigger_size, target_class=args.target_class, device=device
        )
        backdoor_gen_model.fit(data.x, train_edge_index, None, data.y, train_idx, attach_idx, unlabeled_idx)
        feat, train_edge_index, train_edge_weight, labels = backdoor_gen_model.get_poisoned(
            data.x, train_edge_index, None, data.y, attach_idx
        )
        train_node_idx = torch.cat([train_idx, attach_idx]).to(device)
    elif args.attack_method == 'PerCBA':
        backdoor_gen_model = PerCBA(
            mu=args.percba_mu, eps=args.percba_eps,
            hidden=args.benign_hidden, trojan_epochs=args.trap_trojan_epochs, perturb_epochs=args.percba_perturb_epochs, lr=args.trap_lr,
            weight_decay=args.benign_weight_decay, feat_budget=args.percba_feat_budget, target_class=args.target_class, device=device
        )
        backdoor_gen_model.fit(data.x, train_edge_index, None, data.y, train_idx, attach_idx, unlabeled_idx)
        feat, train_edge_index, train_edge_weight, labels = backdoor_gen_model.get_poisoned(
            data.x, train_edge_index, None, data.y, attach_idx
        )
        train_node_idx = torch.cat([train_idx, attach_idx]).to(device)
    elif args.attack_method == 'AdaDA':
        if args.adaba_non_training:
            # Training backdoor generation model
            backdoor_gen_model = AdaDA(
                seed=args.seed, thrd=args.adaba_thrd, hidden=args.benign_hidden, trojan_epochs=args.adaba_trojan_epochs,
                inner_epochs=args.adaba_inner_epochs, lr=args.adaba_lr, weight_decay=args.benign_weight_decay,
                target_loss_weight=args.adaba_target_loss_weight, homo_loss_weight=args.adaba_homo_loss_weight,
                reg_loss_weight=args.adaba_reg_loss_weight, homo_boost_thrd=args.adaba_homo_boost_thrd, trigger_size=args.trigger_size,
                target_class=args.target_class, edge_drop_ratio=args.adaba_edge_drop_ratio, ssl_tau=args.adaba_ssl_tau, dataset=args.dataset,
                device=device
            )
            backdoor_gen_model.fit(data.x, train_edge_index, None, data.y, train_idx, attach_idx, unlabeled_idx)

            feat, train_edge_index, labels = data.x, train_edge_index, data.y
            train_edge_weight = torch.ones([train_edge_index.shape[1]], device=device, dtype=torch.float)
            train_node_idx = train_idx
        else:
            backdoor_gen_model = AdaDA(
                seed=args.seed, thrd=args.adaba_thrd, hidden=args.benign_hidden, trojan_epochs=args.adaba_trojan_epochs,
                inner_epochs=args.adaba_inner_epochs, lr=args.adaba_lr, weight_decay=args.benign_weight_decay,
                target_loss_weight=args.adaba_target_loss_weight, homo_loss_weight=args.adaba_homo_loss_weight,
                reg_loss_weight=args.adaba_reg_loss_weight, homo_boost_thrd=args.adaba_homo_boost_thrd, trigger_size=args.trigger_size,
                target_class=args.target_class, edge_drop_ratio=args.adaba_edge_drop_ratio, ssl_tau=args.adaba_ssl_tau, dataset=args.dataset,
                device=device
            )
            backdoor_gen_model.fit(data.x, train_edge_index, None, data.y, train_idx, attach_idx, unlabeled_idx)
            feat, train_edge_index, train_edge_weight, labels = backdoor_gen_model.get_poisoned(
                data.x, train_edge_index, None, data.y, attach_idx
            )
            train_node_idx = torch.cat([train_idx, attach_idx]).to(device)
    elif args.attack_method == 'AdaCA':
        if args.adaca_non_training:

            # Training backdoor generation model
            backdoor_gen_model = AdaCA(
                num_feat=data.x.shape[1], num_hidden=args.adaca_num_hidden, num_labels=torch.max(data.y).item() + 1,
                feat_budget=args.adaca_feat_budget, trojan_epochs=args.adaca_trojan_epochs, umap_epochs=args.adaca_umap_epochs,
                ssl_tau=args.adaca_ssl_tau, tau=args.adaca_tau, lr=args.benign_lr, weight_decay=args.benign_weight_decay,
                reg_weight=args.adaca_reg_loss_weight, edge_drop_ratio=args.adaca_edge_drop_ratio, target_class=args.target_class, device=device
            )
            backdoor_gen_model.fit(data.x, train_edge_index, None, data.y, train_idx, attach_idx, dataset=args.dataset)

            feat, train_edge_index, labels = data.x, train_edge_index, data.y
            train_edge_weight = torch.ones([train_edge_index.shape[1]], device=device, dtype=torch.float)
            train_node_idx = train_idx
        else:
            backdoor_gen_model = AdaCA(
                num_feat=data.x.shape[1], num_hidden=args.adaca_num_hidden, num_labels=torch.max(data.y).item() + 1,
                feat_budget=args.adaca_feat_budget, trojan_epochs=args.adaca_trojan_epochs, umap_epochs=args.adaca_umap_epochs,
                ssl_tau=args.adaca_ssl_tau, tau=args.adaca_tau, lr=args.benign_lr, weight_decay=args.benign_weight_decay,
                reg_weight=args.adaca_reg_loss_weight, edge_drop_ratio=args.adaca_edge_drop_ratio, target_class=args.target_class, device=device
            )
            backdoor_gen_model.fit(data.x, train_edge_index, None, data.y, train_idx, attach_idx, dataset=args.dataset)
            feat, train_edge_index, train_edge_weight, labels, _ = backdoor_gen_model.get_poisoned(
                data.x, train_edge_index, None, data.y, attach_idx
            )
            train_node_idx = torch.cat([train_idx, attach_idx]).to(device)
    elif args.attack_method == 'TargetNodeAttack':
        backdoor_gen_model = TargetNodeAttack(target_class=args.target_class, trigger_size=args.trigger_size,
                                              trigger_type=args.target_node_trigger_type, density=args.target_node_density,
                                              degree=args.target_node_degree, device=device)

        feat, train_edge_index, train_edge_weight, labels = backdoor_gen_model.get_poisoned(
            data.x, train_edge_index, None, data.y, attach_idx
        )
        train_node_idx = torch.cat([train_idx, attach_idx]).to(device)
    elif args.attack_method == 'DPGBA':
        backdoor_gen_model = DPGBA(
            seed=args.seed, thrd=args.dpgba_thrd, hidden=args.benign_hidden, trojan_epochs=args.dpgba_trojan_epochs,
            inner_epochs=args.dpgba_inner_epochs, rec_epochs=args.dpgba_rec_epochs, k=args.dpgba_k, outter_size=args.dpgba_outter_size,
            lr=args.dpgba_lr, weight_decay=args.benign_weight_decay, target_weight=args.dpgba_target_weight, ood_weight=args.dpgba_ood_weight,
            target_class_weight=args.dpgba_target_class_weight, trigger_size=args.trigger_size, target_class=args.target_class,
            dataset_name=args.dataset, device=device
        )
        backdoor_gen_model.fit(data.x, train_edge_index, None, data.y, train_idx, attach_idx, unlabeled_idx)
        feat, train_edge_index, train_edge_weight, labels = backdoor_gen_model.get_poisoned(
            data.x, train_edge_index, None, data.y, attach_idx
        )
        train_node_idx = torch.cat([train_idx, attach_idx]).to(device)
    elif args.attack_method == 'MLGB':
        backdoor_gen_model = MLGB(
            seed=args.seed, trigger_dim=args.mlgb_trigger_dim, trojan_epochs=args.mlgb_trojan_epochs,
            inner_epochs=args.mlgb_inner_epochs, lr=args.mlgb_lr, weight_decay=args.benign_weight_decay,
            num_classes=num_class, hidden=args.benign_hidden, epochs=args.benign_epochs, target_class=args.target_class, device=device
        )
        backdoor_gen_model.fit(data.x, train_edge_index, None, data.y, train_idx, attach_idx, unlabeled_idx)
        feat, train_edge_index, train_edge_weight, labels = backdoor_gen_model.get_poisoned(
            data.x, train_edge_index, None, data.y, attach_idx
        )
        train_node_idx = torch.cat([train_idx, attach_idx]).to(device)
    elif '-' in args.attack_method:
        attack_methods = list(args.attack_method.split('-'))
        num_attack_methods = len(attack_methods)
        num_attach_idx = attach_idx.shape[0]

        cur_train_idx = train_idx.clone()
        target_classes = [int(i) for i in args.target_class.split('-')]
        cur_feat, cur_train_edge_index, cur_train_edge_weight, cur_labels = data.x.clone(), train_edge_index.clone(), None, data.y.clone()
        backdoor_gen_model = []
        for idx, attack_method in enumerate(attack_methods):
            cur_attach_idx = attach_idx[num_attach_idx // num_attack_methods * idx: num_attach_idx // num_attack_methods * (idx + 1)]
            if attack_method == 'UGBA':
                cur_backdoor_gen_model = UGBA(
                    seed=args.seed, thrd=args.ugba_thrd, hidden=args.benign_hidden, trojan_epochs=args.ugba_trojan_epochs,
                    inner_epochs=args.ugba_inner_epochs, lr=args.ugba_lr, weight_decay=args.benign_weight_decay,
                    target_loss_weight=args.ugba_target_loss_weight, homo_loss_weight=args.ugba_homo_loss_weight,
                    homo_boost_thrd=args.ugba_homo_boost_thrd, trigger_size=args.trigger_size, target_class=target_classes[idx], device=device
                )
                cur_backdoor_gen_model.fit(data.x, train_edge_index, None, data.y, train_idx, cur_attach_idx, unlabeled_idx)
                cur_feat, cur_train_edge_index, cur_train_edge_weight, cur_labels = cur_backdoor_gen_model.get_poisoned(
                    cur_feat, cur_train_edge_index, cur_train_edge_weight, cur_labels, cur_attach_idx
                )
            elif attack_method == 'LGCB':
                # Train models
                cur_backdoor_gen_model = LGCBackdoor(
                    num_budgets=args.lgcb_num_budgets, hidden=args.benign_hidden,
                    epochs=args.benign_epochs, target_class=target_classes[idx], device=device
                )
                cur_backdoor_gen_model.build_shadow_model(data.x, train_edge_index, None, data.y, train_idx)
                cur_feat, cur_train_edge_index, cur_train_edge_weight, cur_labels = cur_backdoor_gen_model.get_poisoned(
                    cur_feat, cur_train_edge_index, cur_train_edge_weight, cur_labels, cur_attach_idx
                )
            elif attack_method == 'GCBA':
                cur_backdoor_gen_model = GCBA(
                    num_feat=data.x.shape[1], num_hidden=args.gcba_num_hidden, num_labels=torch.max(data.y).item() + 1,
                    feat_budget=args.gcba_feat_budget, trojan_epochs=args.gcba_trojan_epochs, ssl_tau=args.gcba_ssl_tau,
                    target_class=target_classes[idx],
                    tau=args.gcba_tau, lr=args.benign_lr, weight_decay=args.benign_weight_decay, edge_drop_ratio=args.gcba_edge_drop_ratio,
                    device=device
                )
                cur_backdoor_gen_model.fit(data.x, train_edge_index, None, data.y, train_idx, cur_attach_idx, dataset=args.dataset)
                cur_feat, cur_train_edge_index, cur_train_edge_weight, cur_labels, _ = cur_backdoor_gen_model.get_poisoned(
                    cur_feat, cur_train_edge_index, cur_train_edge_weight, cur_labels, cur_attach_idx
                )

            elif attack_method == 'PerCBA':
                cur_backdoor_gen_model = PerCBA(
                    mu=args.percba_mu, eps=args.percba_eps,
                    hidden=args.benign_hidden, trojan_epochs=args.trap_trojan_epochs, perturb_epochs=args.percba_perturb_epochs, lr=args.trap_lr,
                    weight_decay=args.benign_weight_decay, feat_budget=args.percba_feat_budget, target_class=target_classes[idx], device=device
                )
                cur_backdoor_gen_model.fit(data.x, train_edge_index, None, data.y, train_idx, cur_attach_idx, unlabeled_idx)
                cur_feat, cur_train_edge_index, cur_train_edge_weight, cur_labels = cur_backdoor_gen_model.get_poisoned(
                    cur_feat, cur_train_edge_index, cur_train_edge_weight, cur_labels, cur_attach_idx
                )
            backdoor_gen_model.append(cur_backdoor_gen_model)

            cur_train_idx = torch.cat([cur_train_idx, cur_attach_idx]).to(device)
        feat, train_edge_index, train_edge_weight, labels, train_node_idx = cur_feat, cur_train_edge_index, cur_train_edge_weight, cur_labels, cur_train_idx
    else:
        raise Exception('Error!')

    
    # Visualization
    tmp_model = model_construct(
        dataset=args.dataset, model_name=args.model, feat_dim=feat_dim,
        num_class=num_class, hidden=args.benign_hidden, dropout=args.benign_dropout,
        lr=args.benign_lr, weight_decay=args.benign_weight_decay, device=device).to(device)
    tmp_model.fit(feat, train_edge_index, train_edge_weight,
                  labels, train_node_idx, val_idx, train_iters=args.benign_epochs, verbose=False)
    effect_idx = visualize_embedding(tmp_model, feat, train_edge_index, labels, train_node_idx, attach_idx,
                                    args.target_class, dataset_name=args.dataset, attack_name=args.attack_method)


    # SSL PreTraining
    if args.attack_method in ['UGBA', 'GTA', 'SBA', 'ExplainBackdoor', 'LGCB', 'GB-FGSM']:
        pretrain(feat, train_edge_index, labels, train_node_idx, attach_idx, effect_idx,
                args.benign_epochs, lr=args.gta_lr, weight_decay=args.benign_weight_decay,
                device=device, dataset_name=args.dataset, attack_name=args.attack_method)   
    elif args.attack_method in ['GCBA', 'PerCBA']:

        from defense.dshield import attribute_importance, vis_feat_importance
        import torch.nn.functional as F

        poisoned_model = model_construct(
            dataset=args.dataset, model_name=args.model, feat_dim=feat_dim,
            num_class=num_class, hidden=args.benign_hidden, dropout=args.benign_dropout,
            lr=args.benign_lr, weight_decay=args.benign_weight_decay, device=device
        ).to(device)
        poisoned_model.fit(feat, train_edge_index, train_edge_weight,
                           labels, train_node_idx, val_idx, train_iters=args.benign_epochs, verbose=False)
        num_classes = torch.max(labels).item() + 1

        # Get attribute importance
        _, attribute_import = attribute_importance(poisoned_model, feat, train_edge_index, train_edge_weight, labels)

        # Visualize feature importance
        vis_feat_importance(F.normalize(attribute_import * feat, dim=-1), labels,
                            attach_idx, num_classes, vis_node_idx=train_node_idx, dataset_name=args.dataset, attack_name=args.attack_method)
