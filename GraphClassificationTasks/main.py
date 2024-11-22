import random
import numpy as np
import argparse
import logging

import torch
import torch.optim as optim
import torch.nn as nn

from torch_geometric.loader import DataLoader

from attack.explain_backdoor import ExplainBackdoor
from dataset.utils import get_dataset, get_split, get_attach_idx
from models.utils import model_construct, model_test
from utils import seed_experiment
from attack.sba import SBA
from attack.gcba import GCBA
from attack.sbag import SBAG
from defense.dshield import dshield

try:
    if 'logger' not in globals():
        logging.basicConfig()
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
except NameError:
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)


def main():
    global args

    # GPU Settings
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = ('cuda:{}' if torch.cuda.is_available() and args.cuda else 'cpu').format(args.device_id)

    # Deterministic Running
    seed_experiment(args.seed)

    # Dataset Settings
    use_node_attr = False
    if args.attack_method == 'SBAG':
        use_node_attr = True
    dataset, num_features, num_classes = get_dataset(args.dataset, use_node_attr)
    if args.dataset in ['ENZYMES', 'PROTEINS']:
        num_node_attributes, num_node_labels = dataset.num_node_attributes, dataset.num_node_labels
    else:
        num_node_attributes, num_node_labels = 1, 10
    logger.info('The number of dataset graphs: {}, features: {}, classes: {}'.format(len(dataset), num_features, num_classes))
    train_dataset, clean_test_dataset, atk_test_dataset = get_split(dataset, test_size=0.2, seed=args.seed)
    train_dataset = [data.to(device) for data in train_dataset]
    clean_test_dataset = [data.to(device) for data in clean_test_dataset]
    atk_test_dataset = [data.to(device) for data in atk_test_dataset]

    # Obtain Attach Idx
    clean_label = True if args.attack_method in ('GCBA',) else False
    if args.attack_method == 'SBAG':
        attach_idx, trigger_node, avg_cnt = get_attach_idx(train_dataset, vs_ratio=args.vs_ratio,
                                                           target_label=args.target_class, clean_label=clean_label,
                                                           chosen_method=args.attack_method, num_node_attributes=num_node_attributes,
                                                           num_node_labels=num_node_labels, num_classes=num_classes)
    else:
        attach_idx = get_attach_idx(train_dataset, vs_ratio=args.vs_ratio, target_label=args.target_class, clean_label=clean_label)

    attacker = None
    if args.attack_method == 'none':
        pass
    elif args.attack_method == 'SBA':
        attacker = SBA(seed=args.seed, attack_method=args.sba_attack_method,
                       trigger_prob=args.sba_trigger_prob, trigger_size=args.trigger_size,
                       target_class=args.target_class, dataset=args.dataset, device=device)
        train_dataset = attacker.get_poisoned_rand(train_dataset, attach_idx)
    elif args.attack_method == 'ExplainBackdoor':
        attacker = ExplainBackdoor(batch_size=args.batch_size, trigger_size=args.trigger_size, trig_feat_val=args.eb_trig_feat_val,
                                   hidden=args.benign_hidden, epochs=args.benign_epochs, target_class=args.target_class, device=device)
        attacker.fit(train_dataset, lr=args.benign_lr, weight_decay=args.benign_weight_decay,
                     train_iters=args.benign_epochs, num_features=num_features, num_classes=num_classes)
        train_dataset = attacker.get_poisoned(train_dataset, attach_idx)
    elif args.attack_method == 'GCBA':
        attacker = GCBA(batch_size=args.batch_size, trigger_size=args.trigger_size, trig_feat_val=args.eb_trig_feat_val,
                        hidden=args.benign_hidden, epochs=args.benign_epochs, target_class=args.target_class, device=device)
        attacker.fit(train_dataset, lr=args.benign_lr, weight_decay=args.benign_weight_decay,
                     train_iters=args.benign_epochs, num_features=num_features, num_classes=num_classes)
        train_dataset = attacker.get_poisoned(train_dataset, attach_idx)
    elif args.attack_method == 'SBAG':
        attacker = SBAG(batch_size=args.batch_size, trigger_size=args.trigger_size,
                        hidden=args.benign_hidden, epochs=args.benign_epochs, target_class=args.target_class,
                        trigger_node=trigger_node, num_node_attributes=num_node_attributes, poisoning_num=len(attach_idx), t=avg_cnt, device=device)
        attacker.fit(train_dataset, lr=args.benign_lr, weight_decay=args.benign_weight_decay,
                     train_iters=args.benign_epochs, num_features=num_features, num_classes=num_classes)
        train_dataset = attacker.get_poisoned(train_dataset, attach_idx)

    # Model Initialize
    model = model_construct(args.model, num_features, num_classes, args.benign_hidden, args.benign_dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.benign_lr, weight_decay=args.benign_weight_decay)
    weights = torch.ones(num_classes, dtype=torch.float32).to(device)
    weights[args.target_class] *= 1.5
    criterion = nn.CrossEntropyLoss(weights)

    # Dataloader Settings
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    clean_test_loader = DataLoader(clean_test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # Train Model
    model.train()
    best_state_dict, best_acc = None, 0
    for epoch in range(args.benign_epochs):
        for idx, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data.x, data.edge_index, data.edge_weight, data.batch)
            loss = criterion(output, data.y)
            loss.backward()
            optimizer.step()

        # Test the model
        if epoch % 50 == 0:
            model.eval()
            test_loss, test_accuracy = model_test(model, clean_test_loader, criterion, device)
            logger.info('Epoch: {:03d}, Loss: {:.5f}, Accuracy: {:.2f}'.format(epoch, test_loss, test_accuracy))

            if test_accuracy > best_acc:
                best_acc = test_accuracy
                best_state_dict = model.state_dict()

    # Load Model
    model.load_state_dict(best_state_dict)
    logger.info('Best Accuracy: {:.2f}'.format(best_acc))

    # Defense
    if args.defense_method in ['DShield']:
        model = dshield(
            model, train_dataset, attach_idx, num_features, args.benign_hidden, num_classes, args.target_class,
            clean_test_loader, args.batch_size, lr=args.benign_lr, weight_decay=args.benign_weight_decay, kappa1=args.dshield_kappa1,
            thresh=args.dshield_thresh, edge_drop_ratio=args.dshield_edge_drop_ratio, feature_drop_ratio=args.dshield_feature_drop_ratio,
            tau=args.dshield_tau, pretrain_epochs=args.dshield_pretrain_epochs, classify_epochs=args.dshield_classify_epochs, finetune_epochs=args.dshield_neg_epochs,
            device=device, balance_factor=args.dshield_balance_factor, classify_rounds=args.dshield_classify_rounds
        )

    # Backdoored test model
    if args.attack_method == 'none':
        pass
    elif args.attack_method == 'SBA':
        atk_test_dataset = attacker.get_poisoned_rand(atk_test_dataset, np.arange(len(atk_test_dataset)).tolist())
        atk_test_loader = DataLoader(atk_test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
        atk_test_loss, atk_test_accuracy = model_test(model, atk_test_loader, criterion, device)
        logger.info('ASR: {:.2f}'.format(atk_test_accuracy))
    elif args.attack_method == 'ExplainBackdoor':
        atk_test_dataset = attacker.get_poisoned(atk_test_dataset, np.arange(len(atk_test_dataset)).tolist())
        atk_test_loader = DataLoader(atk_test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
        atk_test_loss, atk_test_accuracy = model_test(model, atk_test_loader, criterion, device)
        logger.info('ASR: {:.2f}'.format(atk_test_accuracy))
    elif args.attack_method == 'GCBA':
        atk_test_dataset = attacker.get_poisoned(atk_test_dataset, np.arange(len(atk_test_dataset)).tolist())
        atk_test_loader = DataLoader(atk_test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
        atk_test_loss, atk_test_accuracy = model_test(model, atk_test_loader, criterion, device)
        logger.info('ASR: {:.2f}'.format(atk_test_accuracy))
    elif args.attack_method == 'SBAG':
        atk_test_dataset = attacker.inject_trigger(atk_test_dataset, np.arange(len(atk_test_dataset)).tolist())
        atk_test_loader = DataLoader(atk_test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
        atk_test_loss, atk_test_accuracy = model_test(model, atk_test_loader, criterion, device)
        logger.info('ASR: {:.2f}'.format(atk_test_accuracy))

    # Normal test model
    clean_test_loader = DataLoader(clean_test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loss, test_accuracy = model_test(model, clean_test_loader, criterion, device)
    logger.info('Accuracy: {:.2f}'.format(test_accuracy))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1027, help='Random seed.')

    # GPU setting
    parser.add_argument('--device_id', type=int, default=0, help="Threshold of pruning edges")
    parser.add_argument('--instance', type=str, default='Attack', help='the instance name of wandb')
    parser.add_argument('--wandb_group', type=str, default='GraphBackdoor', help='the group name of wandb')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training.')

    # benign settings
    parser.add_argument('--model', type=str, default='GCN', help='model', choices=['GCN', 'GAT', 'GraphSage', 'GIN'])
    parser.add_argument('--dataset', type=str, default='MUTAG', help='Dataset', choices=['ENZYMES', 'PROTEINS', 'MNIST'])
    parser.add_argument('--benign_lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--benign_weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--benign_hidden', type=int, default=32, help='Number of hidden units.')
    parser.add_argument('--benign_dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch Size')
    parser.add_argument('--benign_epochs', type=int, default=200, help='Number of epochs to train benign and backdoor model.')

    # backdoor settings
    parser.add_argument('--target_class', type=int, default=0)
    parser.add_argument('--trigger_size', type=int, default=3, help='trigger_size')
    parser.add_argument('--vs_ratio', type=float, default=0.1, help="number of poisoning nodes relative to the full graph")
    parser.add_argument('--attack_method', type=str, default="none",
                        choices=['SBA', 'ExplainBackdoor', 'GCBA', 'SBAG'], help="defense method")

    # SBA attack
    parser.add_argument('--sba_attack_method', type=str, default='Rand_Gene', choices=['Rand_Gene', 'Rand_Samp', 'Basic', 'None'],
                        help='Method to select idx_attach for training trojan model (none means randomly select)')
    parser.add_argument('--sba_trigger_prob', type=float, default=0.5,
                        help="The probability to generate the trigger's edges in random method")

    # ExplainBackdoor
    parser.add_argument('--eb_trig_feat_val', type=float, default=1.0)

    # defense setting
    parser.add_argument('--defense_method', type=str, default="DShield",
                        choices=['DShield'],
                        help="defense method")

    # DShield
    parser.add_argument('--dshield_pretrain_epochs', type=int, default=500, help='SSL pretrain epochs')
    parser.add_argument('--dshield_finetune_epochs', type=int, default=500, help='SSL finetune epochs')
    parser.add_argument('--dshield_classify_epochs', type=int, default=400, help='Classify epochs')
    parser.add_argument('--dshield_neg_epochs', type=int, default=500, help='Negative nodes fine-tuning')
    parser.add_argument('--dshield_kappa1', type=float, default=5, help='Loss balance factor')
    parser.add_argument('--dshield_edge_drop_ratio', type=float, default=0.30, help='probability to drop edges')
    parser.add_argument('--dshield_feature_drop_ratio', type=float, default=0.30, help='probability to drop attributes')
    parser.add_argument('--dshield_tau', type=float, default=0.90, help='Temperature factor')
    parser.add_argument('--dshield_balance_factor', type=float, default=0.50, help='Balance factor')
    parser.add_argument('--dshield_classify_rounds', type=int, default=1, help='Number of rounds')
    parser.add_argument('--dshield_thresh', type=float, default=3.5, help='MAD threshold')

    args = parser.parse_known_args()[0]
    main()
