import logging

from models.GCN import GCN
from models.GAT import GAT
from models.SAGE import GraphSage
from models.GCN_Encoder import GCN_Encoder
from models.GNNGuard import GNNGuard
from models.RobustGCN import RobustGCN

try:
    if 'logger' not in globals():
        logging.basicConfig()
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
except NameError:
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)


def model_construct(dataset, model_name, feat_dim, num_class, hidden, dropout, lr, weight_decay, device):

    if dataset == 'Reddit2':
        use_ln = True
        layer_norm_first = False
    else:
        use_ln = False
        layer_norm_first = False

    model = None
    if model_name == 'GCN':
        model = GCN(n_feat=feat_dim, n_hid=hidden, n_class=num_class, dropout=dropout,
                    lr=lr, weight_decay=weight_decay, device=device, use_ln=use_ln, layer_norm_first=layer_norm_first)
    elif model_name == 'GAT':
        model = GAT(n_feat=feat_dim, n_hid=hidden, n_class=num_class,
                    heads=8, dropout=dropout, lr=lr, weight_decay=weight_decay, device=device)
    elif model_name == 'GraphSage':
        model = GraphSage(n_feat=feat_dim, n_hid=hidden, n_class=num_class, dropout=dropout,
                          lr=lr, weight_decay=weight_decay, device=device)
    elif model_name == 'GCN_Encoder':
        model = GCN_Encoder(n_feat=feat_dim, n_hid=hidden, n_class=num_class,
                            dropout=dropout, lr=lr, weight_decay=weight_decay,
                            device=device, use_ln=use_ln, layer_norm_first=layer_norm_first)
    elif model_name == 'GNNGuard':
        model = GNNGuard(n_feat=feat_dim, n_hid=hidden, n_class=num_class, dropout=dropout,
                         lr=lr, weight_decay=weight_decay, use_ln=use_ln, device=device)
    elif model_name == 'GNNGuard':
        model = RobustGCN(n_feat=feat_dim, n_hid=hidden, n_class=num_class,
                          dropout=dropout, lr=lr, weight_decay=weight_decay, device=device)
    else:
        logger.info("Not implement {}".format(model_name))
    return model
