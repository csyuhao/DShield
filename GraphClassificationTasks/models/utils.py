import logging
import torch

from models.GCN import GCN
from models.GAT import GAT


try:
    if 'logger' not in globals():
        logging.basicConfig()
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
except NameError:
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)


def model_construct(model_name, feat_dim, num_class, hidden, dropout):

    model = None
    if model_name == 'GCN':
        model = GCN(n_feat=feat_dim, n_hid=hidden, n_class=num_class, dropout=dropout)
    elif model_name == 'GAT':
        model = GAT(n_feat=feat_dim, n_hid=hidden, n_class=num_class, dropout=dropout)
    else:
        logger.info("Not implement {}".format(model_name))
    return model


@torch.no_grad()
def model_test(model, loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    for data in loader:
        data = data.to(device)
        output = model(data.x, data.edge_index, data.edge_weight, data.batch)
        loss = criterion(output, data.y)
        total_loss += loss.item()
        _, predicted = torch.max(output, 1)
        total += data.y.size(0)
        correct += (predicted == data.y).sum().item()
    return total_loss / len(loader), correct / total
