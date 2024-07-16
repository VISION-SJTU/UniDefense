import torch.nn as nn
from .calib_loss import FactorizationLoss
from .triplet_loss import AsymmetricalWeightedTripletLoss


def get_loss(name="cross_entropy", device="cuda:0"):
    print(f"Using loss: '{LOSSES[name]}'")
    return LOSSES[name].to(device)


LOSSES = {
    "mse": nn.MSELoss(),
    "bce": nn.BCEWithLogitsLoss(),
    "factorization": FactorizationLoss(),
    "cross_entropy": nn.CrossEntropyLoss(),
    "aw_triplet": AsymmetricalWeightedTripletLoss(),
    "kl_div": nn.KLDivLoss(reduction="batchmean", log_target=True),
}
