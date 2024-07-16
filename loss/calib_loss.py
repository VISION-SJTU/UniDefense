import torch
import torch.nn as nn


class FactorizationLoss(nn.Module):

    def __init__(self, off_diag_weight=0.005):
        super(FactorizationLoss, self).__init__()
        self.off_diag_weight = off_diag_weight

    @staticmethod
    def off_diagonal(x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, emb_a, emb_b, eps=1e-6):
        # emb: (N, c)
        # empirical cross-correlation matrix
        emb_a_norm = (emb_a - emb_a.mean(0)) / (emb_a.std(0) + eps)
        emb_b_norm = (emb_b - emb_b.mean(0)) / (emb_b.std(0) + eps)
        c = torch.matmul(emb_a_norm.T, emb_b_norm) / emb_a_norm.size(0)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).mean()
        off_diag = self.off_diagonal(c).pow_(2).mean()
        loss = on_diag + self.off_diag_weight * off_diag

        return loss
