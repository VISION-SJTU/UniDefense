import torch
import torch.nn as nn


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist -= 2 * torch.matmul(x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def triplet_mining(dist_mat: torch.Tensor, labels: torch.Tensor):
    """
    Anchors only consider real faces
    """
    eps = 1e-12
    N = dist_mat.shape[0]
    N_real = torch.where(1 - labels)[0].shape[0]

    indices_not_equal = ~torch.eye(N, device=labels.device, dtype=torch.bool)
    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_pos = torch.logical_and(is_pos, indices_not_equal)

    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # anchors only compute for real faces 
    # real faces are always presented before fake faces in training mini-batches
    # in our implementation
    dist_ap = dist_mat[: N_real][is_pos[: N_real]].reshape(N_real, -1)
    # print(f'dist ap:\n{dist_ap}')
    dist_an = dist_mat[: N_real][is_neg[: N_real]].reshape(N_real, -1)
    # print(f'dist an:\n{dist_an}')

    exp_dist_ap = torch.exp(dist_ap)
    exp_dist_an = torch.exp(-dist_an)

    wp = exp_dist_ap / (exp_dist_ap.sum(1, keepdim=True) + eps)
    wn = exp_dist_an / (exp_dist_an.sum(1, keepdim=True) + eps)

    # shape [N_real]
    final_wp = torch.sum(wp * dist_ap, dim=1)
    final_wn = torch.sum(wn * dist_an, dim=1)

    return final_wp, final_wn


class AsymmetricalWeightedTripletLoss(nn.Module):

    def __init__(self):
        super(AsymmetricalWeightedTripletLoss, self).__init__()
        self.margin_loss = nn.SoftMarginLoss()

    def forward(self, global_feat, labels, normalize_feature=False):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)

        dist_mat = euclidean_dist(global_feat, global_feat)
        final_wp, final_wn = triplet_mining(dist_mat, labels)
        y = final_wn.new().resize_as_(final_wn).fill_(1)
        return self.margin_loss(final_wn - final_wp, y)
