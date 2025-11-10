# models/losses.py
import torch

def chamfer_distance(p1, p2):
    """
    Chamfer Distance between two point sets.
    p1: (B, N, 3)
    p2: (B, M, 3)
    Returns: scalar (average over batch)
    Implemented using torch.cdist for speed (computes pairwise distances).
    """
    # make sure float
    p1 = p1.float()
    p2 = p2.float()

    # pairwise distance matrix: (B, N, M)
    # torch.cdist computes Euclidean distances
    dist = torch.cdist(p1, p2, p=2)  # (B,N,M)

    # for each p1 find nearest p2
    dist1, _ = torch.min(dist, dim=2)  # (B,N)
    # for each p2 find nearest p1
    dist2, _ = torch.min(dist, dim=1)  # (B,M)

    # mean over points then batch
    loss = dist1.mean(dim=1) + dist2.mean(dim=1)  # (B,)
    return loss.mean()
