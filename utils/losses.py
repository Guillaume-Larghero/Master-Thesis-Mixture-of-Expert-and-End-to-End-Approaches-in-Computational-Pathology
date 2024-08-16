import torch
import torch.nn as nn


class ZLoss(nn.Module):
    """
    See [1] B. Zoph et al., "ST-MoE: Designing Stable and Transferable Sparse Expert Models." arXiv, Feb. 17, 2022. doi: 10.48550/arXiv.2202.08906.
    """

    def __init__(self, scale=0.1):
        super(ZLoss, self).__init__()
        self.scale = scale

    def forward(self, logits):
        """
        Implements the z-loss, which penalizes large logits to stabilize training.
        """
        z_loss = torch.logsumexp(logits, dim=1) ** 2
        return z_loss.mean() * self.scale


def entropy_penalty(weights, epsilon=1e-5):
    entropy = -torch.sum(weights * torch.log(weights + epsilon), dim=-1)
    entropy_penalty = torch.mean(entropy)
    return entropy_penalty


def load_balancing_loss(gating_weights):
    expert_loads = gating_weights.sum(dim=0)
    total_load = expert_loads.sum()
    average_load_per_expert = expert_loads / total_load
    num_experts = gating_weights.size(1)
    ideal_load = 1.0 / num_experts
    load_balancing_loss = ((average_load_per_expert - ideal_load) ** 2).sum()
    return load_balancing_loss
