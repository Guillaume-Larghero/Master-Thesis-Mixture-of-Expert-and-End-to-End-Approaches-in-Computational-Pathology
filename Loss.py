import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

class CollapseOfExpertsLoss(nn.Module):
    def __init__(self, alpha=0.1):
        super(CollapseOfExpertsLoss, self).__init__()
        self.alpha = alpha  # Penalty hyperparameter

    def forward(self, expert_weights):
        '''
        Compute the collapse of experts loss.

        Args:
        - expert_weights: Tensor containing the softmax output of the gating network,
                          shape: (batch_size, num_experts)

        Returns:
        - loss: Scalar tensor representing the collapse of experts loss
        '''
        num_experts = expert_weights.size(1)
        mean_weight = 1.0 / num_experts  # Mean weight assuming uniform distribution

        # Compute the penalty term by measuring the KL divergence between the weight distribution and a standard distribution
        kl_divergence = -torch.mean(torch.sum(expert_weights * torch.log(expert_weights / mean_weight), dim=1))

        # Apply regularization term to discourage large differences from standard distribution
        loss = kl_divergence + self.alpha * torch.mean(torch.abs(expert_weights - mean_weight))

        return loss


class ClassificationLoss(nn.Module):
    def __init__(self):
        super(ClassificationLoss, self).__init__()

    def forward(self, expert_logits, targets):
        '''
        Compute the classification loss using cross-entropy.

        Args:
        - expert_logits: Tensor containing the logits predicted by the experts,
                          shape: (batch_size, num_experts, num_classes)
        - targets: Tensor containing the ground truth labels,
                          shape: (batch_size,)

        Returns:
        - loss: Scalar tensor representing the classification loss
        '''
        classification_loss = torch.mean(torch.sum(-F.log_softmax(expert_logits, dim=2) * F.one_hot(targets, num_classes=expert_logits.size(2)), dim=2))

        return classification_loss
    
    
class FinalLoss(nn.Module):
    def __init__(self, collapse_of_experts_weight=1.0, classification_weight=1.0):
        super(FinalLoss, self).__init__()
        self.collapse_of_experts_weight = collapse_of_experts_weight
        self.classification_weight = classification_weight
        self.collapse_of_experts_loss = CollapseOfExpertsLoss()
        self.classification_loss = ClassificationLoss()

    def forward(self, expert_weights, expert_logits, targets):
        '''
        Compute the final combined loss.

        Args:
        - expert_weights: Tensor containing the softmax output of the gating network,
                          shape: (batch_size, num_experts)
        - expert_logits: Tensor containing the logits predicted by the experts,
                          shape: (batch_size, num_experts, num_classes)
        - targets: Tensor containing the ground truth labels,
                          shape: (batch_size,)

        Returns:
        - loss: Scalar tensor representing the final combined loss
        '''
        collapse_of_experts_loss = self.collapse_of_experts_loss(expert_weights)
        classification_loss = self.classification_loss(expert_logits, targets)

        # Combine everything together:
        final_loss = self.collapse_of_experts_weight * collapse_of_experts_loss + self.classification_weight * classification_loss

        return final_loss