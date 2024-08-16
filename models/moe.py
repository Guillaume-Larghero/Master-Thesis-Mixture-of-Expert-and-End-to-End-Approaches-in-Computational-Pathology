from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.expert import Expert, MILExpert
from models.router import MILRouter, Router


class MILMoE(nn.Module):
    """
    Implements a  simple implementation of a Mixture of Experts model for MIL.

    The model expects a list of torch.Tensors as input, where each tensor is the output of a different expert.
    The router takes in the concatenated input (across the feature dimension), and outputs a probability
    distribution over the experts. The output of each expert is then multiplied by the corresponding weight,
    and the results are summed to produce the final output.

    Example:
      Input: [B, N, D_1], [B, N, D_2], [B, N, D_3]
      Router input: [B, N, D_1 + D_2 + D_3]
      Router output: [B, n_experts]
      Expert output: [B, n_experts, n_classes]
      Final output: (class predictions and router weights)
        [B, n_classes], [B, n_experts]
    """

    def __init__(
        self,
        experts: List[MILExpert],
        router: MILRouter,
        strategy: str = "weighted_sum",
        temperature: float = 1.0,
    ):
        super(MILMoE, self).__init__()
        self.experts = nn.ModuleList(experts)
        self.router = router
        self.strategy = strategy
        self.temperature = temperature
        # top_k == weighted_sum if k == n_experts
        if "top_" in self.strategy:
            self.top_k = int(self.strategy.split("_")[1])
            assert self.top_k <= len(
                experts
            ), "top_k must not exceed the number of experts."

    def forward(
        self, x: List[torch.Tensor], masks: Optional[List[torch.Tensor]] = None
    ):
        """
        Expects input of shape:
          x: [B, N, D_1], [B, N, D_2], [B, N, D_3], ...
          masks: [B, N, 1], [B, N, 1], [B, N, 1], ...
        Where:
          B: Batch size
          N: Number of patch features
          D_X: The dimension of the X-th expert feature dimension

        Example:
        We chose Chief (D_1: 768), UNI (D_2: 1024), and resnet50 (D_3: 2048) expert models.
        The expected input would be:
          [[B, N, 768], [B, N, 1024], [B, N, 2048]]

        The input to the router will be
          [B, N, 3840]

        The output of the router (weights) will be:
          [B, 3]

        The output of each expert will be:
          [B, n_classes]

        The stacked output thus will be:
          [B, 3, n_classes]

        The final output will be:
          [B, n_classes]

        We return both the output probabilities and the weights to handle router-specific losses.
        """
        concatenated_input = torch.cat(x, dim=-1)
        probabilities, logits = self.router(concatenated_input)

        # TODO: fix the case where masks is None
        outputs = torch.stack(
            [
                expert(features, m)
                for features, m, expert in zip(x, masks, self.experts)
            ],
            dim=1,
        )

        if self.strategy == "weighted_sum":
            final_output = torch.einsum("be,bec->bc", probabilities, outputs)
        elif "top_" in self.strategy:
            _, top_k_indices = torch.topk(probabilities, self.top_k, dim=1)
            top_k_outputs = outputs[
                torch.arange(outputs.size(0))[:, None], top_k_indices
            ]
            top_k_weights = probabilities[
                torch.arange(probabilities.size(0))[:, None], top_k_indices
            ]
            final_output = torch.einsum("bk,bkc->bc", top_k_weights, top_k_outputs)
        else:
            raise ValueError(
                "Unsupported strategy. Choose 'weighted_sum' or 'select_max'."
            )

        return (
            F.softmax(final_output / self.temperature, dim=1),
            probabilities,
            logits,
        )


class MoE(nn.Module):
    """
    Implements a  simple implementation of a Mixture of Experts model for tile-level prediction.

    The model expects a list of torch.Tensors as input, where each tensor is the output of a different expert.
    The router takes in the concatenated input (across the feature dimension), and outputs a probability
    distribution over the experts. The output of each expert is then multiplied by the corresponding weight,
    and the results are summed to produce the final output.

    Example:
      Input: [B, D_1], [B, D_2], [B, D_3]
      Router input: [B, D_1 + D_2 + D_3]
      Router output: [B, n_experts]
      Expert output: [B, n_experts, n_classes]
      Final output: (class predictions and router weights)
        [B, n_classes], [B, n_experts]
    """

    def __init__(
        self,
        experts: List[Expert],
        router: Router,
        strategy: str = "weighted_sum",
        temperature: float = 1.0,
    ):
        super(MoE, self).__init__()
        self.experts = nn.ModuleList(experts)
        self.router = router
        self.strategy = strategy
        self.temperature = temperature
        # top_k == weighted_sum if k == n_experts
        if "top_" in self.strategy:
            self.top_k = int(self.strategy.split("_")[1])
            assert self.top_k <= len(
                experts
            ), "top_k must not exceed the number of experts."

    def forward(self, x: List[torch.Tensor]):
        """
        Expects input of shape:
          x: [B, D_1], [B, D_2], [B, D_3], ...
        Where:
          B: Batch size
          D_X: The dimension of the X-th expert feature dimension

        Example:
        We chose Chief (D_1: 768), UNI (D_2: 1024), and resnet50 (D_3: 2048) expert models.
        The expected input would be:
          [[B, 768], [B, 1024], [B, 2048]]

        The input to the router will be
          [B, 3840]

        The output of the router (weights) will be:
          [B, 3]

        The output of each expert will be:
          [B, n_classes]

        The stacked output thus will be:
          [B, 3, n_classes]

        The final output will be:
          [B, n_classes]

        We return both the output probabilities and the weights to handle router-specific losses.
        """
        concatenated_input = torch.cat(x, dim=-1)
        probabilities, logits = self.router(concatenated_input)
        outputs = torch.stack(
            [expert(x) for x, expert in zip(x, self.experts)],
            dim=1,
        )
        if self.strategy == "weighted_sum":
            final_output = torch.einsum("be,bec->bc", probabilities, outputs)
        elif "top_" in self.strategy:
            _, top_k_indices = torch.topk(probabilities, self.top_k, dim=1)
            top_k_outputs = outputs[
                torch.arange(outputs.size(0))[:, None], top_k_indices
            ]
            top_k_weights = probabilities[
                torch.arange(probabilities.size(0))[:, None], top_k_indices
            ]
            final_output = torch.einsum("bk,bkc->bc", top_k_weights, top_k_outputs)
        else:
            raise ValueError("Unsupported strategy. Choose 'weighted_sum' or 'top_k'.")

        return (
            F.softmax(final_output / self.temperature, dim=1),
            probabilities,
            logits,
        )