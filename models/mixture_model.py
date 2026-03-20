import torch
import torch.nn as nn

from models.hmm import HMM
from models.rnn_expert import RNNExpert


class HMMRNNMixture(nn.Module):
    def __init__(
        self,
        num_states: int,
        input_dim: int,
        hidden_dim: int,
        action_dim: int
    ):
        super().__init__()
        self.num_states = num_states
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.hmm = HMM(num_states=num_states)

        self.experts = nn.ModuleList([
            RNNExpert(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                action_dim=action_dim
            )
            for _ in range(num_states)
        ])

    def expert_predictions(self, x: torch.Tensor):
        """
        Args:
            x: [batch, seq_len, input_dim]

        Returns:
            logits_list: list of length K, each [batch, seq_len, action_dim]
            probs_list:  list of length K, each [batch, seq_len, action_dim]
        """
        logits_list = []
        probs_list = []

        for expert in self.experts:
            logits, probs, _ = expert(x)
            logits_list.append(logits)
            probs_list.append(probs)

        return logits_list, probs_list

    def combine_with_posteriors(
        self,
        probs_list: list,
        gamma: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            probs_list: list of length K, each [batch, seq_len, action_dim]
            gamma:      [batch, seq_len, K]

        Returns:
            mixed_probs: [batch, seq_len, action_dim]
        """
        stacked_probs = torch.stack(probs_list, dim=2)
        gamma = gamma.unsqueeze(-1)
        mixed_probs = (stacked_probs * gamma).sum(dim=2)
        return mixed_probs

    def get_emissions_for_actions(
        self,
        probs_list: list,
        actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            probs_list: list of length K, each [batch, seq_len, action_dim]
            actions:    [batch, seq_len]

        Returns:
            emissions:  [batch, seq_len, K]
                        emissions[b, t, z] = probability assigned by expert z
                        to the observed action at (b, t)
        """
        emission_list = []

        for probs in probs_list:
            chosen_action_probs = probs.gather(
                dim=-1,
                index=actions.unsqueeze(-1)
            ).squeeze(-1)
            emission_list.append(chosen_action_probs)

        emissions = torch.stack(emission_list, dim=-1)
        return emissions

    def forward(
        self,
        x: torch.Tensor,
        gamma: torch.Tensor,
        actions: torch.Tensor = None
    ):
        """
        Args:
            x:       [batch, seq_len, input_dim]
            gamma:   [batch, seq_len, K]
            actions: [batch, seq_len] or None

        Returns:
            logits_list: list of length K
            probs_list:  list of length K
            mixed_probs: [batch, seq_len, action_dim]
            emissions:   [batch, seq_len, K] or None
        """
        logits_list, probs_list = self.expert_predictions(x)
        mixed_probs = self.combine_with_posteriors(probs_list, gamma)

        emissions = None
        if actions is not None:
            emissions = self.get_emissions_for_actions(probs_list, actions)

        return logits_list, probs_list, mixed_probs, emissions