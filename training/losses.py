import torch


def negative_log_likelihood(
    mixed_probs: torch.Tensor,
    actions: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Args:
        mixed_probs: [batch, seq_len, action_dim]
        actions:     [batch, seq_len]
        eps:         small constant for numerical stability

    Returns:
        loss: scalar tensor
    """
    chosen_action_probs = mixed_probs.gather(
        dim=-1,
        index=actions.unsqueeze(-1)
    ).squeeze(-1)

    log_probs = torch.log(chosen_action_probs + eps)
    loss = -log_probs.mean()
    return loss