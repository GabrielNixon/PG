import torch


def negative_log_likelihood(
    mixed_probs: torch.Tensor,
    actions: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    chosen_action_probs = mixed_probs.gather(
        dim=-1,
        index=actions.unsqueeze(-1)
    ).squeeze(-1)

    log_probs = torch.log(chosen_action_probs + eps)
    loss = -log_probs.mean()
    return loss


def expert_responsibility_loss(
    probs_list: list[torch.Tensor],
    gamma_prior: torch.Tensor,
    actions: torch.Tensor,
    eps: float = 1e-8
):
    """
    Args:
        probs_list:   list of length K, each [B, T, A]
        gamma_prior:  [B, T, K]
        actions:      [B, T]

    Returns:
        loss:         scalar
        posterior:    [B, T, K]
        chosen_probs: [B, T, K]
    """
    chosen_probs_per_expert = []

    for probs in probs_list:
        chosen = probs.gather(
            dim=-1,
            index=actions.unsqueeze(-1)
        ).squeeze(-1)  # [B, T]
        chosen_probs_per_expert.append(chosen)

    chosen_probs = torch.stack(chosen_probs_per_expert, dim=-1)  # [B, T, K]
    log_probs = torch.log(chosen_probs + eps)

    posterior = gamma_prior * chosen_probs
    posterior = posterior / (posterior.sum(dim=-1, keepdim=True) + eps)

    loss = -(posterior.detach() * log_probs).sum(dim=-1).mean()

    return loss, posterior, chosen_probs