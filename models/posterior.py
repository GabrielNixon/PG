import torch


def logsumexp(x: torch.Tensor, dim: int):
    return torch.logsumexp(x, dim=dim)


def forward_pass_log(
    log_pi: torch.Tensor,
    log_A: torch.Tensor,
    log_emissions: torch.Tensor
):
    """
    Args:
        log_pi:        [K]
        log_A:         [K, K]
        log_emissions: [B, T, K]

    Returns:
        log_alpha: [B, T, K]
    """
    B, T, K = log_emissions.shape
    log_alpha = torch.zeros(B, T, K, device=log_emissions.device)

    log_alpha[:, 0, :] = log_pi.unsqueeze(0) + log_emissions[:, 0, :]

    for t in range(1, T):
        prev = log_alpha[:, t - 1, :].unsqueeze(-1)
        trans = log_A.unsqueeze(0)
        scores = prev + trans
        log_alpha[:, t, :] = log_emissions[:, t, :] + logsumexp(scores, dim=1)

    return log_alpha


def backward_pass_log(
    log_A: torch.Tensor,
    log_emissions: torch.Tensor
):
    """
    Args:
        log_A:         [K, K]
        log_emissions: [B, T, K]

    Returns:
        log_beta: [B, T, K]
    """
    B, T, K = log_emissions.shape
    log_beta = torch.zeros(B, T, K, device=log_emissions.device)

    for t in range(T - 2, -1, -1):
        trans = log_A.unsqueeze(0)
        next_beta = log_beta[:, t + 1, :].unsqueeze(1)
        next_emission = log_emissions[:, t + 1, :].unsqueeze(1)
        scores = trans + next_emission + next_beta
        log_beta[:, t, :] = logsumexp(scores, dim=2)

    return log_beta


def compute_gamma_log(
    log_alpha: torch.Tensor,
    log_beta: torch.Tensor
):
    """
    Args:
        log_alpha: [B, T, K]
        log_beta:  [B, T, K]

    Returns:
        gamma: [B, T, K]
    """
    log_gamma_unnorm = log_alpha + log_beta
    log_norm = torch.logsumexp(log_gamma_unnorm, dim=-1, keepdim=True)
    log_gamma = log_gamma_unnorm - log_norm
    gamma = torch.exp(log_gamma)
    return gamma


def compute_posteriors(
    pi: torch.Tensor,
    A: torch.Tensor,
    emissions: torch.Tensor,
    eps: float = 1e-8
):
    """
    Args:
        pi:        [K]
        A:         [K, K]
        emissions: [B, T, K]

    Returns:
        gamma:     [B, T, K]
        log_alpha: [B, T, K]
        log_beta:  [B, T, K]
    """
    log_pi = torch.log(pi + eps)
    log_A = torch.log(A + eps)
    log_emissions = torch.log(emissions + eps)

    log_alpha = forward_pass_log(log_pi, log_A, log_emissions)
    log_beta = backward_pass_log(log_A, log_emissions)
    gamma = compute_gamma_log(log_alpha, log_beta)

    return gamma, log_alpha, log_beta