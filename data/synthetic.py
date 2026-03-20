import torch


def generate_state_sequence(seq_len: int, stay_prob: float = 0.9):
    z = torch.zeros(seq_len, dtype=torch.long)
    z[0] = torch.randint(0, 2, (1,))

    for t in range(1, seq_len):
        if torch.rand(1).item() < stay_prob:
            z[t] = z[t - 1]
        else:
            z[t] = 1 - z[t - 1]

    return z


def generate_actions_from_states(x: torch.Tensor, z: torch.Tensor):
    """
    x: [seq_len, input_dim]
    z: [seq_len]

    Returns:
        actions: [seq_len]
    """
    seq_len = x.shape[0]
    actions = torch.zeros(seq_len, dtype=torch.long)

    for t in range(seq_len):
        signal = x[t, 0].item()

        if z[t].item() == 0:
            if signal > 0:
                probs = torch.tensor([0.85, 0.15])
            else:
                probs = torch.tensor([0.15, 0.85])
        else:
            if signal > 0:
                probs = torch.tensor([0.15, 0.85])
            else:
                probs = torch.tensor([0.85, 0.15])

        actions[t] = torch.multinomial(probs, num_samples=1)

    return actions


def generate_synthetic_batch(
    batch_size: int,
    seq_len: int,
    input_dim: int,
    num_states: int = 2,
    stay_prob: float = 0.9
):
    if num_states != 2:
        raise ValueError("This first synthetic generator currently supports num_states=2 only.")

    x_batch = []
    z_batch = []
    actions_batch = []
    gamma_batch = []

    for _ in range(batch_size):
        x = torch.randn(seq_len, input_dim)
        z = generate_state_sequence(seq_len=seq_len, stay_prob=stay_prob)
        actions = generate_actions_from_states(x, z)

        gamma = torch.zeros(seq_len, num_states)
        gamma[torch.arange(seq_len), z] = 1.0

        x_batch.append(x)
        z_batch.append(z)
        actions_batch.append(actions)
        gamma_batch.append(gamma)

    x_batch = torch.stack(x_batch, dim=0)
    z_batch = torch.stack(z_batch, dim=0)
    actions_batch = torch.stack(actions_batch, dim=0)
    gamma_batch = torch.stack(gamma_batch, dim=0)

    return x_batch, z_batch, actions_batch, gamma_batch