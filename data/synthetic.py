import torch


def generate_block_state_sequence(block_lengths, block_states=None):
    """
    Example:
        block_lengths = [40, 40, 40, 40]
        block_states  = [0, 1, 0, 1]

    Returns:
        z: [seq_len]
    """
    if block_states is None:
        block_states = [i % 2 for i in range(len(block_lengths))]

    if len(block_lengths) != len(block_states):
        raise ValueError("block_lengths and block_states must have the same length.")

    chunks = []
    for length, state in zip(block_lengths, block_states):
        chunks.append(torch.full((length,), state, dtype=torch.long))

    z = torch.cat(chunks, dim=0)
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


def generate_debug_sequence(
    input_dim: int,
    block_lengths=(40, 40, 40, 40),
    block_states=(0, 1, 0, 1),
):
    """
    Returns one controlled sequence for visualization.

    Outputs:
        x:       [1, seq_len, input_dim]
        z:       [1, seq_len]
        actions: [1, seq_len]
        gamma:   [1, seq_len, 2]
    """
    z = generate_block_state_sequence(block_lengths, block_states)
    seq_len = z.shape[0]

    x = torch.randn(seq_len, input_dim)
    actions = generate_actions_from_states(x, z)

    gamma = torch.zeros(seq_len, 2)
    gamma[torch.arange(seq_len), z] = 1.0

def generate_synthetic_batch(
    batch_size: int,
    seq_len: int,
    input_dim: int,
    num_states: int,
    stay_prob: float = 0.9
):
    """
    Generates a batch of sequences using simple Markov switching.

    Returns:
        x:       [B, T, input_dim]
        z:       [B, T]
        actions: [B, T]
        gamma:   [B, T, K]
    """
    x_batch = []
    z_batch = []
    actions_batch = []
    gamma_batch = []

    for _ in range(batch_size):
        z = torch.zeros(seq_len, dtype=torch.long)

        for t in range(1, seq_len):
            if torch.rand(1).item() < stay_prob:
                z[t] = z[t - 1]
            else:
                z[t] = 1 - z[t - 1]

        x = torch.randn(seq_len, input_dim)
        actions = generate_actions_from_states(x, z)

        gamma = torch.zeros(seq_len, num_states)
        gamma[torch.arange(seq_len), z] = 1.0

        x_batch.append(x)
        z_batch.append(z)
        actions_batch.append(actions)
        gamma_batch.append(gamma)

    return (
        torch.stack(x_batch),
        torch.stack(z_batch),
        torch.stack(actions_batch),
        torch.stack(gamma_batch),
    )
    
    return (
        x.unsqueeze(0),
        z.unsqueeze(0),
        actions.unsqueeze(0),
        gamma.unsqueeze(0),
    )