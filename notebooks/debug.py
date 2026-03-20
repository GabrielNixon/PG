import os
import time
import torch
import matplotlib.pyplot as plt

from data.synthetic import generate_debug_sequence
from models.mixture_model import HMMRNNMixture
from models.posterior import compute_posteriors


def normalize_series(x):
    x = x.copy()
    mean = x.mean()
    std = x.std()
    if std < 1e-8:
        return x * 0.0
    return (x - mean) / std


def debug_single_sequence(save_dir="outputs/debug_plots"):
    os.makedirs(save_dir, exist_ok=True)

    input_dim = 4
    hidden_dim = 16
    action_dim = 2
    num_states = 2

    x, z, actions, gamma_true = generate_debug_sequence(
        input_dim=input_dim,
        block_lengths=(40, 40, 40, 40),
        block_states=(0, 1, 0, 1),
    )

    model = HMMRNNMixture(
        num_states=num_states,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        action_dim=action_dim
    )

    logits_list, probs_list = model.expert_predictions(x)
    emissions = model.get_emissions_for_actions(probs_list, actions)

    pi, A = model.hmm()
    gamma_pred, _, _ = compute_posteriors(pi, A, emissions)

    z = z[0].numpy()
    actions = actions[0].numpy()
    signal = x[0, :, 0].numpy()
    signal_norm = normalize_series(signal)

    gamma_true = gamma_true[0].numpy()
    gamma_pred = gamma_pred[0].detach().numpy()

    t = list(range(len(z)))

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(t, z, linewidth=2, label="true state")
    axes[0].set_title("True latent state")
    axes[0].set_ylim(-0.2, 1.2)
    axes[0].legend()

    axes[1].plot(t, gamma_true[:, 0], label="true gamma state 0")
    axes[1].plot(t, gamma_true[:, 1], label="true gamma state 1")
    axes[1].set_title("Oracle state assignment")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend()

    axes[2].plot(t, gamma_pred[:, 0], label="pred gamma state 0")
    axes[2].plot(t, gamma_pred[:, 1], label="pred gamma state 1")
    axes[2].set_title("Predicted posterior")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].legend()

    axes[3].plot(t, signal_norm, label="normalized input signal")
    axes[3].plot(t, actions, label="action", alpha=0.8)
    axes[3].set_title("Normalized signal and actions")
    axes[3].legend()

    plt.xlabel("time")
    plt.tight_layout()

    ts = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(save_dir, f"debug_blocks_{ts}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to: {path}")

    plt.show()


if __name__ == "__main__":
    debug_single_sequence()