import os
import time
import torch
import matplotlib.pyplot as plt

from data.synthetic import generate_synthetic_batch
from models.mixture_model import HMMRNNMixture
from models.posterior import compute_posteriors


def debug_single_sequence(save_dir="outputs/debug_plots"):
    os.makedirs(save_dir, exist_ok=True)

    batch = 1
    seq_len = 50
    input_dim = 4
    hidden_dim = 16
    action_dim = 2
    num_states = 2

    x, z, actions, _ = generate_synthetic_batch(
        batch_size=batch,
        seq_len=seq_len,
        input_dim=input_dim,
        num_states=num_states,
        stay_prob=0.9
    )

    model = HMMRNNMixture(
        num_states=num_states,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        action_dim=action_dim
    )

    # forward pass
    logits_list, probs_list = model.expert_predictions(x)
    emissions = model.get_emissions_for_actions(probs_list, actions)

    pi, A = model.hmm()
    gamma, _, _ = compute_posteriors(pi, A, emissions)

    z = z[0].numpy()
    gamma = gamma[0].detach().numpy()
    actions = actions[0].numpy()
    signal = x[0, :, 0].numpy()

    plt.figure(figsize=(12, 6))

    plt.plot(z, label="true state", linewidth=2)
    plt.plot(gamma[:, 0], label="gamma state 0")
    plt.plot(gamma[:, 1], label="gamma state 1")
    plt.plot(signal, label="input signal", alpha=0.5)

    plt.legend()
    plt.title("State vs Posterior vs Signal")

    # Save with timestamp
    ts = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(save_dir, f"debug_{ts}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to: {path}")

    plt.show()


if __name__ == "__main__":
    debug_single_sequence()