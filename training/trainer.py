import torch
import torch.optim as optim

from data.synthetic import generate_synthetic_batch
from models.mixture_model import HMMRNNMixture
from models.posterior import compute_posteriors
from training.losses import negative_log_likelihood


def build_model(
    num_states: int,
    input_dim: int,
    hidden_dim: int,
    action_dim: int
):
    model = HMMRNNMixture(
        num_states=num_states,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        action_dim=action_dim
    )
    return model

def state_accuracy_with_label_flip(hard_pred: torch.Tensor, z_true: torch.Tensor):
    acc_direct = (hard_pred == z_true).float().mean().item()
    acc_flipped = ((1 - hard_pred) == z_true).float().mean().item()
    return max(acc_direct, acc_flipped), acc_direct, acc_flipped


def run_forward_pass_with_hmm(model, x, actions):
    logits_list, probs_list = model.expert_predictions(x)

    emissions = model.get_emissions_for_actions(probs_list, actions)

    pi, A = model.hmm()
    gamma, log_alpha, log_beta = compute_posteriors(pi, A, emissions)

    mixed_probs = model.combine_with_posteriors(probs_list, gamma)
    loss = negative_log_likelihood(mixed_probs, actions)

    return {
        "loss": loss,
        "mixed_probs": mixed_probs,
        "gamma": gamma,
        "emissions": emissions,
        "logits_list": logits_list,
        "probs_list": probs_list,
        "pi": pi,
        "A": A,
        "log_alpha": log_alpha,
        "log_beta": log_beta,
    }


def run_multi_step_training_with_hmm():
    batch = 32
    seq_len = 30
    input_dim = 4
    hidden_dim = 16
    action_dim = 2
    num_states = 2
    lr = 1e-3
    steps = 200

    model = build_model(
        num_states=num_states,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        action_dim=action_dim
    )

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for step in range(steps):
        x, z, actions, gamma_true = generate_synthetic_batch(
            batch_size=batch,
            seq_len=seq_len,
            input_dim=input_dim,
            num_states=num_states,
            stay_prob=0.9
        )

        outputs = run_forward_pass_with_hmm(model, x, actions)
        loss = outputs["loss"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 20 == 0:
            gamma_pred = outputs["gamma"]
            hard_pred = gamma_pred.argmax(dim=-1)

            best_acc, acc_direct, acc_flipped = state_accuracy_with_label_flip(
                hard_pred, z
            )

            avg_state_mass = gamma_pred.mean(dim=(0, 1))
            A = outputs["A"].detach()

            print(
                f"step {step:03d} | "
                f"loss = {loss.item():.4f} | "
                f"acc_direct = {acc_direct:.4f} | "
                f"acc_flipped = {acc_flipped:.4f} | "
                f"best_acc = {best_acc:.4f}"
            )
            print("avg_state_mass:", avg_state_mass.tolist())
            print("A:")
            print(A)
            print()