import torch

from models.rnn_expert import RNNExpert
from models.hmm import HMM
from models.mixture_model import HMMRNNMixture
from training.losses import negative_log_likelihood
from models.posterior import compute_posteriors

def test_rnn_expert():
    batch = 4
    seq_len = 10
    input_dim = 6
    hidden_dim = 12
    action_dim = 3

    x = torch.randn(batch, seq_len, input_dim)

    model = RNNExpert(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        action_dim=action_dim
    )

    logits, probs, h_n = model(x)

    print("RNN expert")
    print("logits shape:", logits.shape)
    print("probs shape:", probs.shape)
    print("h_n shape:", h_n.shape)
    print()


def test_hmm():
    num_states = 2
    hmm = HMM(num_states=num_states)

    pi, A = hmm()

    print("HMM")
    print("pi shape:", pi.shape)
    print("A shape:", A.shape)
    print("pi:", pi)
    print("A:", A)
    print("row sums of A:", A.sum(dim=1))
    print()

def test_mixture_model():
    batch = 4
    seq_len = 10
    input_dim = 6
    hidden_dim = 12
    action_dim = 3
    num_states = 2

    x = torch.randn(batch, seq_len, input_dim)
    gamma = torch.softmax(torch.randn(batch, seq_len, num_states), dim=-1)
    actions = torch.randint(0, action_dim, (batch, seq_len))

    model = HMMRNNMixture(
        num_states=num_states,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        action_dim=action_dim
    )

    logits_list, probs_list, mixed_probs, emissions = model(x, gamma, actions)

    print("Mixture model")
    print("number of experts:", len(logits_list))
    print("expert 0 probs shape:", probs_list[0].shape)
    print("gamma shape:", gamma.shape)
    print("mixed_probs shape:", mixed_probs.shape)
    print("mixed_probs row sums:", mixed_probs.sum(dim=-1)[0, :5])
    print("emissions shape:", emissions.shape)
    print("emissions sample:", emissions[0, :5])
    print()

def test_loss():
    batch = 4
    seq_len = 10
    input_dim = 6
    hidden_dim = 12
    action_dim = 3
    num_states = 2

    x = torch.randn(batch, seq_len, input_dim)
    gamma = torch.softmax(torch.randn(batch, seq_len, num_states), dim=-1)
    actions = torch.randint(0, action_dim, (batch, seq_len))

    model = HMMRNNMixture(
        num_states=num_states,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        action_dim=action_dim
    )

    _, _, mixed_probs, _ = model(x, gamma)
    loss = negative_log_likelihood(mixed_probs, actions)

    print("Loss")
    print("actions shape:", actions.shape)
    print("loss:", loss.item())
    print()

def test_posteriors():
    batch = 4
    seq_len = 10
    num_states = 2

    pi = torch.tensor([0.5, 0.5])
    A = torch.tensor([
        [0.9, 0.1],
        [0.2, 0.8]
    ])

    emissions = torch.softmax(
        torch.randn(batch, seq_len, num_states),
        dim=-1
    )

    gamma, log_alpha, log_beta = compute_posteriors(pi, A, emissions)

    print("Posterior")
    print("emissions shape:", emissions.shape)
    print("gamma shape:", gamma.shape)
    print("log_alpha shape:", log_alpha.shape)
    print("log_beta shape:", log_beta.shape)
    print("gamma row sums:", gamma.sum(dim=-1)[0, :5])
    print()


if __name__ == "__main__":
    test_rnn_expert()
    test_hmm()
    test_mixture_model()
    test_loss()
    test_posteriors()