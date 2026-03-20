import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNExpert(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.readout = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor, h0: torch.Tensor = None):
        """
        Args:
            x:  Tensor of shape [batch, seq_len, input_dim]
            h0: Optional initial hidden state of shape [1, batch, hidden_dim]

        Returns:
            logits: Tensor of shape [batch, seq_len, action_dim]
            probs:  Tensor of shape [batch, seq_len, action_dim]
            h_n:    Tensor of shape [1, batch, hidden_dim]
        """
        rnn_out, h_n = self.rnn(x, h0)
        logits = self.readout(rnn_out)
        probs = F.softmax(logits, dim=-1)
        return logits, probs, h_n