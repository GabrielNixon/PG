import torch
import torch.nn as nn
import torch.nn.functional as F


class HMM(nn.Module):
    def __init__(self, num_states: int, stay_logit: float = 2.0, switch_logit: float = -2.0):
        super().__init__()
        self.num_states = num_states

        self.logits_pi = nn.Parameter(torch.zeros(num_states))

        init_A = torch.full((num_states, num_states), switch_logit)
        init_A.fill_(switch_logit)
        for i in range(num_states):
            init_A[i, i] = stay_logit

        self.logits_A = nn.Parameter(init_A)

    def get_pi(self) -> torch.Tensor:
        return F.softmax(self.logits_pi, dim=0)

    def get_A(self) -> torch.Tensor:
        return F.softmax(self.logits_A, dim=1)

    def forward(self):
        pi = self.get_pi()
        A = self.get_A()
        return pi, A