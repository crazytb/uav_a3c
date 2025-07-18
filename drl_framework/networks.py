import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .params import device

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy = nn.Linear(hidden_dim, action_dim)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.shared(x)
        policy_logits = self.policy(x)
        state_value = self.value(x)
        return policy_logits, state_value