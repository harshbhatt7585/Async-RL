# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 128)
        self.policy = nn.Linear(128, action_dim)
        self.value = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc(x))
        policy_logits = self.policy(x)
        value = self.value(x)
        return policy_logits, value
