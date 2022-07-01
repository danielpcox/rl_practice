from matplotlib import pyplot as plt
import numpy as np

# HACK because multinomial not yet implemented on MPS
import torch
import torch.nn as nn

class PongActor(nn.Module):
    def __init__(self, action_dim, hid_dim, activation=nn.LeakyReLU):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(1764, hid_dim),
            activation(),
            nn.Linear(hid_dim, hid_dim),
            activation(),
            nn.Linear(hid_dim, action_dim),
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        logits = self.mlp(x)
        policy = torch.distributions.Categorical(logits=logits)
        return policy

class PongCritic(nn.Module):
    def __init__(self, hid_dim, activation=nn.LeakyReLU):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(1764, hid_dim), # TODO compute rather than magic num
            activation(),
            nn.Linear(hid_dim, hid_dim),
            activation(),
            nn.Linear(hid_dim, 1),
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        out = self.mlp(x)
        return out.squeeze(dim=1)

class ActorCritic(nn.Module):
    def __init__(self, action_dim, hid_dim, device):
        super().__init__()
        self.device = device
        self.actor = PongActor(action_dim, hid_dim)
        self.critic = PongCritic(hid_dim)

    def forward(self, x):
        policy = self.actor(x)
        value = self.critic(x)
        action = policy.sample()
        logp = policy.log_prob(action)
        return action, logp, value

