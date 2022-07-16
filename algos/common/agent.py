# HACK because multinomial not yet implemented on MPS
from pdb import set_trace

import torch
import torch.nn as nn
from torch.distributions import Categorical
from torchtyping import TensorType as T
from typeguard import typechecked

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

    @typechecked
    def forward(self, x: T['B':...,'C','H','W']) -> (Categorical, T['B', 'A']):
        x = x.reshape(x.shape[0], -1)
        logits = self.mlp(x)
        policy = Categorical(logits=logits)
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

    @typechecked
    def forward(self, x: T['B':...,'C','H','W']) -> T['B']:
        x = x.reshape(x.shape[0], -1)
        out = self.mlp(x)
        return out.squeeze(dim=1)

class ActorCritic(nn.Module):
    def __init__(self, action_dim: int, hid_dim: int, device: torch.device):
        super().__init__()
        self.device = device
        self.actor = PongActor(action_dim, hid_dim)
        self.critic = PongCritic(hid_dim)
        self.to(device)

    @typechecked
    def forward(self, x: T['B':...,'C','H','W']) -> (T['B'],T['B'],T['B','A'],T['B']):
        policy = self.actor(x)
        value = self.critic(x)
        action = policy.sample()
        logp = policy.log_prob(action)
        return action, logp, policy.logits, value
