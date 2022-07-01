from matplotlib import pyplot as plt
import numpy as np

# HACK because multinomial not yet implemented on MPS
import torch
import torch.nn as nn

class PongActor(nn.Module):
    def __init__(self, action_dim, hid_dim, activation=nn.LeakyReLU):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(6400, hid_dim),
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
            nn.Linear(6400, hid_dim),
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
        return policy, value

# First attempt, at least learns really really slowly
class PongActorCritic(nn.Module):
    def __init__(self, action_dim, hid_dim, device, activation=nn.Tanh):
        super().__init__()

        self.last_obs = np.zeros((80, 80))

        # unused, atm
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=6, stride=2),
            nn.MaxPool2d(2, stride=2),
            activation(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            activation(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            activation(),
        )

        self.mlp = nn.Sequential(
            # nn.Linear(2304, hid_dim),
            nn.Linear(6400, hid_dim),
            activation(),
            nn.Linear(hid_dim, hid_dim),
            activation(),
        )

        self.actor_head = nn.Linear(hid_dim, action_dim)

        # unused, atm
        self.critic_head = nn.Linear(hid_dim, 1)

        self.to(device)
        self.device = device

    def forward(self, x):
        body = self.preprocess(x)
        # body = self.cnn(body)
        body = body.reshape(body.shape[0], -1)
        body = self.mlp(body)

        logits = self.actor_head(body)
        pi = torch.distributions.Categorical(logits=logits)
        action = pi.sample()
        logp = pi.log_prob(action)

        # state_value = self.critic_head(body).squeeze()

        return action.item(), logp #, state_value

    def preprocess(self, obs):
        """Preprocess Pong observation

        in: numpy (210, 160, 3)
        out: torch (1, 80, 80)
        """

        # slice off top and bottom, and downsample
        obs = obs[34:194:2, ::2, 2]

        # result is diff between two frames to make easier for NN w/o memory
        result = obs - self.last_obs

        # Simplify values to {0, 1}
        result[result != 0] = 1
        result[result != 1] = 0

        self.last_obs = obs

        result = torch.as_tensor(result,
                                 dtype=torch.float32,
                                 device=self.device)
        result = result.unsqueeze(dim=0)

        return result
