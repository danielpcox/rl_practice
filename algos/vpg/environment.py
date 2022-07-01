import torch
import gym
from setup import device
import numpy as np

class TensorPong():
    def __init__(self, name='ALE/Pong-v5', render_mode='rgb_array'):
        self.env = gym.make(name, render_mode=render_mode)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.last_obs = np.zeros((80, 80))

        self.device = device

    def reset(self):
        obs = self.env.reset()
        return self.preprocess(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action.item())
        return self.preprocess(obs), torch.tensor([reward]), done, info

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
