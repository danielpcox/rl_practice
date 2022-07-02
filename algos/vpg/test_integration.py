# hack to run from repo root
import sys

sys.path.insert(0, '.')

# ensure typechecking
from torchtyping import patch_typeguard

patch_typeguard()

import torch

from algos.vpg.agent import ActorCritic
from algos.vpg.environment import TensorPong
from algos.vpg.hyperparameters import HID_DIM, LR
from algos.vpg.train import train_one_epoch

device = torch.device('cpu')


def test_integration():
    """
    Run a few epochs of training to confirm dimension typechecks
    and

    TODO also confirm, e.g., value loss reduction
    """
    env = TensorPong(name='ALE/Pong-v5', render_mode='rgb_array')
    agent = ActorCritic(action_dim=env.action_space.n, hid_dim=HID_DIM, device=device)
    actor_opt = torch.optim.Adam(agent.actor.parameters(), lr=LR)
    critic_opt = torch.optim.Adam(agent.critic.parameters(), lr=LR)

    for epoch in range(3):
        rewards, length = train_one_epoch(env, agent, actor_opt, critic_opt)
