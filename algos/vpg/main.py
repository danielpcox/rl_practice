import logging
import signal
import sys

import setup
import torch

from agent import ActorCritic
from environment import TensorPong
from hyperparameters import HID_DIM, LR, EPOCHS
from train import train_one_epoch

setup.initialize_logging()



if __name__ == '__main__':
    logging.info('Beginning setup')
    env = TensorPong(name='ALE/Pong-v5', render_mode='rgb_array')
    agent = ActorCritic(action_dim=env.action_space.n, hid_dim=HID_DIM, device=setup.device)
    actor_opt = torch.optim.Adam(agent.actor.parameters(), lr=LR)
    critic_opt = torch.optim.Adam(agent.critic.parameters(), lr=LR)

    logging.info('Beginning training')
    try:
        for epoch in range(EPOCHS):
            rewards, length = train_one_epoch(env, agent, actor_opt, critic_opt)
            logging.info(f'{epoch} Rewards:{rewards}, Length:{length}')
    finally:
        logging.info('Done. Saving model.')
        torch.save(agent, '/tmp/agent.pt')
        torch.save(actor_opt, '/tmp/actor_opt.pt')
        torch.save(critic_opt, '/tmp/critic_opt.pt')