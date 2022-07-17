import sys
sys.path.insert(0, '.')
import logging

import torch

import algos.common.utils as utils
from algos.common.agent import ActorCritic
from algos.common.environment import TensorPong
from algos.common.hyperparameters import HID_DIM, LR, EPOCHS
from algos.trpo.train import train_one_epoch

utils.initialize_logging()

if __name__ == '__main__':
    logging.info('Beginning setup')
    env = TensorPong(name='ALE/Pong-v5', render_mode='rgb_array')
    agent = ActorCritic(action_dim=env.action_space.n, hid_dim=HID_DIM, device=utils.device)
    critic_opt = torch.optim.Adam(agent.critic.parameters(), lr=LR)

    logging.info('Beginning training')
    try:
        for epoch in range(EPOCHS):
            rewards, length = train_one_epoch(env, agent, critic_opt)
            logging.info(f'{epoch} Rewards:{rewards}, Length:{length}')
    finally:
        logging.info('Done. Saving model.')
        torch.save(agent, '/tmp/trpo_agent.pt')
        torch.save(critic_opt, '/tmp/trpo_critic_opt.pt')
