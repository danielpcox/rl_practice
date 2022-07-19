import importlib
import logging

import torch
import typer

app = typer.Typer()

from algos.common import utils
from algos.common.environment import TensorPong
from algos.common.agent import ActorCritic
import algos.common.hyperparameters as hyp

utils.initialize_logging()



@app.command()
def train(algorithm: str):
    env = TensorPong(name='ALE/Pong-v5', render_mode='rgb_array')
    agent = ActorCritic(action_dim=env.action_space.n, hid_dim=hyp.HID_DIM, device=utils.device)
    actor_opt = torch.optim.Adam(agent.actor.parameters(), lr=hyp.LR)
    critic_opt = torch.optim.Adam(agent.critic.parameters(), lr=hyp.LR)

    module = importlib.import_module(f'algos.{algorithm}.train')
    train_one_epoch = getattr(module, 'train_one_epoch')

    logging.info(f'Beginning training with {algorithm.upper()}')
    try:
        for epoch in range(hyp.EPOCHS):
            rewards, length = train_one_epoch(env, agent, actor_opt, critic_opt)
            logging.info(f'{epoch} Rewards:{rewards}, Length:{length}')
    finally:
        logging.info('Done. Saving model.')
        torch.save(agent, '/tmp/agent.pt')


# TODO fix this now that there's more than one algorithm
@app.command()
def run(model_path: str):
    # segments = algorithm.split('.')
    # mod = importlib.import_module('.'.join(segments[:-1]))
    # klass = getattr(mod, segments[-1])
    # instance = klass()
    env = TensorPong(name='ALE/Pong-v5', render_mode='human')
    agent = torch.load(model_path, map_location='cpu')
    agent.eval()
    done, obs = False, env.reset()
    with torch.no_grad():
        while not done:
            policy, _ = agent.actor(obs)
            action = policy.probs.argmax()
            obs, _, _, _ = env.step(action)


if __name__ == '__main__':
    app()
