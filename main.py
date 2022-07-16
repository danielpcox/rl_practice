import torch
import typer

from algos.common import utils
from algos.common.environment import TensorPong
from algos.common.agent import ActorCritic
from algos.common.hyperparameters import HID_DIM


def main(model_path: str, env_name: str = 'ALE/Pong-v5'):
    print(f'Running {model_path} in {env_name}...')
    env = TensorPong(name=env_name, render_mode='human')
    agent = ActorCritic(action_dim=env.action_space.n, hid_dim=HID_DIM, device=utils.device)
    if model_path is not None:
        import sys; sys.path.insert(0, './algos/vpg') # HACK
        agent = torch.load(model_path, map_location='cpu')
    agent.eval()
    done = False
    obs = env.reset()
    with torch.no_grad():
        while not done:
            policy = agent.actor(obs)
            action = policy.probs.argmax()
            obs, _, _, _ = env.step(action)


if __name__ == '__main__':
    typer.run(main)
