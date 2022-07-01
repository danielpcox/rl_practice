import torch
import typer

from algos.vpg import common
from algos.vpg.environment import TensorPong
from algos.vpg.agent import ActorCritic
from algos.vpg.hyperparameters import HID_DIM


def main(env_name: str = 'ALE/Pong-v5', model_path: str = None):
    print(f'Running {model_path} in {env_name}...')
    env = TensorPong(name=env_name, render_mode='human')
    agent = ActorCritic(action_dim=env.action_space.n, hid_dim=HID_DIM, device=common.device)
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
