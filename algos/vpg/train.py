import logging

import torch

from common import dotdict
from hyperparameters import GAMMA, LAMBDA, V_EPOCHS
from pdb import set_trace
from agent import ActorCritic

def get_ground_truths(τ):
    reward_to_go = 0.
    advantage = 0.
    V_t1 = 0.
    rewards_to_go = []
    advantages = []

    for i in reversed(range(len(τ.r))):
        reward_to_go = τ.r[i] + GAMMA*reward_to_go
        rewards_to_go.append(reward_to_go)

        V_t = τ.v[i]
        td_error = (τ.r[i] + GAMMA*V_t1) - V_t
        advantage = td_error + GAMMA*LAMBDA*advantage
        advantages.append(advantage)

        V_t1 = V_t

    rewards_to_go = torch.as_tensor(list(reversed(rewards_to_go)), dtype=torch.float32).unsqueeze(dim=1)
    advantages = torch.as_tensor(list(reversed(advantages)), dtype=torch.float32).unsqueeze(dim=1)

    return advantages, rewards_to_go


def train_one_epoch(env, agent: ActorCritic, actor_opt, critic_opt):
    obs = env.reset()
    done = False
    D = []

    # TODO don't wait until it's done - update periodically
    while not done:
        action, logp, value = agent(obs)
        obs, reward, done, info = env.step(action)
        D.append({'o':obs, 'r':reward, 'logp':logp, 'v':value})

    # stack up D's tensors for vector computations
    # list[dict[str,Tensor] -> dict[str, Tensor]
    τ = dotdict({k:torch.stack([traj[k] for traj in D]) for k in D[0].keys()})

    advantages, rewards_to_go = get_ground_truths(τ)

    # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    actor_loss = -(τ.logp * advantages).mean()
    actor_opt.zero_grad()
    actor_loss.backward()
    # torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 1)
    actor_opt.step()

    # re-run obs through model for multiple critic update epochs
    for ve in range(V_EPOCHS):
        critic_loss = (τ.v - rewards_to_go).pow(2).mean()
        critic_opt.zero_grad()
        critic_loss.backward()
        critic_opt.step()
        τ.v = agent.critic(τ.o)

    return τ.r.sum().item(), len(D)
