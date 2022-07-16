import torch

from algos.common.utils import dotdict, get_ground_truths
from algos.common.hyperparameters import V_EPOCHS, NEVER_DIV0
from algos.common.agent import ActorCritic

def train_one_epoch(env, agent: ActorCritic, actor_opt, critic_opt):
    obs = env.reset()
    done = False
    D = []

    # TODO don't wait until it's done - update periodically
    while not done:
        action, logp, _, value = agent(obs)
        obs, reward, done, info = env.step(action)
        D.append({'o': obs, 'r': reward, 'logp': logp, 'v': value})

    # stack up D's tensors for vector computations
    # list[dict[str,Tensor] -> dict[str, Tensor]
    τ = dotdict({k: torch.stack([traj[k] for traj in D]) for k in D[0].keys()})

    # advantages and rewards-to-go
    τ.adv, τ.rtg = get_ground_truths(τ)

    actor_loss = -(τ.logp * τ.adv).mean()
    actor_opt.zero_grad()
    actor_loss.backward()
    # torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 1)
    actor_opt.step()

    # re-run obs through model for multiple critic update epochs
    for ve in range(V_EPOCHS):
        critic_loss = (τ.v - τ.rtg).pow(2).mean()
        critic_opt.zero_grad()
        critic_loss.backward()
        critic_opt.step()
        τ.v = agent.critic(τ.o)

    return τ.r.sum().item(), len(D)
