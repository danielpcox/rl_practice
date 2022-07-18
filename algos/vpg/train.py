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
    tau = dotdict({k: torch.cat([traj[k] for traj in D]) for k in D[0].keys()})

    # advantages and rewards-to-go
    tau.adv, tau.rtg = get_ground_truths(tau)

    actor_loss = -(tau.logp * tau.adv).mean()
    actor_opt.zero_grad()
    actor_loss.backward()
    actor_opt.step()

    # re-run obs through model for multiple critic update epochs
    for ve in range(V_EPOCHS):
        critic_loss = (tau.v - tau.rtg).pow(2).mean()
        critic_opt.zero_grad()
        critic_loss.backward()
        critic_opt.step()
        tau.v = agent.critic(tau.o)

    return tau.r.sum().item(), len(D)
