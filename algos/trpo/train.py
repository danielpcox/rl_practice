from functools import partial

import torch
from torch import autograd
from torch.distributions import Categorical, kl_divergence
from torch.nn.utils import parameters_to_vector

from algos.common import utils
from algos.common.utils import dotdict
import algos.common.hyperparameters as hyp
from algos.common.agent import ActorCritic
from torchtyping import patch_typeguard, TensorType as T
from typeguard import typechecked
patch_typeguard()

def reevaluate(agent, τ):
    pi = agent.actor(τ.o)
    ratio = pi.log_prob(τ.a) - τ.logp.detach()
    loss = -(ratio * τ.adv).mean() # importance-sampled policy loss
    # explicit gradient repo
    g = parameters_to_vector(autograd.grad(loss, agent.actor.parameters(), retain_graph=True))
    pi_old = Categorical(logits=τ.logits.detach())
    Dkl = kl_divergence(pi_old, pi)
    return loss, g, Dkl

def train_one_epoch(env, agent: ActorCritic, actor_opt, critic_opt):
    obs = env.reset()
    done = False
    D = []

    while not done:
        action, logp, logits, value = agent(obs)
        obs, reward, done, info = env.step(action)
        D.append({'o': obs, 'a': action, 'r': reward, 'logp': logp, 'logits': logits, 'v': value})

    τ = dotdict({k: torch.stack([traj[k] for traj in D]) for k in D[0].keys()})

    # advantages and rewards-to-go
    τ.adv, τ.rtg = utils.get_ground_truths(τ)

    # actor_loss = -(τ.logp * τ.adv).mean()
    # actor_opt.zero_grad()
    # actor_loss.backward()
    # actor_opt.step()

    ### TRPO
    loss, g, Dkl = reevaluate(agent, τ)
    Hs = partial(utils.hessian_vector_product, Dkl, agent.actor.parameters())
    s = utils.conjugate_gradient(Hs, g)

    for j in range(hyp.BACKTRACK_ITERS):
        step_mult = hyp.TRPO_ALPHA ** j
        # TODO implement
        pass

    ### /TRPO

    # re-run obs through model for multiple critic update epochs
    for ve in range(hyp.V_EPOCHS):
        critic_loss = (τ.v - τ.rtg).pow(2).mean()
        critic_opt.zero_grad()
        critic_loss.backward()
        critic_opt.step()
        τ.v = agent.critic(τ.o)

    return τ.r.sum().item(), len(D)
