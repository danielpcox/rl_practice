import logging
from functools import partial

import torch
from torch import autograd
from torch.distributions import Categorical, kl_divergence
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from algos.common import utils
from algos.common.utils import dotdict
import algos.common.hyperparameters as hyp
from algos.common.agent import ActorCritic
from torchtyping import patch_typeguard, TensorType as T
from typeguard import typechecked

patch_typeguard()


def reevaluate(agent, tau, pi_old):
    pi = agent.actor(tau.o)
    ratio = (pi.log_prob(tau.a) - tau.logp).exp()
    loss = -(ratio * tau.adv).mean()  # importance-sampled policy loss
    Dkl = kl_divergence(pi_old, pi).mean()
    return pi, loss, Dkl


def train_one_epoch(env, agent: ActorCritic, actor_opt, critic_opt):
    obs = env.reset()
    done = False
    D = []

    # Play the game and record data
    while not done:
        action, logp, logits, value = agent(obs)
        obs, reward, done, info = env.step(action)
        D.append({'o': obs, 'a': action, 'r': reward, 'logp': logp.detach(), 'logits': logits.detach(), 'v': value})

    tau = dotdict({k: torch.cat([traj[k] for traj in D]) for k in D[0].keys()})

    # advantages and rewards-to-go
    tau.adv, tau.rtg = utils.get_ground_truths(tau)

    #########################
    # TRPO actor optimization
    #########################
    pi_old = Categorical(logits=tau.logits)
    pi, loss, Dkl = reevaluate(agent, tau, pi_old)

    # get search direction s
    g = parameters_to_vector(autograd.grad(loss, agent.actor.parameters(), retain_graph=True))
    Hs = partial(utils.hessian_vector_product, Dkl, agent.actor)
    s = utils.conjugate_gradient(Hs, g)

    # get max step size beta, original parameters theta, and original loss
    beta = torch.sqrt(2 * hyp.MAX_Dkl / (torch.dot(s, Hs(s)) + hyp.NEVER_DIV0)).item()
    theta_old = parameters_to_vector(agent.actor.parameters()).detach()
    old_loss = loss.item()

    with torch.no_grad():
        for j in range(hyp.BACKTRACK_ITERS):
            step = hyp.TRPO_ALPHA ** j
            vector_to_parameters(theta_old + step * beta * s, agent.actor.parameters())

            pi, loss, Dkl = reevaluate(agent, tau, pi_old)

            if loss <= old_loss and Dkl.item() <= hyp.MAX_Dkl:
                break
            elif j == hyp.BACKTRACK_ITERS - 1:
                logging.info('Failed to satisfy constraints - no update this episode')
                vector_to_parameters(theta_old, agent.actor.parameters())

    ##########################
    # /TRPO actor optimization
    ##########################

    # re-run obs through model for multiple critic update epochs
    for ve in range(hyp.V_EPOCHS):
        critic_loss = (tau.v - tau.rtg).pow(2).mean()
        critic_opt.zero_grad()
        critic_loss.backward()
        critic_opt.step()
        tau.v = agent.critic(tau.o)

    return tau.r.sum().item(), len(D)
