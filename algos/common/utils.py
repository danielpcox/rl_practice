import os
import sys

from torch import autograd, nn
from torch.nn.utils import parameters_to_vector

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = "1"

from torchtyping import patch_typeguard, TensorType as T
from typeguard import typechecked

patch_typeguard()

import torch
import logging

LOG_LEVEL = os.environ.get('LOG_LEVEL', 'info').upper()

import algos.common.hyperparameters as hyp


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    __dir__ = dict.keys


def initialize_logging():
    print(f'Initializing logging with level {LOG_LEVEL}')
    numeric_level = getattr(logging, LOG_LEVEL, None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % LOG_LEVEL)
    logging.basicConfig(format='%(asctime)s/%(levelname)s: %(message)s',
                        stream=sys.stdout,
                        filemode='a',
                        level=numeric_level,
                        datefmt='%Y-%m-%d %H:%M:%S%Z',
                        force=True)


useMPS = False
if useMPS and torch.backends.mps.is_available():
    device = torch.device("mps")  # mps for my M1 Mac
else:
    device = torch.device("cpu")
print(torch.__version__)
print(device)


@typechecked  # TODO it seems typechecked isn't checking return values?
def get_ground_truths(tau: dict[str, T['B'] | T['B', 'A'] | T['B', 'H', 'W']]) -> (T['B'], T['B']):
    reward_to_go = 0.
    advantage = 0.
    V_t1 = 0.
    rewards_to_go = []
    advantages = []

    for i in reversed(range(len(tau.r))):
        reward_to_go = tau.r[i] + hyp.GAMMA * reward_to_go
        rewards_to_go.append(reward_to_go)

        V_t = tau.v[i]
        td_error = (tau.r[i] + hyp.GAMMA * V_t1) - V_t
        advantage = td_error + hyp.GAMMA * hyp.LAMBDA * advantage
        advantages.append(advantage)

        V_t1 = V_t

    rewards_to_go = torch.as_tensor(list(reversed(rewards_to_go)), dtype=torch.float32)
    advantages = torch.as_tensor(list(reversed(advantages)), dtype=torch.float32)
    advantages = (advantages - advantages.mean()) / (advantages.std() + hyp.NEVER_DIV0)

    return advantages, rewards_to_go


# https://github.com/Kaixhin/spinning-up-basic/blob/f9cbe/trpo.py#L19-L21
def hessian_vector_product(M, model, x):
    """Compute the product of (the Hessian of M) and x wrt params"""
    g = parameters_to_vector(autograd.grad(M, model.parameters(), create_graph=True))
    return parameters_to_vector(autograd.grad((g * x.detach()).sum(), model.parameters(), retain_graph=True)) + hyp.Hx_DAMP * x


# https://github.com/Kaixhin/spinning-up-basic/blob/f9cbe/trpo.py#L24-L37
def conjugate_gradient(Ax, b, iters=10):
    """Let Ax = b. Given Ax, and b, approximately solve for x."""
    x = torch.zeros_like(b)
    r = b - Ax(x)  # Residual
    p = r  # Conjugate vector
    r_dot_old = torch.dot(r, r)
    for _ in range(iters):
        Ap = Ax(p)
        alpha = r_dot_old / (torch.dot(p, Ap) + hyp.NEVER_DIV0)
        x += alpha * p
        r -= alpha * Ap
        r_dot_new = torch.dot(r, r)
        p = r + (r_dot_new / r_dot_old) * p
        r_dot_old = r_dot_new
    return x
