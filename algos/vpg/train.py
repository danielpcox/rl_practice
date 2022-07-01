import torch
import scipy.signal
import setup
from hyperparameters import GAMMA
from pdb import set_trace
from agent import ActorCritic

def get_returns(rewards, discount):
    # https://stackoverflow.com/a/47971187/379547
    r = rewards[::-1]
    a = [1, -discount]
    b = [1]
    y = scipy.signal.lfilter(b, a, x=r)
    y = y[::-1].copy()

    return torch.as_tensor(y, dtype=torch.float32, device=setup.device)

def get_ground_truths(rewards, values):
    """

    :param rewards: list of rewards encountered during environment interaction
    :param values: list of critic-estimated values during environment interaction
    :return: advantages and rewards_to_go
    """
    reward_to_go = 0.
    rewards_to_go = []
    adv_acc = 0.
    advantages = []
    V_t1 = 0.

    for i, reward in enumerate(reversed(rewards)):
        V_t = values[i]
        reward_to_go = reward + GAMMA*reward_to_go
        rewards_to_go.append(reward_to_go)
        td_error = (reward + GAMMA*V_t1) - V_t
        adv_acc = td_error + GAMMA*adv_acc
        advantages.append(adv_acc)
        V_t1 = V_t

    return torch.as_tensor(list(reversed(advantages)), dtype=torch.float32),\
           torch.as_tensor(list(reversed(rewards_to_go)), dtype=torch.float32)

def get_losses(logprobs, rewards, values):
    """
    :param logprobs: list of dimensionless tensor logprobs
    :param rewards: list of dimensionless tensor rewards
    :param values: list of dimensionless tensor values
    """

    logprobs = torch.stack(logprobs)
    rewards = torch.stack(rewards)
    values = torch.stack(values)

    advantages, rewards_to_go = get_ground_truths(rewards, values)

    actor_loss = -(logprobs * advantages).mean()
    critic_loss = (values - rewards_to_go).pow(2).mean()
    # TODO add entropy bonus
    return actor_loss, critic_loss


def train_one_epoch(env, agent: ActorCritic, actor_opt, critic_opt):
    obs = env.reset()
    done = False
    rewards = []
    logprobs = []
    values = []

    # TODO don't wait until it's done - update periodically
    while not done:
        # action, logp, value = agent(obs)
        policy, value = agent(obs)
        action = policy.sample()
        logp = policy.log_prob(action)
        obs, reward, done, info = env.step(action)
        rewards.append(reward)
        logprobs.append(logp)
        values.append(value)

    actor_opt.zero_grad()
    critic_opt.zero_grad()

    actor_loss, critic_loss = get_losses(logprobs, rewards, values)

    actor_loss.backward()
    actor_opt.step()

    # TODO multiple critic training epochs?
    critic_loss.backward()
    critic_opt.step()

    return torch.as_tensor(rewards).sum().item(), len(rewards)
