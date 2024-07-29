import torch
import numpy as np


class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)

    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)


class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x, update=True):
        # Whether to update the mean and std. During the evaluating, update=False.
        if update:
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)

        return x


class RewardScaling:
    def __init__(self, shape, gamma):
        self.shape = shape  # reward shape=1
        self.gamma = gamma  # discount factor
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape)

    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)  # Only divided std
        return x

    def reset(self):  # When an episode is done,we should reset 'self.R'
        self.R = np.zeros(self.shape)


def reward_reshape(rewards):
    """
    Reshape the rewards of an episode. Return reshaped_rewards.
    """
    return rewards[:-1] - rewards[1:]


def reshape_reward(rewards, valid_steps):
    use_steps = valid_steps + 1
    # Use `loss_decrease` as reward
    episode_rewards = []
    for i in range(rewards.size(0)):
        valid_rewards = rewards[i][:use_steps[i]]
        loss_decrease = valid_rewards[:-1] - valid_rewards[1:]
        episode_rewards.append(loss_decrease.cpu().numpy())
    return episode_rewards


def reward_scaling(rewards, reward_scaler):
    scaled_rewards = []
    for reward in rewards:
        scaled_rewards.append(reward_scaler(reward))
    scaled_rewards = np.stack(scaled_rewards, axis=0)
    reward_scaler.reset()  # When an episode is done, we should reset 'self.R'

    return torch.from_numpy(scaled_rewards)


def compute_adv_ret(args, critic, states, rewards, next_states, dones, nodes_per_graph, reward_trasnform):
    """
    Calculate advantage and return-to-go of an epsiode.
    """
    device = states[0].device
    num_episode = len(rewards)
    
    if reward_trasnform.running_ms.n == 0:
        for i in range(num_episode):
            for reward in rewards[i]:
                reward_trasnform.running_ms.update(reward)
    scaled_rewards = []
    for i in range(num_episode):
        step_scaled_rewards = []
        for reward in rewards[i]:
            step_scaled_rewards.append(reward_trasnform(reward))
        scaled_rewards.append(torch.tensor(np.stack(step_scaled_rewards)).squeeze(-1).float().to(device))

    states = torch.split(states, nodes_per_graph); next_states = torch.split(next_states, nodes_per_graph); dones = torch.split(dones, nodes_per_graph)
    # compute advantage by GAE
    advs, rets = [], []
    with torch.no_grad():
        for i in range(num_episode):
            state_values = critic(states[i])
            next_state_values = critic(next_states[i])
            gae = torch.tensor(0).to(device)
            adv = []
            deltas = scaled_rewards[i] + args.gamma * (1.0 - dones[i]) * next_state_values - state_values
            for delta, done in zip(reversed(deltas), reversed(dones[i])):
                gae = delta + args.gamma * args.lam * gae * (1.0 - done)
                adv.insert(0, gae)
            adv = torch.stack(adv)
            ret = adv + state_values
            advs.append(adv)
            rets.append(ret)

    return torch.cat(advs), torch.cat(rets), torch.stack([ret[0] for ret in rets]), scaled_rewards
