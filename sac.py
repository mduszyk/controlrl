from collections import deque
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


def mlp(input_dim: int, output_dim: int, hidden_size: int, num_hidden: int) -> nn.Module:
    model = nn.Sequnetial()
    for _ in range(num_hidden):
        model.append(nn.Linear(input_dim, hidden_size))
        model.append(nn.ReLU())
        input_dim = hidden_size
    model.append(nn.Linear(input_dim, output_dim))
    return model


def grad_step(optimizer, loss):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def update_params(src_net: nn.Module, dst_net: nn.Module, tau: Optional[float] = None):
    for param_src, param_dst in zip(src_net.parameters(), dst_net.parameters()):
        if tau is not None:
            param_dst.data.copy_(param_dst.data * (1 - tau) + param_src.data * tau)
        else:
            param_dst.data.copy_(param_src.data)


def sample(buffer, batch_size):
    state_batch, action_batch, reward_batch, next_state_batch = [], [], [], []
    for s, a, r, next_s in random.sample(buffer, batch_size):
        state_batch.append(s)
        action_batch.append(a)
        reward_batch.append(r)
        next_state_batch.append(next_s)
    state_batch = torch.tensor(state_batch)
    action_batch = torch.tensor(action_batch)
    reward_batch = torch.tensor(reward_batch)
    next_state_batch = torch.tensor(next_state_batch)
    # normalize rewards
    reward_batch = (reward_batch - reward_batch.mean()) / (reward_batch.std() + 1e-7)
    return state_batch, action_batch, reward_batch, next_state_batch


class Actor:

    def __init__(self, state_dim, action_dim, lr):
        self.policy_net = mlp(state_dim, 2 * action_dim, 256, 2)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.action_dim = action_dim
        self.log_std_min = -20
        self.log_std_max = 2

    def deterministic_action(self, state):
        out = self.policy_net(state)
        u_mean = out[:, :self.action_dim]
        return torch.tanh(u_mean)

    def stochastic_action(self, state):
        out = self.policy_net(state)
        u_mean = out[:, :self.action_dim]
        u_log_std = out[:, self.action_dim:]
        u_log_std = torch.clamp(u_log_std, self.log_std_min, self.log_std_max)
        u_std = torch.exp(u_log_std)
        # sample action
        u = u_mean + u_std * torch.normal(0, 1, size=u_mean.shape)
        return torch.tanh(u), u_mean, u_std

    def action_and_log_prob(self, state):
        action, u_mean, u_std = self.stochastic_action(state)
        pdf_u = Normal(u_mean, u_std)
        log_prob = pdf_u.log_prob(u) - torch.log(1 - action ** 2 + 1e-6)
        return action, log_prob


class Critic:

    def __init__(self, state_dim, action_dim, lr):
        self.v = mlp(state_dim + action_dim, 1, 256, 2)
        self.v_target = mlp(state_dim + action_dim, 1, 256, 2)
        self.q1 = mlp(state_dim + action_dim, 1, 256, 2)
        self.q2 = mlp(state_dim + action_dim, 1, 256, 2)
        self.optimizer_v = torch.optim.Adam(self.v.parameters(), lr)
        self.optimizer_q1 = torch.optim.Adam(self.q1.parameters(), lr)
        self.optimizer_q2 = torch.optim.Adam(self.q2.parameters(), lr)


class SAC:

    def __init__(self, params):
        self.params = params
        self.buffer = deque(maxlen=params.buffer_max_size)
        self.actor = Actor(state_dim, action_dim, params.actor_lr)
        self.critic = Critic(state_dim, action_dim, params.critic_lr)

    def action(self, state):
        if self.actor.policy_net.training:
            return self.actor.stochastic_action(state)
        return self.actor.stochastic_action(state)

    def train_step(self):
        s_batch, a_batch, r_batch, next_s_batch = sample(self.buffer, self.params.batch_size)

        loss_v = self.v_loss(s_batch)
        grad_step(self.critic.optimizer_v, loss_v)

        loss_q1 = self.q_loss(self.critic.q1, s_batch, a_batch, r_batch, s_prime_batch)
        grad_step(self.critic.optimizer_q1, loss_q1)

        loss_q2 = self.q_loss(self.critic.q2, s_batch, a_batch, r_batch, s_prime_batch)
        grad_step(self.critic.optimizer_q2, loss_q2)

        update_params(self.critic.v, self.critic.v_target, tau=self.params.tau)

        loss_policy = self.policy_loss(s_batch)
        grad_step(self.actor.optimizer, loss_policy)

    def v_loss(self, s_batch):
        v = self.critic.v(s_batch)
        actions, log_probs = self.actor.action_and_log_prob(s_batch)
        q = self.min_q(s_batch, actions)
        return ((v - q + log_probs) ** 2).mean()

    def q_loss(self, q_net, s_batch, a_batch, r_batch, s_prime_batch):
        q_input = torch.cat(tensors=(s_batch, a_batch), dim=1)
        q = q_net(q_input)
        q_hat = self.reward_scale * r_batch + self.gamma * self.critic.v_target(s_prime_batch)
        return ((q - q_hat) ** 2).mean()

    def min_q(self, s_batch, a_batch):
        q_input = torch.cat(tensors=(s_batch, a_batch), dim=1)
        q1 = self.critic.q1(q_input)
        q2 = self.critic.q2(q_input)
        return torch.min(q1, q2)

    def policy_loss(self, s_batch):
        actions, log_probs = self.actor.action_and_log_prob(s_batch)
        q = self.min_q(s_batch, actions)
        return (log_probs - q).mean()
