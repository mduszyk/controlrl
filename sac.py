import random
from collections import deque, namedtuple
from typing import Optional

import torch
from torch import nn
from torch.distributions import Normal

from stats import Stats

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state'])


def mlp(input_dim: int, output_dim: int, hidden_size: int, num_hidden: int) -> nn.Module:
    """Create multi layer perceptron."""
    model = nn.Sequential()
    for _ in range(num_hidden):
        model.append(nn.Linear(input_dim, hidden_size))
        model.append(nn.ReLU())
        input_dim = hidden_size
    model.append(nn.Linear(input_dim, output_dim))
    return model


def grad_step(opt, loss):
    opt.zero_grad()
    loss.backward()
    opt.step()


def update_params(src_net: nn.Module, dst_net: nn.Module, tau: Optional[float] = None):
    for param_src, param_dst in zip(src_net.parameters(), dst_net.parameters()):
        if tau is not None:
            # exponentially moving average of network weights
            param_dst.data.copy_(param_dst.data * (1 - tau) + param_src.data * tau)
        else:
            param_dst.data.copy_(param_src.data)


def sample(buffer, batch_size: int, device):
    """Sample batch of transitions from buffer."""
    state_batch, action_batch, reward_batch, next_state_batch = [], [], [], []
    for state, action, reward, next_state in random.sample(buffer, batch_size):
        state_batch.append(state)
        action_batch.append(action)
        reward_batch.append(reward)
        next_state_batch.append(next_state)
    state_batch = torch.stack(state_batch).to(device, non_blocking=True)
    action_batch = torch.stack(action_batch).to(device, non_blocking=True)
    reward_batch = torch.tensor(reward_batch).view(batch_size, 1).to(device, non_blocking=True)
    next_state_batch = torch.stack(next_state_batch).to(device, non_blocking=True)
    # normalize rewards
    reward_batch = (reward_batch - reward_batch.mean()) / (reward_batch.std() + 1e-7)
    return Transition(state_batch, action_batch, reward_batch, next_state_batch)


class Actor:

    def __init__(self, action_dim: int, policy_net: nn.Module, optimizer=None):
        self.policy_net = policy_net
        self.optimizer = optimizer
        self.action_dim = action_dim
        self.log_std_min = -20
        self.log_std_max = 2

    @torch.no_grad()
    def action(self, state: torch.Tensor):
        """Return deterministic action, the mean of the action distribution."""
        out = self.policy_net(state)
        u_mean = out[..., :self.action_dim]
        return torch.tanh(u_mean)

    def _sample_policy_net(self, state):
        out = self.policy_net(state)
        u_mean = out[..., :self.action_dim]
        u_log_std = out[..., self.action_dim:]
        # for numerical stability
        u_log_std = torch.clamp(u_log_std, self.log_std_min, self.log_std_max)
        u_std = torch.exp(u_log_std)
        # reparameterization trick
        u = u_mean + u_std * torch.randn(u_mean.shape, device=u_mean.device)
        return u, u_mean, u_std

    def stochastic_action(self, state):
        u, _, _ = self._sample_policy_net(state)
        # use tanh in order to squash infinite support of Gaussian into [-1, 1]
        action = torch.tanh(u)
        return action

    def action_and_log_prob(self, state):
        u, u_mean, u_std = self._sample_policy_net(state)
        action = torch.tanh(u)
        # calculate log prob of the action via change of variables
        pdf_u = Normal(u_mean, u_std)
        log_prob_action = pdf_u.log_prob(u) - torch.log(1 - action ** 2 + 1e-6)
        return action, log_prob_action


class Critic:

    def __init__(self, state_dim, action_dim, lr, device, dtype):
        self.v = mlp(state_dim, 1, 256, 2).to(dtype).to(device)
        self.v_target = mlp(state_dim, 1, 256, 2).to(dtype).to(device)
        # v and v_target start with the same initialization
        update_params(self.v, self.v_target)
        self.q1 = mlp(state_dim + action_dim, 1, 256, 2).to(dtype).to(device)
        self.q2 = mlp(state_dim + action_dim, 1, 256, 2).to(dtype).to(device)
        self.optimizer_v = torch.optim.Adam(self.v.parameters(), lr)
        self.optimizer_q1 = torch.optim.Adam(self.q1.parameters(), lr)
        self.optimizer_q2 = torch.optim.Adam(self.q2.parameters(), lr)


class SAC:

    def __init__(self, state_dim, action_dim, params, device, dtype):
        self.params = params
        self.device = device
        self.buffer = deque(maxlen=params.max_buffer_size)
        policy_net = mlp(state_dim, 2 * action_dim, 256, 2).to(dtype).to(device)
        policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=params.actor_lr)
        self.actor = Actor(action_dim, policy_net, policy_optimizer)
        self.critic = Critic(state_dim, action_dim, params.critic_lr, device, dtype)
        self.stats = Stats()

    @torch.no_grad()
    def action(self, state: torch.Tensor):
        return self.actor.stochastic_action(state)

    def add_transition(self, state, action, reward, next_state):
        transition = state.cpu(), action.cpu(), reward, next_state.cpu()
        self.buffer.append(transition)

    def train_step(self):
        if len(self.buffer) < self.params.min_buffer_size:
            return

        batch = sample(self.buffer, self.params.batch_size, self.device)

        loss_v = self.v_loss(batch.state)
        grad_step(self.critic.optimizer_v, loss_v)
        self.stats.update('loss_v', loss_v.item())

        loss_q1 = self.q_loss(self.critic.q1, batch)
        grad_step(self.critic.optimizer_q1, loss_q1)
        self.stats.update('loss_q1', loss_q1.item())

        loss_q2 = self.q_loss(self.critic.q2, batch)
        grad_step(self.critic.optimizer_q2, loss_q2)
        self.stats.update('loss_q2', loss_q2.item())

        update_params(self.critic.v, self.critic.v_target, tau=self.params.tau)

        loss_policy = self.policy_loss(batch.state)
        grad_step(self.actor.optimizer, loss_policy)
        self.stats.update('loss_policy', loss_policy.item())

    def v_loss(self, states):
        # Equation 5 in the paper
        v = self.critic.v(states)
        actions, log_probs = self.actor.action_and_log_prob(states)
        q = self.min_q(states, actions)
        return ((v - q + log_probs) ** 2).mean()

    def q_loss(self, q_net, batch):
        # Equation 7 in the paper
        states_actions = torch.cat(tensors=(batch.state, batch.action), dim=1)
        q = q_net(states_actions)
        q_target = self.params.reward_scale * batch.reward + self.params.gamma * self.critic.v_target(batch.next_state)
        return ((q - q_target) ** 2).mean()

    def min_q(self, states, actions):
        # Minimum of two Q-functions is used in order to mitigate
        # the positive bias in the policy improvement.
        states_actions = torch.cat(tensors=(states, actions), dim=1)
        return torch.min(self.critic.q1(states_actions), self.critic.q2(states_actions))

    def policy_loss(self, states):
        # Equation 12 in the paper
        actions, log_probs = self.actor.action_and_log_prob(states)
        q = self.min_q(states, actions)
        return (log_probs - q).mean()
