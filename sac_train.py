import json
import logging

import gymnasium as gym
import paramflow as pf
import torch
from gymnasium.wrappers import RecordVideo, NumpyToTorch

from sac import SAC


def run_episode(env, agent, train: bool = False):
    done = False
    state, _ = env.reset()
    payoff = 0
    while not done:
        action = agent.action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if train:
            agent.add_transition(state, action, reward, next_state)
            agent.train_step()
        state = next_state
        payoff += reward
    return payoff


def train_agent(env, agent, num_episodes):
    for episode in range(num_episodes):
        payoff = run_episode(env, agent, train=True)
        stats = agent.stats.get(reset=True)
        logging.info('episode %d, payoff: %f, %s', episode, payoff, stats)


def main():
    logging.basicConfig(format='%(asctime)s %(module)s %(levelname)s %(message)s', level=logging.INFO)
    params = pf.load('sac.toml')
    logging.info(json.dumps(params, indent=4))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info('device: %s', device)
    env = gym.make('Pendulum-v1', render_mode='rgb_array')
    env = NumpyToTorch(env, device)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = SAC(state_dim, action_dim, params, device)
    train_agent(env, agent, params.train_episodes)
    env = RecordVideo(env, name_prefix='Pendulum-v1', video_folder='videos', episode_trigger=lambda e: True)
    with torch.no_grad():
        payoff = run_episode(env, agent)
        logging.info('test episode payoff: %f', payoff)
    env.close()


if __name__ == '__main__':
    main()
