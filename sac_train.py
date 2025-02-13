import torch

import gymnasium as gym
from gymnasium.wrappers import RecordVideo

import paramflow as pf

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
            transition = state, action, reward, next_state
            agent.buffer.append(transition)
            agent.train_step()
        state = next_state
        payoff += reward
    return payoff


def train_agent(env, agent, num_episodes):
    payoff = 0
    for episode in range(num_episodes):
        payoff += run_episode(env, agent, train=True)
    return payoff


def main():
    props = pf.load('sac.toml')
    print(props)
    agent = SAC(params)
    env = gym.make('Pendulum-v1', render_mode='rgb_array')
    train_agent(env, agent, 10)
    env = RecordVideo(env, video_folder='videos', episode_trigger=lambda e: True)
    with torch.no_grad():
        run_episode(env, agent)
    env.close()


if __name__ == '__main__':
    main()
