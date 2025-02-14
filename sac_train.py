import json
import logging

import gymnasium as gym
import mlflow
import paramflow as pf
import torch
from gymnasium.wrappers import RecordVideo, NumpyToTorch

from sac import SAC


def run_episode(env, agent, train: bool = False):
    done = False
    state, _ = env.reset()
    payoff = 0
    t = 0
    while not done:
        t += 1
        action = agent.action(state, train)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if train:
            agent.add_transition(state, action, reward, next_state)
            agent.train_step()
        state = next_state
        payoff += reward
    return payoff, t


def train_agent(env, agent, num_episodes, checkpoint_freq):
    for episode in range(1, num_episodes + 1):
        payoff, episode_time = run_episode(env, agent, train=True)
        logging.info('episode %d, payoff: %f', episode, payoff)
        metrics = agent.stats.get(reset=True)
        metrics['payoff'] = payoff
        metrics['time'] = episode_time
        mlflow.log_metrics(metrics, episode)
        if episode % checkpoint_freq == 0:
            mlflow.pytorch.log_model(agent.actor.policy_net, f'policy_net_episode_{episode}')


def main():
    logging.basicConfig(format='%(asctime)s %(module)s %(levelname)s %(message)s', level=logging.INFO)
    params = pf.load('sac.toml')
    mlflow.set_experiment('SAC')
    logging.info(json.dumps(params, indent=4))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info('device: %s', device)

    env = gym.make(params.gym_env_id, render_mode='rgb_array')
    env = NumpyToTorch(env, device)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = SAC(state_dim, action_dim, params, device)

    with mlflow.start_run(run_name=params.gym_env_id):
        mlflow.log_params(params)
        train_agent(env, agent, params.train_episodes, params.checkpoint_freq)

    env = RecordVideo(env, name_prefix=params.gym_env_id, video_folder='videos', episode_trigger=lambda e: True)
    with torch.no_grad():
        payoff, episode_time = run_episode(env, agent)
        logging.info('test episode payoff: %f, episode_time: %d', payoff, episode_time)
    env.close()


if __name__ == '__main__':
    main()
