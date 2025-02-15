import json
import logging

import gymnasium as gym
import mlflow
import paramflow as pf
import torch
from gymnasium.wrappers import RecordVideo, NumpyToTorch, RescaleAction

from sac import Actor
from sac_train import run_episode
from stats import Stats


def test_agent(env, agent, num_episodes):
    stats = Stats()
    for episode in range(1, num_episodes + 1):
        payoff, episode_len = run_episode(env, agent)
        logging.info('episode %d payoff %f len %d', episode, payoff, episode_len)
        stats.update('payoff', payoff)
        stats.update('episode_len', episode_len)
    return stats.get()

def main():
    logging.basicConfig(format='%(asctime)s %(module)s %(levelname)s %(message)s', level=logging.INFO)
    params = pf.load('sac_train.toml', 'sac_eval.toml')
    logging.info(json.dumps(params, indent=4))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info('device: %s', device)

    policy_net = mlflow.pytorch.load_model(params.model_uri, map_location=device)
    policy_net.eval()

    env = gym.make(params.gym_env_id, render_mode='rgb_array')
    env = RescaleAction(env, min_action=-1, max_action=1)
    env = NumpyToTorch(env, device)

    action_dim = env.action_space.shape[0]
    agent = Actor(action_dim, policy_net)

    stats = test_agent(env, agent, params.test_episodes)

    model_episode = params.model_uri.split('_')[-1]
    name_prefix = f'{params.gym_env_id}_{model_episode}'
    env = RecordVideo(env, name_prefix=name_prefix, video_folder=params.videos_dir, episode_trigger=lambda e: True)
    with torch.no_grad():
        payoff, episode_time = run_episode(env, agent)
        logging.info('video episode, payoff: %f, time: %d', payoff, episode_time)
    env.close()

    logging.info('test stats:\n%s', json.dumps(stats, indent=4))


if __name__ == '__main__':
    main()
