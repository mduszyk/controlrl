import json
import logging

import gymnasium as gym
import mlflow
import paramflow as pf
import torch
from gymnasium.wrappers import RecordVideo, NumpyToTorch, RescaleAction

from sac import SAC


def run_episode(env, agent, train: bool = False):
    done = False
    state, _ = env.reset()
    payoff = 0
    t = 0
    while not done:
        t += 1
        action = agent.action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if train:
            agent.add_transition(state, action, reward, next_state)
            agent.train_step()
        state = next_state
        payoff += reward
    return payoff, t


def train_agent(env, agent, num_episodes, checkpoint_freq):
    step = 0
    for episode in range(1, num_episodes + 1):
        payoff, episode_len = run_episode(env, agent, train=True)
        step += episode_len
        logging.info('episode %d payoff %f len %d buf %d', episode, payoff, episode_len, len(agent.buffer))
        metrics = agent.stats.get(reset=True)
        metrics['payoff'] = payoff
        metrics['length'] = episode_len
        metrics['buffer'] = len(agent.buffer)
        mlflow.log_metrics(metrics, step)
        if episode % checkpoint_freq == 0:
            mlflow.pytorch.log_model(agent.actor.policy_net, f'policy_net_episode_{episode}')


def main():
    logging.basicConfig(format='%(asctime)s %(module)s %(levelname)s %(message)s', level=logging.INFO)
    params = pf.load('sac_train.toml')
    mlflow.set_experiment('SAC')
    logging.info(json.dumps(params, indent=4))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info('device: %s', device)

    env = gym.make(params.gym_env_id, render_mode='rgb_array')
    env = RescaleAction(env, min_action=-1, max_action=1)
    env = NumpyToTorch(env, device)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    dtype = getattr(torch, params.dtype)
    agent = SAC(state_dim, action_dim, params, device, dtype)

    with mlflow.start_run(run_name=params.gym_env_id):
        mlflow.log_params(params)
        train_agent(env, agent, params.train_episodes, params.checkpoint_freq)

    name_prefix = f'{params.gym_env_id}_{params.train_episodes}'
    env = RecordVideo(env, name_prefix=name_prefix, video_folder=params.videos_dir, episode_trigger=lambda e: True)
    with torch.no_grad():
        payoff, episode_time = run_episode(env, agent)
        logging.info('test episode payoff: %f, episode_time: %d', payoff, episode_time)
    env.close()


if __name__ == '__main__':
    main()
