## SAC
This project provides an implementation of the Soft Actor-Critic (SAC)
algorithm for solving continuous control tasks using environments from
OpenAI Gym (e.g., MuJoCo). SAC is an off-policy actor-critic algorithm
that combines maximum entropy reinforcement learning with function
approximation for efficient and stable training.

See: https://gymnasium.farama.org/environments/mujoco/

### Run local mlflow
```shell
mlflow ui
```

### Train
```shell
python sac_train.py
python sac_train.py --profile ant
python sac_train.py --profile humanoid
```

### Test
```shell
python sac_eval.py
python sac_eval.py --profile ant
python sac_eval.py --profile humanoid
python sac_eval.py --profile ant --model_uri 'runs:/55db85ebf343496783f5f2b88389b604/policy_net_episode_1100'
```
