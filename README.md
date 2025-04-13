## SAC
Implementation of Soft Actor-Critic (SAC) reinforcement learning algorithm for Multi-Joint dynamics with Contact (MuJoCo).

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
