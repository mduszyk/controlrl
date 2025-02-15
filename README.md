## SAC
Implementation of Soft Actor-Critic (SAC) reinforcement learning algorithm.

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
```