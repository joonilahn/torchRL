# torchRL

PyTorch Implementations of basic Reinforcement Learning algorithms.

* You can experiment various RL algorithms such as SARSA, DQN, DDQN, and etc.

* You can easily set up hyper-parameters for training by adjusting yaml-formatted config files.

* Default environment is OpenAI gym's "CartPole-v3".


## Algorithms
1. SARSA
4. Q-Learning
5. [DQN](https://arxiv.org/abs/1312.5602) (Playing Atari with Deep Reinforcement Learning)
6. [DDQN](https://arxiv.org/abs/1509.06461) (Deep Reinforcement Learning with Double Q-learning
)
7. [Dueling DQN](https://arxiv.org/abs/1511.06581) (Dueling Network Architectures for Deep Reinforcement Learning
)
8. Actor-Critic (Actor-Critic Policy)

## Dependencies
1. PyTorch (This package was tested on cpu version of PyTorch v1.9. But chances are you'll be fine with most of older versions of PyTorch!)
2. OpenAI GYM
3. yacs

## Usage
You can make your own training setup by creating a yaml file.

Most of hyper-parameters can be re-used by inheriting those from the base config [module](torchRL/configs/cartpole_defaults.py).

Check some examples in [configs](configs) folder.

To train, 
```bash
# e.g.
python train.py --config {config file}
```