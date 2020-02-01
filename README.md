# Actor Critic with PPO

For intuitive guide to the mechanics of actor-critic methods check out [accompanying comic](https://medium.com/hackernoon/intuitive-rl-intro-to-advantage-actor-critic-a2c-4ff545978752).

Notebook designed for readability and exploration rather than production. Uses a single GPU. For an industrial-strength PPO in PyTorch check out [ikostrikov's](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail). For the 'definitive' implementation of PPO, check out [OpenAI baselines](https://github.com/openai/baselines/tree/master/baselines/ppo2) (tensorflow). For outstanding resources on RL check out OpenAI's [Spinning Up](https://spinningup.openai.com/en/latest/)

The notebook reproduces results from OpenAI's [procedually-generated environments](https://openai.com/blog/procgen-benchmark/) and [corresponding paper (Cobbe 2019)](https://arxiv.org/abs/1912.01588). All hyperparameters taken directly from paper. Built from scratch unless otherwise noted to gain intuition.
