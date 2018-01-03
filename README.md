# Simple Advantage Actor Critic (A2C)

The notebooks in this repo build an A2C from scratch in PyTorch, starting with a Monte Carlo version that takes four floats as input (Cartpole) and gradually increasing complexity until the final model, an n-step A2C with multiple actors which takes in raw pixels. These models are simple in an effort to facilitate understanding. For a more production-strength A2C check out [this model](https://github.com/rgilman33/baselines-A2C) converted from OpenAI baselines.

Notebooks:
1) Monte Carlo A2C
2) Adding N-Step
3) A simplified version of 2a used for teaching purposes. Compliment to comic (show link).
4) Adding in multiple actors
5) Allowing model to take in a stack of "frames" rather that single frame. This in preparation for next step when we add in stack of frames from raw pixels.
6) Transitioning to raw pixel input. Changing FC NN to CNN.

For a deeper dive in deep RL, these are my favorite resources:

[Reinforcement Learning: An Introduction. Barto & Sutton](http://ufal.mff.cuni.cz/~straka/courses/npfl114/2016/sutton-bookdraft2016sep.pdf)

[David Silver's course](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)

[Denny Britz' RL repo](https://github.com/dennybritz/reinforcement-learning)
