### Note: Not Finished.

Implementation of multi-agent deep deterministic policy gradients. 

It's been tested with the simple tag environment in the multiagent-particle-envs repo released by OpenAI, however that version does not have bounds on the environment and has not implemented a Done callback which means that each episode goes to 1000 steps even if the agents have all gone out of bound - which keeps happening and (in my opinion) slows down training. I have put in that done callback function (in the simple tag envt only - though doing it for others should be pretty easy). Please install [my fork](https://github.com/agakshat/multiagent-particle-envs.git) of the multiagent-particle-envs repository to use this repository properly. 
Main Requirements:
1. Tensorflow
2. Keras
3. agakshat/multiagent-particle-envs
4. numpy

How to use:

1. git clone this repo
2. Make sure you have the `multiagent-particle-envs` repo is installed, which means that `import make_env` in Python 3 should be working.
3. Go into the maddpg directory here and run `python3 multiagent.py`. Should run straight out of the box.

Code Breakdown:
1. `training-code.py` is the entry code which takes in user arguments for learning rates, episode length, discount factor etc, creates the actor and critic networks for each agent and calls the training function.
2. `Train.py` implements the actual MADDPG algorithm
3. `actorcriticv2.py` defines the Actor and Critic network classes
4. `ReplayMemory.py` defines the Replay Memory class
5. `ExplorationNoise.py` defines the Ornstein-Uhlenbeck Action Noise that has been used for exploration. I'm not sure if this is the right noise generation process that should be used.


*To-Do*
1. Instead of having a different policy for each agent, have one policy per team for the `simple_tag` environment, might be easier to learn. If anyone does this, please let me know of the results you got!
2. Change the noise process from Ornstein-Uhlenbeck to something like epsilon-greedy, or something more suitable to this domain (since the OU Noise is well-suited for continuous control problems like CartPole, and not this). Again, if you do this, please let me know of the results!
