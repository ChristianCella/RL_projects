"""
Second exercise: build a Gym environment for the CartPole problem
The original file is located at: https://colab.research.google.com/github/albertometelli/rl-phd-2024/blob/main/02_gym_environment.ipynb
This notebook is inspired to the Stable Baselines3 tutorial available at [https://github.com/araffin/rl-tutorial-jnrr19](https://github.com/araffin/rl-tutorial-jnrr19).

In this notebook, we will learn how to build a customized environment with Gymnasium.
Gymnasium Github: [https://github.com/Farama-Foundation/Gymnasium](https://github.com/Farama-Foundation/Gymnasium)
Gymnasium Documentation: [https://gymnasium.farama.org/index.html](https://gymnasium.farama.org/index.html#)
Stable Baselines 3 Github:[https://github.com/DLR-RM/stable-baselines3](https://github.com/DLR-RM/stable-baselines3)
Stable Baseline 3 Documentation: [https://stable-baselines3.readthedocs.io/en/master/](https://stable-baselines3.readthedocs.io/en/master/)

Goal of the exercise:
The Minigolf environment models a simple problem in which the agent has to hit a ball on a green using a putter in order to reach the hole with 
the minimum amount of moves:
    * The green is characterized by a friction f that is selected uniformly random at the beginning of each episode in the interval [0.065, 0.196] 
    and does not change during the episode;
    * The position of the ball is represented by a unidimensional variable x that is initialized uniformly random in the interval [1,20]. The observation
    model is made of the pair s = (x_t,f);
    * The action a_t is the force applied to the putter and has to be bounded in the interval [1e-5,5]. Before being applied the action is subject to a
    Gaussian noise, so that the actual action u_t applied is given by (I have an additive noise sampled from a Gaussian):
    ut=at+ϵwhereϵ∼N(0,σ2)
    where σ=0.1. The movement of the ball is governed by the kinematic law: xt+1=xt−vtτt+12dτ2t
    where;
    * vt is the velocity computed as vt=ut*l
    * d is the deceleration computed as d = 5/7*f*g
    * τt is the time interval computed as τt = vt/d
    
    The remaining constants are the putter length  l=1  and the gravitational acceleration  g=9.81 . 
    * The episode terminates when the next state is such that the ball enters or surpasses (without entering) the hole. 
    * The reward is -1 at every step (I want to collect the samllest number of -1) and -100 if the ball surpasses the hole. 
    To check whether the ball will not reach, enter, or surpass the hole, refer to the following condition:
    * vt<vmin⟹ball does not reach the hole
    * vt>vmax⟹ball surpasses the hole
    * otherwise⟹ball enters the hole
    where:
    * vmin = sqrt(10/7*f*g*x_t)
    * vmax=√g(2h−ρ)2r+vmin2
    where h=0.1 is the hole size and ρ=0.02135 is the ball radius.

"""
 
import gymnasium as gym # abstract class for the environment
import renderlab
import stable_baselines3

print(gym.__version__)
print(stable_baselines3.__version__)

def evaluate(env, policy, gamma = 1., num_episodes = 100): # This was already implemented
    """
    Evaluate a RL agent
    :param env: (Env object) the Gym environment
    :param policy: (BasePolicy object) the policy in stable_baselines3
    :param gamma: (float) the discount factor
    :param num_episodes: (int) number of episodes to evaluate it
    :return: (float) Mean reward for the last num_episodes
    """
    all_episode_rewards = []
    for i in range(num_episodes): # iterate over the episodes
        episode_rewards = []
        done = False
        discounter = 1.
        obs, _ = env.reset()
        while not done: # iterate over the steps until termination
            action, _ = policy.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_rewards.append(reward * discounter) # compute discounted reward
            discounter *= gamma

        all_episode_rewards.append(sum(episode_rewards))

    mean_episode_reward = np.mean(all_episode_rewards)
    std_episode_reward = np.std(all_episode_rewards) / np.sqrt(num_episodes - 1)
    print("Mean reward:", mean_episode_reward,
          "Std reward:", std_episode_reward,
          "Num episodes:", num_episodes)

    return mean_episode_reward, std_episode_reward

"""
Refer to: 'Penner, A. R. "The physics of putting." Canadian Journal of Physics 80.2 (2002): 83-96'.
Complete the constructor `__init__`, methods `reset` and `step` based on the environment description provided above.
"""

import numpy as np
from gymnasium.spaces import Box

class Minigolf(gym.Env):

    def __init__(self):
        super(Minigolf, self).__init__()

        # Constants
        self.min_pos, self.max_pos = 1.0, 20.0
        self.min_action, self.max_action = 1e-5, 5.0
        self.min_friction, self.max_friction = 0.065, 0.196
        self.putter_length = 1.0
        self.hole_size = 0.10
        self.sigma_noise = 0.1
        self.ball_radius = 0.02135


        # Instance the spaces
        low = np.array([self.min_pos, self.min_friction])
        high = np.array([self.max_pos, self.max_friction])

        self.action_space = Box(low=self.min_action,
                                high=self.max_action,
                                shape=(1,),
                                dtype=np.float32)

        self.observation_space = Box(low=low,
                                     high=high,
                                     shape=(2,),
                                     dtype=np.float32)


    def step(self, action): # Method taht executes an action in the environment

        #Retrieve the state components
        x, friction = self.state

        # Clip the action within the allowed range
        action = np.clip(action, self.min_action, self.max_action)

        # Add noise to theb action
        noisy_action = action + np.random.randn() * self.sigma_noise
        
        # Compute the speed
        v = noisy_action * self.putter_length
        v = np.array(v).ravel().item() # to ne sure to get a scalar

        # Compute the speed limits
        v_min = np.sqrt(10 / 7 * friction * 9.81 * x)
        v_max = np.sqrt((2 * self.hole_size - self.ball_radius) ** 2 \
                        * (9.81 / (2 * self.ball_radius)) + v_min ** 2)

        # Compute the deceleration
        decelearation = 5 / 7 * friction * 9.81
        
        # Compute the time interval
        t = v / decelearation

        # Update the state and clip
        x = x - v * t + 0.5 * decelearation * t ** 2
        x = np.clip(x, self.min_pos, self.max_pos) # Project it back on the feasible set

        # Now I compute the reward and episode termination
        reward = 0
        done = True # Boolean controlling the reaching of the terminal state
        
        if v < v_min:
            reward = -1
            done = False # Keep going
        elif v > v_max:
            reward = -100
            done = True # Termination condition reached
        
        self.state = np.array([x, friction]).astype(np.float32)

        return self.state, reward, done, False, {}


    def reset(self, seed = None):

        # randomm generation of initial position and friction
        x, friction = np.random.uniform(low=[self.min_pos, self.min_friction],
                                        high=[self.max_pos, self.max_friction])

        self.state = np.array([x, friction]).astype(np.float32)

        return self.state, {}

"""
To be able to instance the environment with `gym.make`, we need to register the environment
"""
from gymnasium.envs.registration import register

register(
    id = "Minigolf-v1",
    entry_point = "__main__:Minigolf",
    max_episode_steps = 20, #  After 20 steps the episode terminates regardless if I reached or not  a terminal state
    reward_threshold = 0,
)

"""
Validate the environment
Stable Baselines3 provides a [helper](https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html) to check that our environment 
complies with the Gym interface.
"""

from stable_baselines3.common.env_checker import check_env

env = Minigolf()

# If the environment don't follow the interface, an error will be thrown
check_env(env, warn = True)

"""
Evaluate some simple Policies:
    1) Do-nothing policy: a policy plays the zero action: π(s)=0
    2) Max-action policy: a policy that plays the maximum available actions: π(s)=+∞
    3) Zero-mean Gaussian policy: a policy that selects the action sampled from a Gaussian policy with zero mean and variance σ^2=1: π(a|s)=N(0,σ2)
"""

class DoNothingPolicy():

    def predict(self, obs):
        return 0, obs


class MaxActionPolicy(): 

    def predict(self, obs):
        return np.inf, obs


class ZeroMeanGaussianPolicy():

    def predict(self, obs):
        return np.random.randn(), obs

env = gym.make("Minigolf-v1")

do_nothing_policy = DoNothingPolicy()
max_action_policy = MaxActionPolicy()
gauss_policy = ZeroMeanGaussianPolicy()

do_nothing_mean, do_nothing_std = evaluate(env, do_nothing_policy) # If I do not do anything, after 20 steps the episode terminates
max_action_mean, max_action_std = evaluate(env, max_action_policy) # I always surpass the hole if I play the max action
gauss_policy_mean, gauss_policy_std = evaluate(env, gauss_policy) # I do something here: I will obtain a certain reward

"""
Train PPO, DDPG, and SAC
We now train three algorithms suitable for environments with continuous actions to learn the behaviour.
* All of them are instanciated with a Multi-Layer Perceptron (MLP) policy with a single hidden layer of 32 neurons.

See the followijng links for more details:
[Proximal Policy Optimization](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html), 
[Deep Deterministic Policy Gradient](https://stable-baselines3.readthedocs.io/en/master/modules/ddpg.html)
[Soft Actor Critic](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html).
"""

from stable_baselines3 import PPO, DDPG, SAC

# Separate evaluation env
eval_env = gym.make('Minigolf-v1')

ppo = PPO("MlpPolicy", env, verbose = 1, policy_kwargs = dict(net_arch=[32]))
ddpg = DDPG("MlpPolicy", env, verbose = 1, policy_kwargs = dict(net_arch=[32]))
sac = SAC("MlpPolicy", env, verbose = 1, policy_kwargs = dict(net_arch=[32]))

print('PPO')
ppo.learn(total_timesteps=50000, log_interval=4, progress_bar=True)

print('DDPG')
ddpg.learn(total_timesteps=50000, log_interval=1024, progress_bar=True)

print('SAC')
sac.learn(total_timesteps=50000, log_interval=2048, progress_bar=True)

"""Let us now evaluate the results of the training."""

ppo_mean, ppo_std = evaluate(eval_env, ppo) # PPO is the worst in this case
ddpg_mean, ddpg_std = evaluate(eval_env, ddpg) # In less than one action I reach the goal
sac_mean, sac_std = evaluate(eval_env, sac) # It takes almost 2 actions to reach the goal