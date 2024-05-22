"""
This first exercise is a comparison vbetween 5 different RL algorithms.
A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track:

* The State space is composed by 4 values: the position of the cart (0), the velocity of the cart (1), 
    the angle of the pole (2) and the angular velocity of the pole (3). 
    The ranges are the following:
    - position: [-4.8, 4.8], but an episode is terminated if the cart position is outside the range [-2.4, 2.4]
    - velocity: [-inf, inf]
    - angle of the pole: [-24 deg, 24 deg], but an episode is terminated if the pole angle is outside the range [-12 deg, 12 deg]
    - angular velocity of the pole: [-inf, inf]
    The pendulum starts upright, and the goal is to prevent it from falling over. 
    
* Action space: The system is controlled by applying a force of +1 (move right) or 0 (move left) to the cart. 

* A reward of +1 is provided for every timestep that the pole remains upright. 

* All observations are assigned a uniform random value in [-0.05, 0.05] at initialization, which is the same as the classic control environment.

* There are 3 different termination conditions:
    - The cart position is more than 2.4 units from the center
    - The pole angle is more than 12 degrees from the vertical
    - The episode length is greater than 500

Links:
A list of available environments for Gym can be found at https://gym.openai.com/envs/#classic_control.
Cartpole Environment Decription: https://gymnasium.farama.org/environments/classic_control/cart_pole/
Cartpole Source Code: https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/classic_control/cartpole.py
"""

verbose = True

import gymnasium as gym # This library provides a collection of environments for Reinforcement Learning
import renderlab # framework for creating high-quality visualizations
import stable_baselines3

if verbose == True:
    print(f"The version of gymnasium is {gym.__version__}")
    print(f"The version of stable_baseline3 is {stable_baselines3.__version__}")

"""
Initialize the environment.
The Gymnasium environment provides 4 main methods:
- reset: reset the environment to the initial state
- step: execute an action in the environment
    - observation: an environment-specific object representation of your observation of the environment after the action is executed. It corresponds to the 
      observation of the next state  St+1∼p(⋅|St,At);
    - reward: immediate reward Rt+1=r(St,At) obtained by executing action At in state St
    - terminated: whether the reached next state  St+1  is a terminal state
    - truncated: whether the trajectory has reached the maximum number of steps
    - info: a dictionary containing additional information
- render: render the environment  
- seed: set the seed for this env's random number generator  
"""

env = gym.make('CartPole-v1')
env_eval = gym.make('CartPole-v1', render_mode = "rgb_array")
env_eval = renderlab.RenderFrame(env_eval, "./output")

# When I want to interact with the environment it has to be reset. The method '.reset' outputs the initial stater
state, _ = env.reset() # resets the environment in the initial state: I force the environment to start from the initial state
if verbose == True:
    print("Initial state: ", state)

# I sample a random action among the action space (once for each iteration)
for _ in range(30):
    action = env.action_space.sample() # sample a random action

    state, reward, terminated, truncated, _ = env.step(action)  # execute the action in the environment (I send the action to the environment). The '_' is a dictionary to have further infomation
    if verbose == True:
        print("State:", state,
            "Action:", action,
            "Reward:", reward,
            "Terminated:", terminated, # If the environment has reached the terminal state (terminated = 1 ==> The cart poile has taken a position outside the angle range)
            "Truncated:", truncated)

env.close()

"""
We are playing a random action: after some iterations, the pole surely goes outside the prescribed range: terminated will be soon True
    * Observation_space: this attribute provides the format of valid observations S . It is of datatype Space provided by Gymnasium. 
        For example, if the observation space is of type Box and the shape of the object is (4,), this denotes a valid observation will be an array of 4 numbers (R^4).
    * Action_space: this attribute provides the format of valid actions  A . It is of datatype Space provided by Gymnasium. 
        For example, if the action space is of type Discrete and gives the value Discrete(2), this means there are two valid discrete actions: 0 and 1.
"""

if verbose == True:
    print(f"The observation space is: {env.observation_space}") 
    print(f"The action state-space is: {env.action_space}") 
    print(env.observation_space.high)
    print(env.observation_space.low)

"""
Spaces types available in Gymnasium:
    - Box: an  n -dimensional compact space (i.e., a compact subset of  Rn ). The bounds of the space are contained in the high and low attributes.
    - Discrete: a discrete space made of  n  elements, where  {0,1,…,n−1}  are the possible values.
Other Spaces types can be used: Dict, Tuple, MultiBinary, MultiDiscrete.
"""

import numpy as np
from gymnasium.spaces import Box, Discrete

# Build a state-space that is a subset of R^3
observation_space = Box(low = -1.0, high = 2.0, shape = (3,), dtype = np.float32)
if verbose == True:
    print(observation_space.sample()) # This method provides points that are uniformly sampled from the observation space

# Similarly, I build now a discerte state space with 4 variables
observation_space = Discrete(4)
if verbose == True: 
    print(observation_space.sample())

"""
'policy' (according to abseline3) = object that exposes a method providing a pair as output
The following two policies are hard-coded, not obtained with Reinforcement Learning:
    - UniformPolicy: it selects an action randomly (π(a|s) = Uni({0,1}))
    - ReactivePolicy: it selects the action 0 if the pole angle is smaller than zero, otherwise it selects the action 1 (π(a|s) = 0 if s_3 <= 0, 1 otherwise)
"""

class UniformPolicy:
    def predict(self, obs):
        return np.random.randint(0, 2), obs  # return the observation to comply with stable-baselines3


class ReactivePolicy:
    def predict(self, obs):
        if obs[2] <= 0: # If the pole angle (third element) is smaller than zero
            return 0, obs
        else:
            return 1, obs

"""
Creation of a function to evaluate the agent's performance: the following function is just created to run some episodes with an environment executing
a certain policy and using a discount factor.
"""
def evaluate(env, policy, gamma = 1., num_episodes = 100):
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
        while not done: # iterate over the steps until you reach a termination and play the same action!
            action, _ = policy.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action) # obs is the observation of the next state
            done = terminated or truncated # 'or' returns True if at least one of the conditions is True
            episode_rewards.append(reward * discounter) # compute discounted reward
            discounter *= gamma # update the discount factor

        # After each episode, append the sum of the rewards
        all_episode_rewards.append(sum(episode_rewards))

    # After all the episodes, compute the mean and the standard deviation of the rewards
    mean_episode_reward = np.mean(all_episode_rewards)
    std_episode_reward = np.std(all_episode_rewards) / np.sqrt(num_episodes - 1)
    
    if verbose == True:
        print("Mean reward:", mean_episode_reward, # collected over the episodes: if the result is 22.23 ==> I can keep on average the pole in the correct range for 22.23 steps
            "Std reward:", std_episode_reward,
            "Num episodes:", num_episodes)

    return mean_episode_reward, std_episode_reward

# test the uniform policy
uniform_policy = UniformPolicy() # Create an instance of the class UniformPolicy
uniform_policy_mean, uniform_policy_std = evaluate(env, uniform_policy)
_, _ = evaluate(env_eval, uniform_policy, num_episodes = 1)
env_eval.play()

# Test the reactive policy (I do a little better than the random policy: I keep it in the correct range for almost 43 steps, but the behaviour is unstable)
reactive_policy = ReactivePolicy()
reactive_policy_mean, reactive_policy_std = evaluate(env, reactive_policy)
_, _ = evaluate(env_eval, reactive_policy, num_episodes = 1)
env_eval.play()

"""
To sum up: the first 2 hard-coded policies are not good at all. We will now try to train a RL algorithm like PPO (Proximal Policy Optimization) on the environment.
The link for PPo is the following: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html.
* We will consider MlpPolixcy (Multi-Layer Perceptron Policy) for the agent, since the state of the agent is a feature vector, not images.
* The action to be used will be automatically inferred from the environment action space.
* We will actually consider 2 different scenarios for the discrete settings of the problem:
    - A simple linear policy (linear in the state representation). Since the action-state is finite, we consider 
    a Boltzmann (or softmax) policy (prob. of playing an action = e ^ (theta^T * s) / (1 + e ^ (theta^T * s))
    - A 2-hidden layer neural network: I still have a similar policy (Boltzmann form), but no more 
    linear combinations (pi(a|s) = e ^ (NN_theta(s)) / (1 + e ^ (NN_theta(s))).
* If The actions were continuous, the previous policies would become Gaussians: 
    for the first ==> N(theta^T * s. sigma ^ 2);    
    for the second ==> N(NN_theta(s), sigma ^ 2).
"""

from stable_baselines3 import PPO

# Instantiate the algorithm with 32x32 NN approximator for both actor and critic (MLP policy = Multi-Layer Perceptron)
ppo_mlp = PPO("MlpPolicy", env, verbose=1,
                learning_rate=0.01,
                policy_kwargs=dict(net_arch = [dict(pi=[32, 32], vf=[32, 32])])) # pi = actor; vf = value function = critic; the neurons are 32; the activation function is hyperbolic tangent ,not ReLU

# Instantiate the algorithm with linear approximator for both actor and critic
ppo_linear = PPO("MlpPolicy", env, verbose=1,
                   learning_rate=0.01,
                   policy_kwargs=dict(net_arch = [dict(pi=[], vf=[])])) # Linear function = Neural network with some hidden layers

if verbose == True:
    print(f"The policy for the NN with 32 neurons is {ppo_mlp.policy}")
    print(f"The policy for the linear function approximates as a 2-hidden-layer NN is {ppo_linear.policy}")

# Train the agent for 50000 steps (learn is a method)
ppo_mlp.learn(total_timesteps = 50000, log_interval = 4, progress_bar = True)
ppo_linear.learn(total_timesteps = 50000, log_interval = 4, progress_bar = True)

# Evaluate the trained models
ppo_mlp_mean, ppo_mlp_std = evaluate(env, ppo_mlp)
_, _ = evaluate(env_eval, ppo_mlp, num_episodes = 1)
env_eval.play()

ppo_linear_mean, ppo_linear_std = evaluate(env, ppo_linear)
_, _ = evaluate(env_eval, ppo_linear, num_episodes = 1)
env_eval.play()

# Let us have a look at the weights learned by PPO with the linear policy. Since actions are discrete, the policy model is softmax: πθ(a|s)∝exp(sTθ(a)+b(a))
if verbose == True:
    print(f"The weights are{ppo_linear.policy.action_net.weight}") # The weights are 4 since each of the 2 neurons is connecteed to the other
    print(f"The bias is{ppo_linear.policy.action_net.bias}")

"""
Let us now try [DQN](https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html) with an MlpPolicy as well.
"""
from stable_baselines3 import DQN
from torch import nn

# Instantiate the algorithm with 32x32 NN approximator
dqn_mlp = DQN("MlpPolicy", env, verbose=1,
                learning_starts = 3000, # N of steps after which to start learning; remember: DQN needs a replay buffer, that needs to be quite filled before you can start learning
                policy_kwargs=dict(net_arch = [32, 32], activation_fn=nn.Tanh))

if verbose == True:
    print(f"The policy for DQN is {dqn_mlp.policy}")

# Train the agent for 50000 steps
dqn_mlp.learn(total_timesteps=50000, log_interval=100, progress_bar=True)

# Evaluate the trained models
dqn_mlp_mean, dqn_mlp_std = evaluate(env, dqn_mlp)

_, _ = evaluate(env_eval, dqn_mlp, num_episodes=1)
env_eval.play()

"""
Final results
"""

import matplotlib.pyplot as plt

#Plot the results
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])

algs = ['Random', 'Reactive', 'PPO MLP', 'PPO Linear', 'DQN'] # x axis
means = [uniform_policy_mean, reactive_policy_mean, ppo_mlp_mean, ppo_linear_mean, dqn_mlp_mean] # vector of mean values
errors = [uniform_policy_std, reactive_policy_std, ppo_mlp_std, ppo_linear_std, dqn_mlp_std] # vector of standard deviations

ax.bar(algs, means, yerr = errors, align = 'center', alpha = 0.5, ecolor = 'black', capsize = 10)
plt.show()