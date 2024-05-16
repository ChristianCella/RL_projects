import gymnasium as gym
import renderlab
import stable_baselines3

print(f"The version of gymnasium is {gym.__version__}")
print(f"The version of stable_baseline3 is {stable_baselines3.__version__}")

"""## Initializing Environments

Initializing environments in Gym and is done as follows. We can find a list of available environment [here](https://gym.openai.com/envs/#classic_control).
"""

env = gym.make('CartPole-v1')

env_eval = gym.make('CartPole-v1', render_mode = "rgb_array")
env_eval = renderlab.RenderFrame(env_eval, "./output")

# When I want to interact with the environment I have to reset it. the method '.reset' outputs the initial stater
state, _ = env.reset() # resets the environment in the initial state: I force the environment to start from the initial state
print("Initial state: ", state)

# I sample a random action among the action space (once for each iteration)
for _ in range(30):
    action = env.action_space.sample() # sample a random action

    state, reward, terminated, truncated, _ = env.step(action)  # execute the action in the environment (I send the action to the environment). The '_' is a dictgionary to have further infomation
    print("State:", state,
          "Action:", action,
          "Reward:", reward,
          "Terminated:", terminated, # If the environment has reached the terminal state (terminated = 1 ==> The cart poile has taken a position outside the angle range)
          "Truncated:", truncated)

env.close()

# We are playing a random action: after some iterations, the pole surely goes outside the prescribed range: terminated will be soon True


print(env.observation_space) # The observation space is an object of class 'Box' characterized by two vectors; it's used for modelling continuous spaces (For the cartpole, the state-space is a subset of R^4)

print(env.action_space) # Discrete(2) ==> the action space is made of two discrete actions (max force aplied to the left or to the right)

print(env.observation_space.high)

print(env.observation_space.low)



import numpy as np
from gymnasium.spaces import Box, Discrete

# Build a state-space that is a subset of R^3
observation_space = Box(low=-1.0, high=2.0, shape=(3,), dtype=np.float32)
print(observation_space.sample()) # This method provides points that are uniformly sampled from the observation space

# Similarly, I build now a discerte state space with 4 variables
observation_space = Discrete(4)
print(observation_space.sample())

# These two policies are hard-coded, not obtained with Reinforcement Learning
# policy (according to abseline3) = object that exposes a method providing a pair as output
class UniformPolicy:

    def predict(self, obs):
        return np.random.randint(0, 2), obs  # return the observation to comply with stable-baselines3


class ReactivePolicy:

    def predict(self, obs):
        if obs[2] <= 0: # If the pole angle (third element) is smaller than zero
            return 0, obs
        else:
            return 1, obs

"""Let us create a function to evaluate the agent's performance."""

# Function implemented just to run some episodes with an environment executing a certain policy and using a discount factor

def evaluate(env, policy, gamma=1., num_episodes=100):
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
        while not done: # iterate over the steps until termination and play the same action!
            action, _ = policy.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_rewards.append(reward * discounter) # compute discounted reward
            discounter *= gamma

        all_episode_rewards.append(sum(episode_rewards))

    mean_episode_reward = np.mean(all_episode_rewards)
    std_episode_reward = np.std(all_episode_rewards) / np.sqrt(num_episodes - 1)
    print("Mean reward:", mean_episode_reward, # collected over the episodes: if the result is 22.23 ==> I can keep on average the pole in the correct range for 22.23 steps
          "Std reward:", std_episode_reward,
          "Num episodes:", num_episodes)

    return mean_episode_reward, std_episode_reward

"""Let us test the uniform policy."""

uniform_policy = UniformPolicy()

uniform_policy_mean, uniform_policy_std = evaluate(env, uniform_policy)

_, _ = evaluate(env_eval, uniform_policy, num_episodes=1)
env_eval.play()

"""Let us test the reactive policy."""

reactive_policy = ReactivePolicy() # I do a little better (I keep it in the correct range for almost 43 steps, but the behaviour is unstable)

reactive_policy_mean, reactive_policy_std = evaluate(env, reactive_policy)

_, _ = evaluate(env_eval, reactive_policy, num_episodes=1)
env_eval.play()


# the previous hand-coded pèolicies were not so good. We'll now try to train a RL algorithm like PPO on the environment (Proximal Policy Optimization)

from stable_baselines3 import PPO

# We will consider two scenarios (we are in a disceret case): a simple linear policy (linear in the state representation). Since the action-state is finite, we consider a Boltzmann (or softmax) policy (prob. of playing an action = e ^ (theta^T * s) / (1 + e ^ (theta^T * s))

# In the sweconbd scenario we replace a linear function with a 2-hidden layer neural network: I still have a similar policy (Boltzmann form), but no more linear combinations. (pi(a|s) = e ^ (NN_theta(s)) / (1 + e ^ (NN_theta(s)))

# If I had continuous actions, the previous policies would become Gaussians: for the first ==> N(theta^T * s. sigma ^ 2); for the second ==> N(NN_theta(s), sigma ^ 2)

# Instantiate the algorithm with 32x32 NN approximator for both actor and critic (MLP policy = Multi-Layer Perceptron)
ppo_mlp = PPO("MlpPolicy", env, verbose=1,
                learning_rate=0.01,
                policy_kwargs=dict(net_arch = [dict(pi=[32, 32], vf=[32, 32])])) # pi = actor; vf = value function = critic; the neurons are 32; the activation function is hyperbolic tangent ,not ReLU

print(ppo_mlp.policy)

# Instantiate the algorithm with linear approximator for both actor and critic
ppo_linear = PPO("MlpPolicy", env, verbose=1,
                   learning_rate=0.01,
                   policy_kwargs=dict(net_arch = [dict(pi=[], vf=[])])) # Linear function = Neural network with some hidden layers

print(ppo_linear.policy)

"""Let us now train the algorithms. In order to keep track of the performance during learning, we can log the evaluations."""

# Train the agent for 50000 steps (learn is a method)
ppo_mlp.learn(total_timesteps=50000, log_interval=4, progress_bar=True)

ppo_linear.learn(total_timesteps=50000, log_interval=4, progress_bar=True)

# Evaluate the trained models
ppo_mlp_mean, ppo_mlp_std = evaluate(env, ppo_mlp)

_, _ = evaluate(env_eval, ppo_mlp, num_episodes=1)
env_eval.play()


ppo_linear_mean, ppo_linear_std = evaluate(env, ppo_linear)

_, _ = evaluate(env_eval, ppo_linear, num_episodes=1)
env_eval.play()


print(ppo_linear.policy.action_net.weight)
print(ppo_linear.policy.action_net.bias)

"""## DQN Training

Let us now try [DQN](https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html) with an MlpPolicy as well.
"""

# We now try to use DQN

from stable_baselines3 import DQN
from torch import nn


# Instantiate the algorithm with 32x32 NN approximator
dqn_mlp = DQN("MlpPolicy", env, verbose=1,
                learning_starts=3000, # N of steps after which to start learning; remember: DQN needs a replay buffer, that needs to be quite filled before you can start learning
                policy_kwargs=dict(net_arch = [32, 32], activation_fn=nn.Tanh))

print(dqn_mlp.policy)

# Train the agent for 50000 steps
dqn_mlp.learn(total_timesteps=50000, log_interval=100, progress_bar=True)

# Evaluate the trained models
dqn_mlp_mean, dqn_mlp_std = evaluate(env, dqn_mlp)

_, _ = evaluate(env_eval, dqn_mlp, num_episodes=1)
env_eval.play()

"""Let us now plot the final results."""

import matplotlib.pyplot as plt

#Plot the results
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])

algs = ['Random', 'Reactive', 'PPO MLP', 'PPO Linear', 'DQN']
means = [uniform_policy_mean, reactive_policy_mean, ppo_mlp_mean, ppo_linear_mean, dqn_mlp_mean]
errors = [uniform_policy_std, reactive_policy_std, ppo_mlp_std, ppo_linear_std, dqn_mlp_std]

ax.bar(algs, means, yerr=errors, align='center', alpha=0.5, ecolor='black', capsize=10)
plt.show()