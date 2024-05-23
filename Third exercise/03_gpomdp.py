"""
Third exercise: so far we only implermented an environment; in this exercise we will implement a policy gradient method to solve the environment.
Original file is located at https://colab.research.google.com/github/albertometelli/rl-phd-2024/blob/main/03_gpomdp.ipynb
This exercise is inspired to the Stable Baselines3 tutorial available at [https://github.com/araffin/rl-tutorial-jnrr19](https://github.com/araffin/rl-tutorial-jnrr19).

Remember to have a look at the following links:
Gymnasium Github: [https://github.com/Farama-Foundation/Gymnasium](https://github.com/Farama-Foundation/Gymnasium)
Gymnasium Documentation: [https://gymnasium.farama.org/index.html](https://gymnasium.farama.org/index.html#)
Stable Baselines 3 Github:[https://github.com/DLR-RM/stable-baselines3](https://github.com/DLR-RM/stable-baselines3)
Stable Baseline 3 Documentation: [https://stable-baselines3.readthedocs.io/en/master/](https://stable-baselines3.readthedocs.io/en/master/)
GPOMDP: (https://github.com/albertometelli/rl-phd-2024/blob/main/gpomdp.png?raw=1)
reference: Baxter, Jonathan, and Peter L. Bartlett. "Infinite-horizon policy-gradient estimation." Journal of Artificial Intelligence Research 15 (2001): 319-350.

GPOMDP: Policy-based method for Reinforcement Learning; it uses the causality property in order to get rid of useless terms and gain
advantage on the variance of the gradient estimator.
    * Policy: we will use a Gaussian linear policy, linear in the state variables and with fixed (non-learnable) standard deviation:
    πθ(a|s)=N(a|θTs,σ2) ==> action to be played is thie following inner product = θ^T * state + noise
    The policy must be implemented with the usual predict method and some additional methods for computing the policy gradient. Specifically, 
    we will need a grad_log method to return the gradient of the logarithm of the policy (the score):
    ∇θlogπθ(a|s) = (a−θTs)s/σ^2 ==>  it returns sthe logarithm of the policy after passing a state
    
To test our GPOMDP implementation, we will use the `MountainCarContinuous-v0` environment. The links are teh following:
MountainCarContinuous Environment Decription: [https://gymnasium.farama.org/environments/classic_control/mountain_car_continuous/](https://gymnasium.farama.org/environments/classic_control/mountain_car_continuous/)
MountainCarContinuous Source Code: [https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/classic_control/continuous_mountain_car.py](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/classic_control/continuous_mountain_car.py)

The car cannot move duirectly from the valley up to the top of the hill and this is a very difficult exploration problem.
We assume to have a more powerful engine, so we can reach the goal without the need of going back and then up: we will increase the power of the car.
"""

import gymnasium as gym
import renderlab
import stable_baselines3

print(gym.__version__)
print(stable_baselines3.__version__)

def evaluate(env, policy, gamma=1., num_episodes=100): # usual function
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
A helper function to plot the learning curves.
"""

import matplotlib.pyplot as plt


def plot_results(results):
    plt.figure()

    _mean = []
    _std = []
    for m, s in results:
        _mean.append(m)
        _std.append(s)

    _mean = np.array(_mean)
    _std = np.array(_std)

    ts = np.arange(len(_mean))
    plt.plot(ts, _mean, label='G(PO)MDP')
    plt.fill_between(ts, _mean-_std, _mean+_std, alpha=.2)

    plt.xlabel('Trajectories')
    plt.ylabel('Average return')
    plt.legend(loc='lower right')

    plt.show()

"""
Complete the implementation of the methods `predict` and `grad_log`
"""

class GaussianPolicy:

    # Constructor
    def __init__(self, dim, std=0.1):
        """
        :param dim: number of state variables
        :param std: fixed standard deviation
        """

        self.std = std
        self.dim = dim # Dimension of the state variable
        self.theta = np.zeros((dim,))  # zero initializatoin for theta

    def get_theta(self):
        return self.theta

    def set_theta(self, value):
        self.theta = value

    def predict(self, obs):
        """
        :param obs: (ndarray) the state observation (dim,)
        :return: the sampled action to be played and the same observation
        """
        action = 0.

        #TODO

        return np.array([action]), obs

    def grad_log(self, obs, action):
        """
        :param obs: (ndarray) the state observation (dim,)
        :param action: (float) the action
        :return: (ndarray) the score of the policy (dim,)
        """
        mean = np.dot(obs, self.theta) # dot product
        grad_log = (action - mean) * obs / self.std ** 2
        return grad_log

"""
Now we use the Gaussian Policy to train the GPOMDP algorithm.
We provide the already implemented skeleton of the training routine that samples at every iterations $m$ trajectories from the environment.
"""

def collect_rollouts(env, policy, m, T):
    """
    Collects m rollouts by running the policy in the
        environment
    :param env: (Env object) the Gym environment
    :param policy: (Policy object) the policy
    :param gamma: (float) the discount factor
    :param m: (int) number of episodes per iterations
    :param K: (int) maximum number of iterations
    :param theta0: (ndarray) initial parameters (d,)
    :param alpha: (float) the constant learning rate
    :param T: (int) the trajectory horizon
    :return: (list of lists) one list per episode
                each containing triples (s, a, r)
    """

    ll = []
    for j in range(m):
        s, _ = env.reset()
        t = 0
        done = False
        l = []
        while t < T and not done:
            a, _ = policy.predict(s)
            s1, r, done, _, _ = env.step(a)
            l.append((s, a, r))
            s = s1
            t += 1
        ll.append(l)
    return ll

# this training function manages the whole training process
def train(env, policy, gamma, m, K, alpha, T):
    """
    Train a policy with G(PO)MDP
    :param env: (Env object) the Gym environment
    :param policy: (Policy object) the policy
    :param gamma: (float) the discount factor
    :param m: (int) number of episodes to be collected per iterations (I sample m trajectories per iteration)
    :param K: (int) maximum number of iterations
    :param alpha: (float) the constant learning rate
    :param T: (int) the trajectory horizon (number of steps of which the trajectory is composed)
    :return: list (ndarray, ndarray) the evaluations
    """

    results = []

    # Evaluate the initial policy
    res = evaluate(env, policy, gamma)
    results.append(res)

    for k in range(K): # in each iteration

        print('Iteration:', k)

        # Generate rollouts (rollout = trajectory = episode)
        rollouts = collect_rollouts(env, policy, m, T)

        # Get policy parameter
        theta = policy.get_theta()

        # Call your G(PO)MDP estimator (implemented by me)
        pg = gpomdp(rollouts, policy, gamma)

        # Update policy parameter
        theta = theta + alpha * pg

        # Set policy parameters
        policy.set_theta(theta)

        # Evaluate the updated policy
        res = evaluate(env, policy, gamma)
        results.append(res)

    return results

"""
Complete the following function `gpomdp` that computes the G(PO)MDP gradient estimator given rollout trajectories.
"""

def gpomdp_inefficient(rollouts, policy, gamma):
    """
    :param rollouts: (list of lists) generated by 'collect_rollouts'
    :param policy: (Policy object) the policy
    :param gamma: (float) the discount factor
    :return: (ndarray) the policy gradient (dim,)
    """

    grad = 0

    # very very inefficient implementation because I have 3 levels of for loops nested
    for roll in rollouts: # rollouts ia a list of lists
        H = len(roll) # roll is a list
        sum_rew = 0 # variable for accumulating the sum of rewards
        for t in range(H): # for loop for the outer summation
            sum_scores = 0
            for l in range(t + 1): # for loop for the inner summation (ranges from 0 up to T included)
                s, a, _ = roll[l]
                score = policy.grad_log(s, a)
                sum_scores += score # cumulate the score
            _, _, r = roll[t] # I just care about the reward at time t
            sum_rew += gamma ** t * r * sum_scores
            
        grad += sum_rew

    return grad / len(rollouts) # Return the mean here

# More efficient implementation: the inner summation is just the previous score one plus the new one
def gpomdp(rollouts, policy, gamma):
    """
    :param rollouts: (list of lists) generated by 'collect_rollouts'
    :param policy: (Policy object) the policy
    :param gamma: (float) the discount factor
    :return: (ndarray) the policy gradient (dim,)
    """

    grad = 0

    # A little more efficient implementation: I just use two for loops
    for roll in rollouts:
        H = len(roll)
        disc_rew = np.zeros((H, 1))
        scores = np.zeros((H, policy.dim)) # This is a matrix
        for t in range(H):
            s, a, r = roll[t]
            disc_rew[t] = gamma ** t * r
            scores[t] = policy.grad_log(s, a) # matrix of scores
            
        cum_scores = np.cumsum(scores, axis = 0) # sum all the scores
        grad += np.sum(disc_rew * cum_scores, axis = 0)

    return grad / len(rollouts)


# test the GPOMDP implementation
import gymnasium as gym
from gymnasium.envs.classic_control.continuous_mountain_car import Continuous_MountainCarEnv
from gymnasium.envs.registration import register
from typing import Optional


class SimplifiedContinuous_MountainCarEnv(Continuous_MountainCarEnv):

    def __init__(self, render_mode: Optional[str] = None, goal_velocity=0):
        super(SimplifiedContinuous_MountainCarEnv, self).__init__(render_mode, goal_velocity)

        # We make the environment a little bit simpler by increasing the power
        self.power =  0.02

# this is exactly what we did previously
register(
    id="SimplifiedMountainCarContinuous-v1",
    entry_point="__main__:SimplifiedContinuous_MountainCarEnv",
    max_episode_steps = 200,
    reward_threshold = 100,
)

import numpy as np

# Instantiate the environment
env = gym.make('SimplifiedMountainCarContinuous-v1')

env_eval = gym.make('SimplifiedMountainCarContinuous-v1', render_mode = "rgb_array") # build the environment for evaluation
env_eval = renderlab.RenderFrame(env_eval, "./output")

# Instantiate the policy
policy = GaussianPolicy(env.observation_space.shape[0], std = 0.2)

gamma = 0.999  # discount factor
m = 100        # number of trajectories per iteration
K = 100        # maximum number of iterations
alpha = 0.001  # learning rate
T = 200        # lenght of each trajectory

# Start training
results = train(env, policy, gamma, m, K, alpha, T)

"""Let us render the results."""

perf_mean, perf_std = evaluate(env, policy)

evaluate(env_eval, policy, num_episodes=1)
env_eval.play()

plot_results(results)

