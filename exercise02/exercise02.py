import gym
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import random
from collections import namedtuple

def flatdim(space):
    if isinstance(space, gym.spaces.Discrete):
        return int(space.n)
    elif isinstance(space, gym.spaces.Tuple):
        return int(np.prod([flatdim(s) for s in space.spaces]))
    else:
        RuntimeWarning("space not recognized")

def _flatten(space, x, y):
    if isinstance(space, gym.spaces.Discrete):
        n = flatdim(space)
        y = y * n + x
    elif isinstance(space, gym.spaces.Tuple):
        for x_part, s in zip(x, space.spaces):
            y = _flatten(s, x_part, y)
    else:
        raise NotImplementedError
    return y

def flatten(space, x):
    return _flatten(space, x, 0)

class FlattenedObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.n = flatdim(env.observation_space)
        self.wrapped_observation_space = env.observation_space
        self.observation_space = gym.spaces.Discrete(self.n)
    
    def observation(self, obs):
        return flatten(self.wrapped_observation_space, obs)

def sample_epsilon_greedy_from_q(q, epsilon, state):
    """
    given q-values sample with probability epsilon an arbitrary action and with probability 1-epsilon the maximum q-value action (ties broken arbitrarily)
    """
    ### complete code here ###


def pi(state, q):
    return np.argmax(q[state,])

def b(state, q, epsilon):
    if epsilon >= (x := np.random.rand()): # suboptimal action
        return np.random.choice(range(len(q[state,])))
    else:
        # see: https://stackoverflow.com/questions/17568612/how-to-make-numpy-argmax-return-all-occurrences-of-the-maximum
        winners = np.argwhere(q[state,] == max(q[state,])).flatten().tolist()
        if len(winners) == 1:
            return winners[0]
        else: # multiple equally good actions
            return np.random.choice(range(len(winners)))

def get_b(state, q, epsilon, action):
    winners = np.argwhere(q[state,] == max(q[state,])).flatten().tolist()

    winneramount = len(winners)
    actionamount = 1/len(q[state,])

    if action in winners:
        return (1-epsilon)/winneramount + (actionamount*epsilon)
    else:
        return actionamount*epsilon

def MCOffPolicyControl(env, epsilon=0.1, nr_episodes=5000, max_t=1000, gamma=0.99):
    """
    MC-based off-policy control using weighted importance sampling
    """
    nr_actions = env.action_space.n
    nr_states = env.observation_space.n

    q = np.zeros((nr_states, nr_actions))
    c = np.zeros((nr_states, nr_actions))

    Q = namedtuple('Q', ['state', 'action', 'reward'])
    episode_returns = []
    episode_lengths = []

    with tqdm.trange(nr_episodes, desc='Training', unit='episodes') as tepisodes:
        for e in tepisodes:
            trajectory = [] # save the trajectory using Q-tuples here
            # generate trajectory
            state = env.reset()
            for t in range(max_t):
                ### your code here ###

                action = b(state, q, epsilon) # get action from behavior policy
                observation, reward, done, info = env.step(action) # perform action

                # if reward == 0 and not done:
                #     # punish steps
                #     reward = -0.04

                # if done and state != 15:
                #     # env.render()
                #     reward = -1

                trajectory.append(Q(state, action, reward)) # save the trajectory as Q-tuples
                state = observation # update new state

                if e >= 9900:
                    env.render()

                if done:
                    break # stop sampling when terminal state is reached
            
            # compute episode reward
            discounts = [gamma ** i for i in range(len(trajectory) + 1)]
            R = sum([a * b for a, (_, _, b) in zip(discounts, trajectory)])
            episode_returns.append(R)
            episode_lengths.append(len(trajectory))

            # update q-values from trajectory
            g = 0 # running return
            w = 1 # running importance sampling ratio
            for state, action, reward_i in reversed(trajectory):
                ### your code here ###
                g = gamma*g + reward_i
                
                c[state, action] = c[state, action] + w
                q[state, action] = q[state, action] + w/c[state, action] * (g - q[state, action])

                if (a := pi(state, q)) != action:
                    break
                
                w = w*(1/get_b(state, q, epsilon, action))
                

    return np.argmax(q, 1)


def SARSA(env, epsilon=0.1, alpha=0.01, nr_episodes=50000, max_t=1000, gamma=0.99):
    """
    On-policy SARSA with epsilon-greedy policy
    """
    nr_actions = env.action_space.n
    nr_states = env.observation_space.n

    # SARSA usees an epsilon-greedy policy
    # The underlying deterministic policy is derived from the q-values
    q = np.full((nr_states, nr_actions), 0, dtype=np.float32)

    # history of episode returns
    episode_returns = [] 
    episode_lengths = []

    # iterate over episodes
    with tqdm.trange(nr_episodes, desc='Training', unit='episodes') as tepisodes:
        for e in tepisodes:
            state = env.reset()
            action = sample_epsilon_greedy_from_q(q, epsilon, state)
            rewards = []

            # Collect trajectory
            for t in range(max_t):
                next_state, reward, done, _ = env.step(action)
                rewards.append(reward)

                #### your code here ###

            discounts = [gamma ** i for i in range(len(rewards) + 1)]
            R = sum([a * b for a, b in zip(discounts, rewards)])
            episode_returns.append(R)
            episode_lengths.append(len(rewards))

            # print average return of the last 100 episodes
            if(e % 100 == 0):
                avg_return = np.mean(episode_returns[-100:])
                avg_length = np.mean(episode_lengths[-100:])
                tepisodes.set_postfix({
                'episode return': avg_return,
                'episode length': avg_length
                })
                
    return np.argmax(q, 1)

def evaluate_greedy_policy(env, policy, nr_episodes=1000, t_max=1000):
    reward_sums = []
    for t in range(nr_episodes):
        state = env.reset()
        rewards = []
        for i in range(t_max):
            action = policy[state]
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            if done:
                break
    
        reward_sums.append(np.sum(rewards))
    
    return np.mean(reward_sums)

env_frozenlake = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True)
env_blackjack = FlattenedObservationWrapper(gym.make('Blackjack-v1'))

# below are some default parameters for the control algorithms. You might want to tune them to achieve better results.

# SARSA_frozenlake_policy = SARSA(env_frozenlake, epsilon=0.051, alpha=0.1, nr_episodes=10000, max_t=1000, gamma=0.99)
# print("Mean episode reward from SARSA trained policy on FrozenLake: ", evaluate_greedy_policy(env_frozenlake, SARSA_frozenlake_policy))

# SARSA_blackjack_policy = SARSA(env_blackjack, epsilon=0.051, alpha=0.1, nr_episodes=10000, max_t=1000, gamma=0.99)
# print("Mean episode reward from SARSA trained policy on BlackJack: ", evaluate_greedy_policy(env_blackjack, SARSA_blackjack_policy))


MC_frozenlake_policy = MCOffPolicyControl(env_frozenlake, epsilon=0.051, nr_episodes=10000, max_t=1000, gamma=0.99)
print("Mean episode reward from MC trained policy on FrozenLake: ", evaluate_greedy_policy(env_frozenlake, MC_frozenlake_policy))

MC_blackjack_policy = MCOffPolicyControl(env_blackjack, epsilon=0.051, nr_episodes=10000, max_t=1000, gamma=0.99)
print("Mean episode reward from MC trained policy on BlackJack: ", evaluate_greedy_policy(env_blackjack, MC_blackjack_policy))