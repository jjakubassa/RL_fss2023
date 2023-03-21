import gymnasium as gym
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from matplotlib import animation
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

def unflatten(space, x):
    pass

def render_FrozenLake(env, policy, filename, max_t=1000):
    frames = []
    state = env.reset()[0]
    frames.append(env.render())
    for t in range(max_t):
        action = policy[state]
        state, _, done, _, _ = env.step(action)
        frames.append(env.render())
        if done:
            break
    
    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=1250)
    anim.save(filename, writer='imagemagick', fps=6)

def visualize_lake_policy(env, policy):
    # Create a grid to hold the arrow plots
    fig, ax = plt.subplots()
    ax.set_aspect('equal', 'box')
    ax.set_xlim(0, env.desc.shape[1])
    ax.set_ylim(0, env.desc.shape[0])
    ax.set_xticks(np.arange(0.5, env.desc.shape[1], 1))
    ax.set_yticks(np.arange(0.5, env.desc.shape[0], 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(False)
    
    # Define the arrow colors
    arrow_colors = {
        0: 'blue',   # move left
        1: 'red',    # move down
        2: 'purple', # move right
        3: 'green'   # move up
    }
    
    # Define the state mapping
    state_mapping = {}
    for s in range(env.action_space.n):
        x, y = np.unravel_index(s, env.desc.shape)
        state_mapping[s] = (x, y)
    
    # Plot the arrows and terminal states
    for s in range(env.observation_space.n):
        x, y = np.unravel_index(s, env.desc.shape)
        if env.desc[x][y] == b'H' or env.desc[x][y] == b'G':
            # Terminal state
            cell = plt.Rectangle((y, env.desc.shape[0] - x - 1), width=1, height=1, facecolor='white', edgecolor='black')
            ax.add_artist(cell)
            if env.desc[x][y] == b'H':
                # Hole
                circle = plt.Circle((y + 0.5, env.desc.shape[0] - x - 0.5), radius=0.3, color='black')
                ax.add_artist(circle)
            else:
                # Goal
                circle = plt.Circle((y + 0.5, env.desc.shape[0] - x - 0.5), radius=0.3, color='blue')
                ax.add_artist(circle)
        else:
            # Non-terminal state
            action = policy[s]
            dx, dy = {
                0: (0, -1),   # move left
                1: (1, 0),    # move down
                2: (0, 1),    # move right
                3: (-1, 0)    # move up
            }[action]
            arrow_color = arrow_colors[action]
            cell = plt.Rectangle((y, env.desc.shape[0] - x - 1), width=1, height=1, facecolor='white', edgecolor='black')
            ax.add_artist(cell)
            ax.arrow(y + 0.5, env.desc.shape[0] - x - 0.5, dy * 0.3, -dx * 0.3, head_width=0.2, head_length=0.2, color=arrow_color)
    
    return plt

def plot_convergence_curve(episode_lengths, cum_episode_returns, episode_returns, mean_episode_returns):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 6), facecolor='#292929')
    fig.patch.set_facecolor('#2B2B2B')

    # Plot 0: Episode length
    ax[0,0].set_facecolor('#2B2B2B')
    ax[0,0].plot(episode_lengths, color='#01d000')
    ax[0,0].set_xlabel('Episodes', fontsize=12, color='white')
    # ax[0,0].set_ylabel('Steps per episode', fontsize=12, color='white')
    ax[0,0].tick_params(axis='x', colors='white')
    ax[0,0].tick_params(axis='y', colors='white')
    ax[0,0].spines['bottom'].set_color('white')
    ax[0,0].spines['left'].set_color('white')
    ax[0,0].set_title('Steps per episode', fontsize=14, color='white')


    # Plot 1: Cumulative Returns
    ax[1,0].set_facecolor('#2B2B2B')
    ax[1,0].plot(cum_episode_returns, color='#30a2da')
    ax[1,0].set_xlabel('Episodes', fontsize=12, color='white')
    # ax[1,0].set_ylabel('Cumulative Episode Returns', fontsize=12, color='white')
    ax[1,0].tick_params(axis='x', colors='white')
    ax[1,0].tick_params(axis='y', colors='white')
    ax[1,0].spines['bottom'].set_color('white')
    ax[1,0].spines['left'].set_color('white')
    ax[1,0].set_title('Cumulative Episode Returns', fontsize=14, color='white')

    # Plot 2: Episode Returns
    ax[0,1].set_facecolor('#2B2B2B')
    ax[0,1].plot(episode_returns, color='#fc4f30')
    ax[0,1].set_xlabel('Episodes', fontsize=12, color='white')
    # ax[0,1].set_ylabel('Episode Returns', fontsize=12, color='white')
    ax[0,1].tick_params(axis='x', colors='white')
    ax[0,1].tick_params(axis='y', colors='white')
    ax[0,1].spines['bottom'].set_color('white')
    ax[0,1].spines['left'].set_color('white')
    ax[0,1].set_title('Episode Returns', fontsize=14, color='white')

    # Plot 3: Mean Episode Returns
    ax[1,1].set_facecolor('#2B2B2B')
    ax[1,1].plot(mean_episode_returns, color='#e5ae38')
    ax[1,1].set_xlabel('Episodes', fontsize=12, color='white')
    # ax[1,1].set_ylabel('Mean Episode Returns', fontsize=12, color='white')
    ax[1,1].tick_params(axis='x', colors='white')
    ax[1,1].tick_params(axis='y', colors='white')
    ax[1,1].spines['bottom'].set_color('white')
    ax[1,1].spines['left'].set_color('white')
    ax[1,1].set_title('Mean Episode Returns', fontsize=14, color='white')

    # fire! 
    plt.tight_layout()
    return plt

def sample_epsilon_greedy_from_q(q, epsilon, state):
    """
    return random action with probability epsilon and the best action according to q-value otherwis
    """
    if epsilon >= np.random.rand(): 
        # random suboptimal action
        return np.random.choice(range(len(q[state,])))
    else:
        # always pick first optimal action
        return np.argmax(q[state,])

def MCOffPolicyControl(env, epsilon=0.1, nr_episodes=5000, max_t=1000, gamma=0.99):
    """
    MC-based off-policy control using weighted importance sampling
    """
    nr_actions = env.action_space.n
    nr_states = env.observation_space.n

    # initialize q arbitrarily between 0 and 0.01 (during implementation this range was found to be helpful for quicker convergence)
    q = np.random.random((nr_states, nr_actions))*0.01 
    c = np.full((nr_states, nr_actions), 0.0, dtype=np.float32) # initialize cumulative sum of WIS weights with 0
    pi = np.argmax(q, axis=1) # init policy based on q

    SAR = namedtuple('SAR', ['state', 'action', 'reward'])
    episode_returns = []
    episode_lengths = []
    backtrack_percentages = []

    # custom lists for plotting epsiode return
    cum_episode_returns = [0]
    mean_episode_returns = []

    with tqdm.trange(nr_episodes, desc='Training', unit='episodes') as tepisodes:
        for e in tepisodes:
            # generate trajectory
            trajectory = []
            state = env.reset()[0]
            for t in range(max_t):
                action = sample_epsilon_greedy_from_q(q, epsilon, state) # sample action
                observation, reward, terminated, _, _ = env.step(action) # perform action
                trajectory.append(SAR(state, action, reward)) # save the trajectory as Q-tuples
                state = observation # update new state

                if terminated: # stop sampling when terminal state is reached
                    break 
            
            # compute episode reward
            discounts = [gamma ** i for i in range(len(trajectory) + 1)]
            R = sum([a * b for a, (_, _, b) in zip(discounts, trajectory)])
            episode_returns.append(R)
            episode_lengths.append(len(trajectory))

            # custom metrics for plotting
            cum_episode_returns.append(cum_episode_returns[-1]+R)
            mean_episode_returns.append(np.mean(episode_returns))

            # update q-values from trajectory
            g = 0 # running return
            w = 1 # running importance sampling ratio
            for state, action, reward_i in reversed(trajectory):
                g = gamma*g + reward_i # update return
                c[state, action] = c[state, action] + w # update cumulative sum of WIS weights
                q[state, action] = q[state, action] + w/c[state, action] * (g - q[state, action]) # update q
                pi[state] = np.argmax(q[state,]) # update target policy
                
                if pi[state] != action:
                    break
                
                w = w*(1/(1-epsilon+(epsilon/nr_actions))) # update WIS weight
            
            # print average return of the last 100 episodes
            if(e % 100 == 0):
                avg_return = np.mean(episode_returns[-100:])
                avg_length = np.mean(episode_lengths[-100:])
                avg_backtrack_percentage = np.mean(backtrack_percentages[-100:])
                tepisodes.set_postfix({
                'episode return': "{:.2f}".format(avg_return),
                'episode length': "{:3.2f}".format(avg_length),
                'backtrack': "{:.2f}%".format(avg_backtrack_percentage)
                })

    # generating the convergence curve and other plots
    plt = plot_convergence_curve(episode_lengths, cum_episode_returns[1:], episode_returns, mean_episode_returns)
    return plt, np.argmax(q, 1)

def SARSA(env, epsilon=0.1, alpha=0.01, nr_episodes=50000, max_t=1000, gamma=0.99):
    """
    On-policy SARSA with epsilon-greedy policy
    """
    nr_actions = env.action_space.n
    nr_states = env.observation_space.n

    # SARSA usees an epsilon-greedy policy
    # The underlying deterministic policy is derived from the q-values

    # initialize q arbitrarily between 0 and 1; terminal states will get set to 0 later on*
    # values are between 0 and 0.01 (during implementation this range was found to be helpful for quicker convergence)
    q = np.random.random((nr_states, nr_actions)) * 0.01

    episode_returns = []
    episode_lengths = []

    # custom lists for plotting epsiode return
    cum_episode_returns = [0]
    mean_episode_returns = []

    # iterate over episodes
    with tqdm.trange(nr_episodes, desc='Training', unit='episodes') as tepisodes:
        for e in tepisodes:
            rewards = []
            state = env.reset()[0]
            action = sample_epsilon_greedy_from_q(q, epsilon, state)

            # Collect trajectory
            for t in range(max_t):
                next_state, reward, terminated, _, _ = env.step(action)
                next_action = sample_epsilon_greedy_from_q(q, epsilon, next_state)

                if terminated:
                    # *setting q for terminal states 0; necessary due to arbitrary q initilization earlier
                    q[next_state, next_action] = 0 

                # update q
                q[state, action] = q[state, action] + alpha*(reward + gamma*q[next_state, next_action] - q[state, action])
                rewards.append(reward)
                
                # update action and state for next time step
                action = next_action
                state = next_state

                if terminated: # end of episode
                    break
                
            # compute episode reward
            discounts = [gamma ** i for i in range(len(rewards) + 1)]
            R = sum([a * b for a, b in zip(discounts, rewards)])
            episode_returns.append(R)
            episode_lengths.append(len(rewards))

            # custom metrics for plotting
            cum_episode_returns.append(cum_episode_returns[-1]+R)
            mean_episode_returns.append(np.mean(episode_returns))

            # print average return of the last 100 episodes
            if(e % 100 == 0):
                avg_return = np.mean(episode_returns[-100:])
                avg_length = np.mean(episode_lengths[-100:])
                tepisodes.set_postfix({
                'episode return': avg_return,
                'episode length': avg_length
                })
                
    # generating the convergence curve and other plots
    plt = plot_convergence_curve(episode_lengths, cum_episode_returns[1:], episode_returns, mean_episode_returns)
    return plt, np.argmax(q, axis=1)

def evaluate_greedy_policy(env, policy, nr_episodes=1000, t_max=1000):
    reward_sums = []
    for t in range(nr_episodes):
        state = env.reset()[0]
        rewards = []
        for i in range(t_max):
            action = policy[state]
            state, reward, done, _, _ = env.step(action)
            rewards.append(reward)
            if done:
                break
    
        reward_sums.append(np.sum(rewards))
    
    return np.mean(reward_sums)


env_frozenlake_small = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=False, render_mode="rgb_array")
env_frozenlake_small_slippery = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=True, render_mode="rgb_array")
env_frozenlake_medium = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=False, render_mode="rgb_array")
env_frozenlake_medium_slippery = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True, render_mode="rgb_array")
env_blackjack = FlattenedObservationWrapper(gym.make('Blackjack-v1', render_mode="rgb_array"))

epsilon = 0.1
alpha = 0.1
nr_episodes = 10000
max_t = 400
gamma = 0.9

# below are some default parameters for the control algorithms. You might want to tune them to achieve better results.

# FROZENLAKE
for env, name in {
    env_frozenlake_small: "frozenlake_small",
    env_frozenlake_small_slippery: "frozenlake_small_slippery",
    env_frozenlake_medium: "frozenlake_medium",
    env_frozenlake_medium_slippery: "frozenlake_medium_slippery",
    }.items():
    
    ## MC OFF POLICY CONTROL
    plt0, MC_policy = MCOffPolicyControl(env, epsilon=epsilon, nr_episodes=nr_episodes, max_t=max_t, gamma=gamma)
    print("Mean episode reward from MC trained policy on", name, ": ", evaluate_greedy_policy(env, MC_policy))
    plt0.savefig("MC_conv_" + name + ".png")
    plt0.close()
    render_FrozenLake(env, MC_policy, "MC_pol_" + name + ".gif", max_t=100)
    plt1 = visualize_lake_policy(env, MC_policy)
    plt1.savefig("MC_pol_" + name + ".png")
    plt1.close()


    ## SARSA
    plt2, SARSA_policy = SARSA(env, epsilon=epsilon, alpha=alpha, nr_episodes=nr_episodes, max_t=max_t, gamma=gamma)
    print("Mean episode reward from SARSA trained policy on", name, ": ", evaluate_greedy_policy(env, SARSA_policy))
    plt2.savefig("SARSA_conv_" + name + ".png")
    plt2.close()
    render_FrozenLake(env, SARSA_policy, "SARSA_pol_" + name + ".gif", max_t=100)
    visualize_lake_policy(env, SARSA_policy)
    plt3 = visualize_lake_policy(env, SARSA_policy)
    plt3.savefig("SARSA_pol_" + name + ".png")
    plt3.close()


# BLACKJACK
## MC OFF POLICY CONTROL
plt4, MC_blackjack_policy = MCOffPolicyControl(env_blackjack, epsilon=0.051, nr_episodes=10000, max_t=1000, gamma=0.99)
print("Mean episode reward from MC trained policy on BlackJack: ", evaluate_greedy_policy(env_blackjack, MC_blackjack_policy))
plt4.savefig("MC_conv_blackjack.png")
plt4.close()

## SARSA
plt5, SARSA_blackjack_policy = SARSA(env_blackjack, alpha=0.1, epsilon=0.051, nr_episodes=10000, max_t=1000, gamma=0.99)
print("Mean episode reward from SARSA trained policy on BlackJack: ", evaluate_greedy_policy(env_blackjack, SARSA_blackjack_policy))
plt5.savefig("SARSA_conv_blackjack.png")
plt5.close()