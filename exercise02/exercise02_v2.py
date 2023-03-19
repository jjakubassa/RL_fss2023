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

    # Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis("off")

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(
        plt.gcf(), animate, frames=len(frames), interval=1250
    )
    anim.save(filename, writer="imagemagick", fps=6)


def sample_epsilon_greedy_from_q(q, epsilon, state):
    """
    return random action with probability epsilon and the best action according to q-value otherwise
    """
    nr_actions = q.shape[1]
    if np.random.random() < epsilon:
        a = np.random.choice(nr_actions)
        return a
    else:
        return np.argmax(q[state,])


def MCOffPolicyControl(env, epsilon=0.1, nr_episodes=5000, max_t=1000, gamma=0.99):
    """
    MC-based off-policy control using weighted importance sampling
    """
    nr_actions = env.action_space.n
    nr_states = env.observation_space.n

    q = np.full((nr_states, nr_actions), 1.0, dtype=np.float32)
    c = np.full((nr_states, nr_actions), 0.0, dtype=np.float32)

    SAR = namedtuple("SAR", ["state", "action", "reward"])
    episode_returns = []
    episode_lengths = []
    backtrack_percentages = []

    with tqdm.trange(nr_episodes, desc="Training", unit="episodes") as tepisodes:
        for e in tepisodes:
            trajectory = []
            state = env.reset(seed = 42)[0]
            # env.state = np.random.choice(nr_states)

            # generate trajectory
            for t in range(max_t):
                # Your code here
                action = sample_epsilon_greedy_from_q(q, epsilon, state)
                observation, reward, done, truncated, info = env.step(action)
                trajectory.append(SAR(state, action, reward))
                state = observation
                if state == 63 or reward != 0:
                    pass

                if done:
                    break

            # compute episode reward
            discounts = [gamma**i for i in range(len(trajectory) + 1)]  # why +1?
            R = sum([a * b for a, (_, _, b) in zip(discounts, trajectory)])
            episode_returns.append(R)
            episode_lengths.append(len(trajectory))

            # update q-values from trajectory
            # your code here #
            g = 0
            w = 1
            for s, a, r in reversed(trajectory):
                g = gamma * g + r
                c[s, a] = c[s, a] + w
                q[s, a] = q[s, a] + w / c[s, a] * (g - q[s, a])
                if a != np.argmax(q[s,]):  # a != pi
                    break
                w *= 1 / (1 - epsilon + (epsilon / nr_actions))

            # print average return of the last 100 episodes
            if e % 100 == 0:
                avg_return = np.mean(episode_returns[-100:])
                avg_length = np.mean(episode_lengths[-100:])
                avg_backtrack_percentage = np.mean(backtrack_percentages[-100:])
                tepisodes.set_postfix(
                    {
                        "episode return": "{:.2f}".format(avg_return),
                        "episode length": "{:3.2f}".format(avg_length),
                        "backtrack": "{:.2f}%".format(avg_backtrack_percentage),
                    }
                )

    # Plot training
    n = 100  # average over n episodes
    avg_returns = np.mean(np.array(episode_returns).reshape(-1, n), 1)
    plt.plot(avg_returns, scalex=n)
    plt.savefig(f"MCOffPolicyControl-{env.spec.id}")

    return np.argmax(q, 1)


def SARSA(env, epsilon=0.1, alpha=0.01, nr_episodes=50000, max_t=1000, gamma=0.99):
    """
    On-policy SARSA with epsilon-greedy policy
    """
    nr_actions = env.action_space.n
    nr_states = env.observation_space.n

    # SARSA usees an epsilon-greedy policy
    # The underlying deterministic policy is derived from the q-values
    q = np.full((nr_states, nr_actions), 10, dtype=np.float32)
    # count how often a state action pair is sampled
    c = np.zeros_like(q, dtype=np.int64)

    # history of episode returns
    episode_returns = []
    episode_lengths = []

    # iterate over episodes
    with tqdm.trange(nr_episodes, desc="Training", unit="episodes") as tepisodes:
        for e in tepisodes:
            state = env.reset()[0]
            action = sample_epsilon_greedy_from_q(q, epsilon, state)
            c[state, action] += 1
            rewards = []

            # Collect trajectory
            for t in range(max_t):
                # print(state, action)

                next_state, reward, done, _, _ = env.step(action)
                rewards.append(reward)

                # your code here #

            discounts = [gamma**i for i in range(len(rewards) + 1)]
            R = sum([a * b for a, b in zip(discounts, rewards)])
            episode_returns.append(R)
            episode_lengths.append(len(rewards))

            # print average return of the last 100 episodes
            if e % 100 == 0:
                avg_return = np.mean(episode_returns[-100:])
                avg_length = np.mean(episode_lengths[-100:])
                tepisodes.set_postfix(
                    {"episode return": avg_return, "episode length": avg_length}
                )

    return np.argmax(q, 1)


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


env_frozenlake_small = gym.make(
    "FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode="rgb_array"
)
env_frozenlake_small_slippery = gym.make(
    "FrozenLake-v1", map_name="4x4", is_slippery=True, render_mode="rgb_array"
)
env_frozenlake_medium = gym.make(
    "FrozenLake-v1", map_name="8x8", is_slippery=False, render_mode="rgb_array"
)
env_frozenlake_medium_slippery = gym.make(
    "FrozenLake-v1", map_name="8x8", is_slippery=True, render_mode="rgb_array"
)
env_blackjack = FlattenedObservationWrapper(
    gym.make("Blackjack-v1", render_mode="rgb_array")
)

epsilon = 0.3
alpha = 0.1
nr_episodes = 1_000_000
max_t = 400  # default 400
gamma = 0.9

# below are some default parameters for the control algorithms. You might want to tune them to achieve better results.
for env, name in {
    # env_frozenlake_small: "frozenlake_small",
    # env_frozenlake_small_slippery: "frozenlake_small_slippery",
    env_frozenlake_medium: "frozenlake_medium",
    # env_frozenlake_medium_slippery: "frozenlake_medium_slippery",
}.items():
    MC_policy = MCOffPolicyControl(
        env, epsilon=epsilon, nr_episodes=nr_episodes, max_t=max_t, gamma=gamma
    )
    print(
        "Mean episode reward from MC trained policy on",
        name,
        ": ",
        evaluate_greedy_policy(env, MC_policy),
    )
    render_FrozenLake(env, MC_policy, name + "_MC.gif", max_t=100)

    # SARSA_policy = SARSA(env, epsilon=epsilon, alpha=alpha, nr_episodes=nr_episodes, max_t=max_t, gamma=gamma)
    # print("Mean episode reward from SARSA trained policy on", name, ": ", evaluate_greedy_policy(env, SARSA_policy))
    # render_FrozenLake(env, SARSA_policy, name + "_SARSA.gif", max_t=max_t)


# MC_blackjack_policy = MCOffPolicyControl(env_blackjack, epsilon=0.051, nr_episodes=10000, max_t=1000, gamma=0.99)
# print("Mean episode reward from MC trained policy on BlackJack: ", evaluate_greedy_policy(env_blackjack, MC_blackjack_policy))

# SARSA_blackjack_policy = SARSA(env_blackjack, alpha=0.1, epsilon=0.051, nr_episodes=10000, max_t=1000, gamma=0.99)
# print("Mean episode reward from SARSA trained policy on BlackJack: ", evaluate_greedy_policy(env_blackjack, SARSA_blackjack_policy))
