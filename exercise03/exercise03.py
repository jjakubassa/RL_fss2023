from os import truncate
from typing import Any
import torch
from torch import nn

import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import tqdm
import random
from collections import namedtuple, deque
import matplotlib.pyplot as plt

from hyperopt import fmin, tpe, hp
import datetime
from copy import deepcopy


def gym_video(policy, env, filename, nr_steps=1000):
    """
    Writes a video of policy acting in the environment.
    """
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=OUTPUT_PATH,
        name_prefix=filename,
        disable_logger=True,
    )

    for _ in range(3):
        state = env.reset()[0]
        done = False
        for t in range(nr_steps):
            action = policy(state)
            next_state, reward, done, _, _ = env.step(action)
            state = next_state
            if done:
                break
    env.close()


def evaluate_greedy_policy(env, policy, nr_episodes=1000, t_max=1000):
    reward_sums = []
    for t in range(nr_episodes):
        state = env.reset()[0]
        rewards = []
        for i in range(t_max):
            action = policy(state)
            state, reward, done, truncated, _ = env.step(action)
            rewards.append(reward)
            if done or truncated:
                break

        reward_sums.append(np.sum(rewards))
    return np.mean(reward_sums)


def plot_training(e_returns, env, algo_name: str, n: int = 1000):
    # calculate average over every n episodes
    avg_returns = np.mean(np.array(e_returns).reshape(-1, n), 1)

    # create plot
    n_eps = n * np.arange(len(avg_returns))
    fig, ax = plt.subplots()
    ax.plot(
        n_eps, avg_returns, label="avg epsilon greedy returns", linewidth=2, alpha=0.5
    )
    ax.set_ylabel("episode return")
    ax.set_xlabel("episode e")
    ax.legend()

    # save plot
    plt.savefig(
        f"{OUTPUT_PATH}/{NOW.strftime('%Y-%m-%d_%H-%M-%S')}-{algo_name}-{env.spec.id}"
    )


class QNet(nn.Module):
    def __init__(
        self, state_size: int, action_size: int, hidden_size: int, n_layers: int = 2
    ):
        super(QNet, self).__init__()
        # your code here #
        # use torch.nn.init.xavier_uniform to initialize the weights
        # super().__init__()

        self.state_size: int = state_size
        self.action_size: int = action_size
        self.flatten = nn.Flatten()

        layers = []
        in_features = state_size
        for i in range(n_layers):
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            in_features = hidden_size

        layers.append(nn.Linear(in_features, action_size))
        self.linear_relu_stack = nn.Sequential(*layers)

        # Initialize the weights of each linear layer with random values
        for layer in self.linear_relu_stack:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)

    def forward(self, state) -> tuple[Any, int]:
        """return q-values and the highest q-value action"""
        # your code here #
        qvals = self.linear_relu_stack(torch.from_numpy(state))
        action = torch.argmax(qvals.data).item()
        assert type(action) == int
        return qvals, action

    def act_epsilon_greedy(self, state, epsilon) -> int:
        """return with probability epsilon a random action and
        with probability 1-epsilon the greedy action"""
        # your code here #
        if np.random.random() < epsilon:
            action = np.random.choice(self.action_size)
        else:
            action = self.act_greedy(state)
        return action

    def act_greedy(self, state) -> int:
        """return the greedy action"""
        # your code here #
        return self.forward(state)[1]


def MC_func_approx(
    env,
    qnet,
    optimizer,
    epsilon: float = 0.1,
    nr_episodes: int = 50000,
    max_t: int = 1000,
    gamma: float = 0.99,
    hide_progress: bool = False,
):
    """Monte Carlo control with function approximation."""
    SAR = namedtuple("SAR", ["state", "action", "reward"])
    episode_returns = []
    episode_lengths = []
    # greedy_return: = []

    with tqdm.trange(
        nr_episodes, desc="Training", unit="episodes", disable=hide_progress
    ) as tepisodes:
        for e in tepisodes:
            trajectory = []
            # generate trajectory
            state = env.reset()[0]

            for t in range(max_t):
                # choose action according to epsilon greedy
                action = qnet.act_epsilon_greedy(state, epsilon)

                # take a step
                observation, reward, terminated, truncated, info = env.step(action)
                trajectory.append(SAR(state, action, reward))

                # update current state
                state = observation

                # if the environment is in a terminal state stop the sampling
                if terminated or truncated:
                    break

            # compute episode reward
            discounts = [gamma**i for i in range(len(trajectory) + 1)]
            R = sum([a * b for a, (_, _, b) in zip(discounts, trajectory)])
            episode_returns.append(R)
            episode_lengths.append(len(trajectory))

            # update q-values from trajectory
            loss = torch.zeros([1])
            g = 0
            for state, action, reward in reversed(trajectory):
                # your code here
                g = gamma * g + reward
                q = qnet.forward(state)[0][action]
                J_k = (g - q) ** 2 / len(trajectory)
                loss += J_k

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print average return of the last 100 episodes
            if e % 10 == 0:
                avg_return = np.mean(episode_returns[-10:])
                avg_length = np.mean(episode_lengths[-10:])
                # greedy_return.append(
                #     evaluate_greedy_policy(  # parallelize?
                #         env, qnet.act_greedy, nr_episodes=10, t_max=1000
                #     )
                # )

                # avg_backtrack_percentage = np.mean(backtrack_percentages[-100:])
                tepisodes.set_postfix(
                    {
                        "episode return": f"{avg_return:.2f}",
                        "episode length": f"{avg_length:3.2f}",
                        # "greedy return": f"{greedy_return[-1]:.2f}",
                        # 'backtrack': "{:.2f}%".format(avg_backtrack_percentage)
                    }
                )

    plot_training(episode_returns, env, "MC_func_approx", nr_episodes // 10)

    # Save the model
    torch.save(
        qnet.state_dict(),
        f"{OUTPUT_PATH}/{NOW.strftime('%Y-%m-%d_%H-%M-%S')}-MC_func_approx_model-{env.spec.id}.pth",
    )
    return qnet


# Named tuple for storing experience steps gathered in training
RB_Experience = namedtuple(
    "RB_Experience",
    ["state", "action", "reward", "done", "new_state"],
)


class ReplayBuffer:
    """
    Replay Buffer for storing past experiences allowing the agent to learn from them
    Args:
        capacity: size of the buffer
    """

    def __init__(self, capacity: int) -> None:
        self.buffer: deque = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def append(self, experience: RB_Experience) -> None:
        """
        Add experience to the buffer
        Args:
            experience: tuple (state, action, reward, done, new_state)
        """
        self.buffer.append(experience)

    def sample(self, batch_size: int):
        # your code here: Sample randomly a batch of experiences of the indicated size
        # from the buffer
        batch = random.sample(
            self.buffer, k=batch_size
        )  # TODO: adjust later to prefer later experiences
        states = np.array([e.state for e in batch])
        actions = np.array([e.action for e in batch])
        rewards = np.array([e.reward for e in batch])
        dones = np.array([e.done for e in batch])
        next_states = np.array([e.new_state for e in batch])
        return states, actions, rewards, dones, next_states


def DQN(
    qnet,
    env,
    optimizer,
    epsilon: float = 0.1,
    gamma: float = 1.0,
    nr_episodes: int = 5000,
    max_t: int = 100,
    replay_buffer_size: int = 1000000,
    batch_size: int = 32,
    warm_start_steps: int = 1000,
    sync_rate: int = 1024,
    train_frequency: int = 8,
):
    """Deep Q-Learning with experience replay and target network."""
    print(
        f"Train policy with DQN for {nr_episodes} episodes using at most {max_t} steps,"
        f" gamma = {gamma}, epsilon = {epsilon}, replay buffer size ="
        f" {replay_buffer_size}, sync rate = {sync_rate}, warm starting steps for"
        f" filling the replay buffer = {warm_start_steps}"
    )

    buffer = ReplayBuffer(replay_buffer_size)
    target_qnet = deepcopy(qnet)
    episode_returns = []
    episode_lengths = []
    nr_terminal_states: list = []

    # populate buffer
    state = env.reset()[0]
    for i in range(warm_start_steps):
        # your code here: populate the buffer with warm_start_steps experiences #
        action = random.randint(0, env.action_space.n - 1)
        observation, reward, terminated, truncated, info = env.step(action)
        buffer.append(RB_Experience(state, action, reward, terminated, observation))
        state = observation
        if terminated or truncated:
            env.reset()

    with tqdm.trange(nr_episodes, desc="DQN Training", unit="episodes") as tepisodes:
        for e in tepisodes:
            state = env.reset()[0]
            episode_return = 0.0
            step_counter = 0

            # Collect trajectory
            for t in range(max_t):
                step_counter = step_counter + 1

                # step through environment with agent and add experience to buffer
                with torch.no_grad():
                    action = qnet.act_epsilon_greedy(state, epsilon)
                    observation, reward, terminated, truncated, info = env.step(action)
                    buffer.append(
                        RB_Experience(state, action, reward, terminated, observation)
                    )
                    state = observation
                    episode_return += reward

                # calculate training loss on sampled batch
                if step_counter % train_frequency == 0:
                    # your code here #
                    states, actions, rewards, dones, next_states = buffer.sample(
                        batch_size
                    )
                    y = torch.zeros(len(states))
                    q = torch.zeros(len(states))

                    for i in range(len(states)):
                        if dones[i]:
                            y[i] = rewards[i]
                        else:
                            qs, max_a = target_qnet.forward(next_states[i])
                            q_next = qs[max_a]
                            y[i] = rewards[i] + gamma * q_next
                        q[i] = qnet.forward(states[i])[0][actions[i]]

                    loss = F.mse_loss(q, y)

                    # update qnet
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # Soft update of target network
                if step_counter % sync_rate == 0:
                    target_qnet.load_state_dict(qnet.state_dict())

                if terminated or truncated:
                    break

            episode_lengths.append(t + 1)
            episode_returns.append(episode_return)
            if e % 10 == 0:
                avg_return = np.mean(episode_returns[-10:])
                avg_length = np.mean(episode_lengths[-10:])

                tepisodes.set_postfix(
                    {
                        "episode return": f"{avg_return:3.2f}",
                        "episode length": f"{avg_length:3.2f}",
                    }
                )

    # Save the model
    torch.save(
        qnet.state_dict(),
        f"{OUTPUT_PATH}/{NOW.strftime('%Y-%m-%d_%H-%M-%S')}-DQN-{env.spec.id}.pth",
    )
    return target_qnet


def run_cartpole(params):
    epsilon = params["epsilon"]
    hidden_size = params["hidden_size"]
    n_layers = params["n_layers"]
    cartpole_env = gym.make("CartPole-v1", render_mode="rgb_array")
    cartpole_observation_space_size = cartpole_env.observation_space.shape[0]
    cartpole_nr_actions = cartpole_env.action_space.n
    cartpole_qnet = QNet(
        cartpole_observation_space_size, cartpole_nr_actions, hidden_size, n_layers
    )
    cartpole_optimizer = torch.optim.SGD(cartpole_qnet.parameters(), lr=1e-3)
    MC_cartpolt_policy = MC_func_approx(
        cartpole_env,
        cartpole_qnet,
        cartpole_optimizer,
        epsilon,
        nr_episodes,
        max_t,
        gamma,
    )
    return -1 * evaluate_greedy_policy(cartpole_env, MC_cartpolt_policy)


def run_mountain_car(params):
    epsilon = params["epsilon"]
    max_t = params["max_t"]
    mountaincar_env = gym.make(
        "MountainCar-v0", render_mode="rgb_array", max_episode_steps=max_t
    )
    mountaincar_observation_space_size = mountaincar_env.observation_space.shape[0]
    mountaincar_nr_actions = mountaincar_env.action_space.n
    mountaincar_qnet = QNet(
        mountaincar_observation_space_size,
        mountaincar_nr_actions,
        5,
        2,
    )
    mountaincar_optimizer = torch.optim.RMSprop(mountaincar_qnet.parameters(), lr=1e-2)

    mountaincar_qnet = MC_func_approx(
        mountaincar_env,
        mountaincar_qnet,
        mountaincar_optimizer,
        epsilon,
        1000,
        max_t,
        gamma,
    )

    return -1 * evaluate_greedy_policy(
        mountaincar_env, mountaincar_qnet.act_greedy, 10, 1000
    )


if __name__ == "__main__":
    # Get the current date and time
    NOW = datetime.datetime.now()
    OUTPUT_PATH = "exercise03/output/"

    # epsilon = 0.7213226283289884
    epsilon = 0.7
    nr_episodes = 100
    max_t = 4_000
    gamma = 0.999999
    replay_buffer_size = 10_000
    hidden_size = 5
    n_layers = 1

    # cartpole_env = gym.make("CartPole-v1", render_mode="rgb_array")
    # cartpole_observation_space_size = cartpole_env.observation_space.shape[0]
    # cartpole_nr_actions = cartpole_env.action_space.n
    # cartpole_qnet = QNet(
    #     cartpole_observation_space_size, cartpole_nr_actions, hidden_size, n_layers
    # )
    # cartpole_optimizer = torch.optim.SGD(cartpole_qnet.parameters(), lr=1e-3)

    # cartpole_qnet = MC_func_approx(
    #     cartpole_env,
    #     cartpole_qnet,
    #     cartpole_optimizer,
    #     epsilon,
    #     nr_episodes,
    #     max_t,
    #     gamma,
    # )
    # print(
    #     "Mean episode reward from MC_func_approx on cartpole: ",
    #     evaluate_greedy_policy(cartpole_env, cartpole_qnet.act_greedy),
    # )
    # gym_video(
    #     cartpole_qnet.act_greedy,
    #     cartpole_env,
    #     f"{NOW.strftime('%Y-%m-%d_%H-%M-%S')}-MC_func_approx-{cartpole_env.spec.id}",
    #     10_00_000,
    # )

    # mountaincar_env = gym.make(
    #     "MountainCar-v0", render_mode="rgb_array", max_episode_steps=max_t
    # )
    # mountaincar_observation_space_size = mountaincar_env.observation_space.shape[0]
    # mountaincar_nr_actions = mountaincar_env.action_space.n
    # mountaincar_qnet = QNet(
    #     mountaincar_observation_space_size,
    #     mountaincar_nr_actions,
    #     hidden_size,
    #     n_layers,
    # )
    # mountaincar_optimizer = torch.optim.RMSprop(mountaincar_qnet.parameters(), lr=1e-2)

    # mountaincar_qnet = MC_func_approx(
    #     mountaincar_env,
    #     mountaincar_qnet,
    #     mountaincar_optimizer,
    #     epsilon,
    #     nr_episodes,
    #     max_t,
    #     gamma,
    # )
    # print(
    #     "Mean episode reward from MC_func_approx on mountaincar policy: ",
    #     evaluate_greedy_policy(mountaincar_env, mountaincar_qnet.act_greedy, 10, 4_000),
    # )
    # gym_video(
    #     mountaincar_qnet.act_greedy,
    #     mountaincar_env,
    #     f"{NOW.strftime('%Y-%m-%d_%H-%M-%S')}-MC_func_approx-{mountaincar_env.spec.id}",
    #     4_000,
    # )

    # Cartpole MC
    # # Define the search space for hyperparameters
    # space = {
    #     "epsilon": hp.uniform("epsilon", 0, 1),
    #     "hidden_size": hp.randint("hidden_size", 10) + 1,
    #     "n_layers": hp.randint("n_layers", 10),
    #     # 'dropout': hp.uniform('dropout', 0, 1),
    # }

    # # Run hyperparameter optimization
    # best_params = fmin(
    #     fn=run_cartpole,
    #     space=space,
    #     algo=tpe.suggest,
    #     max_evals=100,
    # )

    # # Print the best hyperparameters
    # print("Best hyperparameters:", best_params)

    # {'epsilon': 0.7213226283289884, 'hidden_size': 5, 'n_layers': 1}

    # # Mountain car MC
    # # Define the search space for hyperparameters
    # space = {
    #     "epsilon": hp.uniform("epsilon", 0, 1),
    #     "max_t": hp.choice("max_t", [1_000, 2_000, 4_000, 8_000, 16_000]),
    # }

    # # Run hyperparameter optimization
    # best_params = fmin(
    #     fn=run_mountain_car,
    #     space=space,
    #     algo=tpe.suggest,
    #     max_evals=20,
    # )

    # # # Print the best hyperparameters
    # print("Best hyperparameters:", best_params)

    # cartpole_env = gym.make("CartPole-v1", render_mode="rgb_array")
    # cartpole_observation_space_size = cartpole_env.observation_space.shape[0]
    # cartpole_nr_actions = cartpole_env.action_space.n
    # cartpole_qnet = QNet(
    #     cartpole_observation_space_size, cartpole_nr_actions, hidden_size, n_layers
    # )

    # qnet = QNet(4, 2, 8)

    # optimizer = torch.optim.SGD(qnet.parameters(), lr=1e-2)
    # optimizer = torch.optim.RMSprop(qnet.parameters(), lr=0.01)

    cartpole_env = gym.make("CartPole-v1", render_mode="rgb_array")
    cartpole_observation_space_size = cartpole_env.observation_space.shape[0]
    cartpole_nr_actions = cartpole_env.action_space.n
    cartpole_qnet = QNet(
        cartpole_observation_space_size, cartpole_nr_actions, hidden_size, n_layers
    )
    cartpole_optimizer = torch.optim.RMSprop(cartpole_qnet.parameters(), lr=1e-2)

    target_qnet = DQN(
        cartpole_qnet,
        cartpole_env,
        cartpole_optimizer,
        epsilon=0.7,
        gamma=0.99,
        nr_episodes=15_000,
        max_t=4000,
        warm_start_steps=500,
        sync_rate=128,
        replay_buffer_size=1000,
        train_frequency=8,
        batch_size=128,
    )
    print(
        "Mean episode reward from DQN on cartpole policy: ",
        evaluate_greedy_policy(cartpole_env, target_qnet.act_greedy, 10, 4_000),
    )

    gym_video(
        cartpole_qnet.act_greedy,
        cartpole_env,
        f"{NOW.strftime('%Y-%m-%d_%H-%M-%S')}-MC_func_approx-{cartpole_env.spec.id}",
        5000,
    )

    mountaincar_env = gym.make(
        "MountainCar-v0", render_mode="rgb_array", max_episode_steps=max_t
    )
    mountaincar_observation_space_size = mountaincar_env.observation_space.shape[0]
    mountaincar_nr_actions = mountaincar_env.action_space.n
    mountaincar_qnet = QNet(
        mountaincar_observation_space_size,
        mountaincar_nr_actions,
        hidden_size,
        n_layers,
    )
    mountaincar_optimizer = torch.optim.RMSprop(mountaincar_qnet.parameters(), lr=1e-2)

    mountaincar_qnet = DQN(
        mountaincar_qnet,
        mountaincar_env,
        mountaincar_optimizer,
        epsilon=0.05,
        gamma=0.99,
        nr_episodes=400,
        max_t=4000,
        warm_start_steps=500,
        sync_rate=128,
        replay_buffer_size=5000,
        train_frequency=8,
    )
    print(
        "Mean episode reward from MC_func_approx on mountaincar policy: ",
        evaluate_greedy_policy(mountaincar_env, mountaincar_qnet.act_greedy, 10, 4_000),
    )

    gym_video(
        mountaincar_qnet.act_greedy,
        mountaincar_env,
        f"{NOW.strftime('%Y-%m-%d_%H-%M-%S')}-MC_func_approx-{mountaincar_env.spec.id}",
        5000,
    )
