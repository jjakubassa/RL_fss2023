# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import argparse
import os
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from gymnasium.experimental.wrappers import RecordVideoV0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-name",
        type=str,
        default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment",
    )
    parser.add_argument("--seed", type=int, default=1, help="seed of the experiment")
    parser.add_argument(
        "--capture-video",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help=(
            "whether to capture videos of the agent performances (check out `videos`"
            " folder)"
        ),
    )

    # Algorithm specific arguments
    parser.add_argument(
        "--env-id", type=str, default="HalfCheetah-v4", help="the id of the environment"
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=1000000,
        help="total timesteps of the experiments",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="the learning rate of the optimizer",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=2048,
        help="the number of steps to run in each environment per policy rollout",
    )
    parser.add_argument(
        "--anneal-lr",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="Toggle learning rate annealing for policy and value networks",
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="the discount factor gamma"
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="the lambda for the general advantage estimation",
    )
    parser.add_argument(
        "--num-minibatches", type=int, default=32, help="the number of mini-batches"
    )
    parser.add_argument(
        "--update-epochs",
        type=int,
        default=10,
        help="the K epochs to update the policy",
    )
    parser.add_argument(
        "--clip-coef",
        type=float,
        default=0.2,
        help="the surrogate clipping coefficient",
    )
    parser.add_argument(
        "--ent-coef", type=float, default=0.0, help="coefficient of the entropy"
    )
    parser.add_argument(
        "--vf-coef", type=float, default=0.5, help="coefficient of the value function"
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.5,
        help="the maximum norm for the gradient clipping",
    )
    parser.add_argument(
        "--target-kl",
        type=float,
        default=None,
        help="the target KL divergence threshold",
    )
    args = parser.parse_args()
    args.batch_size = args.num_steps
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


def make_env(env_id, capture_video, run_name, gamma):
    if capture_video:
        env = gym.make(env_id, render_mode="rgb_array")
    else:
        env = gym.make(env_id)
    env = gym.wrappers.FlattenObservation(
        env
    )  # deal with dm_control's Dict observation space
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if capture_video:
        env = RecordVideoV0(
            env,
            f"videos/{run_name}",
            episode_trigger=lambda episode_id: episode_id % 1000 == 0,
        )
    env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
    env = gym.wrappers.NormalizeReward(env, gamma=gamma)
    env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
    return env


def layer_init(layer, std=0.01, bias_const=0.0):
    if layer.__class__.__name__ == "Linear":
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        # last layer should have very small weights
    return layer


class Agent(nn.Module):
    def __init__(self, env):
        """A simple actor-critic agent"""
        super().__init__()
        ## TODO ##
        # General:
        # - Use orthogonal initialization for weights and 0 initialization for biases using the layer_init function
        # - Use Tanh activation for all hidden layers

        # add critic: A fully connected NN with (n_states, 64) -> (64, 64) -> (64, 1), where n_states is the dimension of the observation space.

        # Initialize critic network
        n_states = np.prod(env.observation_space.shape).astype(int)
        self.critic = nn.Sequential(
            nn.Linear(n_states, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.critic.apply(layer_init)  # Use orthogonal initialization

        # The actor will be parametrized by a Gaussian probability distribution.
        # add actor mean: A fully connected NN with (n_states, 64) -> (64, 64) -> (64, n_actions), where n_actions is the dimensionality of the action space

        # Initialize actor network
        n_actions = np.prod(env.action_space.shape).astype(int)
        self.actor_mean = nn.Sequential(
            nn.Linear(n_states, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, n_actions),
        )
        self.actor_mean.apply(layer_init)  # Use orthogonal initialization

        # self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(env.action_space.shape)))
        # add actor variance. This is just a Parameter (a tensor whose values are learned) with n_actions elements. This will serve as a diagonal variance matrix for a Gaussian policy.
        self.actor_logstd = nn.Parameter(
            torch.zeros(np.prod(env.action_space.shape).astype(int))
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        """
        ## TODO ##
        # Get mean and std from the actor network.
        # If action=None: Sample an action from the resulting Gaussian distribution using Normal(mean, std) from torch.distributions.normal
        # The variance is the exponent of the logstd parameter.
        # Return values:
        # (i) the action,
        # (ii) the probability of the action given the state,
        # (iii) the entropy of the action distribution (can be obtained from the Normal class as well)
        # (iv) and the value of the state.
        """

        # Get mean and std from the actor network
        mean = self.actor_mean(x)
        std = torch.exp(self.actor_logstd)

        # Create a Gaussian distribution with the mean and std
        dist = Normal(mean, std)

        if action == None:
            # Sample an action from the resulting Gaussian distribution using
            # Normal(mean, std) from torch.distributions.normal
            action = dist.sample()

        # Return values:
        # (i) the action,
        # (ii) the probability of the action given the state,
        # (iii) the entropy of the action distribution (can be obtained from the Normal class as well)
        # (iv) and the value of the state.
        return (
            action,
            dist.log_prob(action).sum().item(),
            dist.entropy(),
            self.get_value(x),
        )


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/PPO/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()]),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # env setup
    env = make_env(args.env_id, args.capture_video, run_name, args.gamma)
    assert isinstance(
        env.action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    agent = Agent(env)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps,) + env.observation_space.shape)
    actions = torch.zeros((args.num_steps,) + env.action_space.shape)
    logprobs = torch.zeros(args.num_steps)
    rewards = torch.zeros(args.num_steps)
    dones = torch.zeros(args.num_steps)
    values = torch.zeros(args.num_steps)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    episode_return = 0.0
    episode_length = 0
    start_time = time.time()
    next_obs, _ = env.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs)
    next_done = torch.zeros(1)
    num_updates = args.total_timesteps // args.batch_size
    video_filenames = set()

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1
            obs[step, :] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step, :] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminated, truncated, info = env.step(
                action.cpu().numpy()
            )
            episode_return += reward
            episode_length += 1
            done = terminated or truncated
            rewards[step] = reward
            next_obs, next_done = torch.Tensor(next_obs), done

            # for info in infos["final_info"]:
            #    # Skip the envs that are not done
            #    if info is None:
            #        continue
            if done:
                print(
                    f"global_step={global_step}, episodic_return={episode_return},"
                    f" episodic_length={episode_length}"
                )
                writer.add_scalar("charts/episodic_return", episode_return, global_step)
                episode_return = 0.0
                writer.add_scalar("charts/episodic_length", episode_length, global_step)
                episode_length = 0
                state = env.reset()[0]

        # compute generalized advantage estimates into the advantages tensor
        # Go from the last step to the first step backwards
        # Use args.gamma and args.gae_lambda to compute the advantage.
        # Take care of the edge case at the end (i.e. first for the last step that is computed first) as well as whenever an episode was done.
        with torch.no_grad():
            advantages = torch.zeros_like(rewards)
            # delta = torch.zeros_like(rewards)
            advantages[-1] = rewards[-1] - values[-1]

            # your code here:
            for t in reversed(range(args.num_steps - 1)):  # what about the last step?
                # delta[t] = (
                #     rewards[t] + args.gamma * values[t + 1] * (1 - dones[t]) - values[t]
                # )
                # advantages[t] = delta[t] + args.gamma * args.gae_lambda * advantages[
                #     t + 1
                # ] * (1 - dones[t])

                if dones[t]:
                    delta = rewards[t] - values[t]
                    advantages[t] = delta
                else:
                    delta = rewards[t] + args.gamma * values[t + 1] - values[t]
                    advantages[t] = delta # not generalized
                    # advantages[t] = (
                    #     delta + args.gamma * args.gae_lambda * advantages[t + 1]
                    # )
                    
            ###################
            returns = advantages + values

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    obs[mb_inds], actions[mb_inds]
                )
                logratio = newlogprob - logprobs[mb_inds]
                # for first round = 1
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    ]

                mb_advantages = advantages[mb_inds]

                # Policy loss
                # TODO: compute the clipped surrogate objective for the policy.
                pg_loss = (
                    torch.min(
                        ratio,
                        torch.clamp(
                            ratio, min=1 - args.clip_coef, max=1 + args.clip_coef
                        ),
                    )
                    * mb_advantages
                ).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                # TODO: compute the value loss as the MSE between the new value and the returns
                v_loss = nn.MSELoss()(newvalue, returns[mb_inds])  # td error

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = values.numpy(), returns.numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
        )
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print(
            "SPS:",
            int(global_step / (time.time() - start_time)),
        )
        writer.add_scalar(
            "charts/SPS", int(global_step / (time.time() - start_time)), global_step
        )

        if args.capture_video:
            for filename in os.listdir(f"videos/{run_name}"):
                if filename not in video_filenames and filename.endswith(".mp4"):
                    video_filenames.add(filename)

    env.close()
    writer.close()
