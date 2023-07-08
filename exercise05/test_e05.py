import torch
import gym
from exercise05.exercise05 import Agent


def test_critic_output_shape():
    env = gym.make("CartPole-v0")
    agent = Agent(env)
    observation = env.reset()
    observation_tensor = torch.tensor(observation, dtype=torch.float32)
    value_tensor = agent.get_value(observation_tensor)
    assert value_tensor.shape == torch.Size([1])


def test_get_action_and_value_output_types():
    env = gym.make("CartPole-v0")
    agent = Agent(env)
    observation = env.reset()
    observation_tensor = torch.tensor(observation, dtype=torch.float32)
    action_tensor, log_prob_tensor, entropy_tensor, value_tensor = (
        agent.get_action_and_value(observation_tensor)
    )
    assert isinstance(action_tensor, torch.Tensor)
    assert isinstance(log_prob_tensor, torch.Tensor)
    assert isinstance(entropy_tensor, torch.Tensor)
    assert isinstance(value_tensor, torch.Tensor)


def test_get_action_and_value_output_shapes():
    env = gym.make("CartPole-v0")
    agent = Agent(env)
    observation = env.reset()
    observation_tensor = torch.tensor(observation, dtype=torch.float32)
    action_tensor, log_prob_tensor, entropy_tensor, value_tensor = (
        agent.get_action_and_value(observation_tensor)
    )
    assert action_tensor.shape == torch.Size([1])
    assert log_prob_tensor.shape == torch.Size([1])
    assert entropy_tensor.shape == torch.Size([1])
    assert value_tensor.shape == torch.Size([1])


def test_get_action_and_value_with_action_input():
    env = gym.make("CartPole-v0")
    agent = Agent(env)
    observation = env.reset()
    observation_tensor = torch.tensor(observation, dtype=torch.float32)
    action_tensor = torch.tensor([env.action_space.sample()], dtype=torch.float32)
    action_tensor, log_prob_tensor, entropy_tensor, value_tensor = (
        agent.get_action_and_value(observation_tensor, action_tensor)
    )
    assert isinstance(action_tensor, torch.Tensor)
    assert isinstance(log_prob_tensor, torch.Tensor)
    assert isinstance(entropy_tensor, torch.Tensor)
    assert isinstance(value_tensor, torch.Tensor)
    assert action_tensor.shape == torch.Size([1])
    assert log_prob_tensor.shape == torch.Size([1])
    assert entropy_tensor.shape == torch.Size([1])
    assert value_tensor.shape == torch.Size([1])
