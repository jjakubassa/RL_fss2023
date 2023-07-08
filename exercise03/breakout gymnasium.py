# import gymnasium as gym

# # from ale_py.roms import Breakout
# from ale_py import ALEInterface
# import ale_py
# import shimmy

# ale = ALEInterface()
# print(gym.envs.registry.keys())
# env = gym.make('ALE/Breakout-v5')

import gymnasium as gym
import numpy as np
import ale_py

from ale_py import ALEInterface

# if using gymnasium
import shimmy

ale = ALEInterface()
# print(gym.envs.registry.keys())
env = gym.make("ALE/Breakout-v5", render_mode="human")

# Reset the environment
observation = env.reset()

# Play one episode (game)
done = False
while not done:
    # Render the game screen
    env.render()

    # Choose an action (in this case, random)
    action = np.random.randint(env.action_space.n)

    # Take a step in the game with the chosen action
    observation, reward, done, truncated, info = env.step(action)

# Close the environment when finished
env.close()
