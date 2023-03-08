import numpy as np
import gym
import gym_bandits
import matplotlib.pyplot as plt

np.random.seed(42) # make runs deterministic for numpy random number generator

env = gym.make('BanditTenArmedGaussian-v0')

print('observation space:', env.observation_space.n, 'dimensional')
print('action space:', env.action_space.n, 'dimensional')

env.seed(34) # make each run the same 
observation = env.reset()

rewards = []
average_rewards = np.zeros(env.action_space.n)
nr_steps_per_action = np.zeros(env.action_space.n)

### your code goes here ###

# idea: continuously sample arms (n times each) and then progressively eliminate actions with lowest expected return by halving
n = 3 # how often should each arm be sampled between each halving of actions
actions = list(range(0, env.action_space.n)) # all possible actions, in array
rewards_per_action = np.zeros(env.action_space.n)

###########################

for i_episode in range(5000):
  
    print("episode Number is", i_episode)   
    
    # action = env.action_space.sample() # sampling the "action" array which in this case only contains 10 "options" because there is 10 bandits
    # action = i_episode % env.action_space.n
    # print("action is", action)

    ### your code goes here ###

    # only remove actions when defined sampling amount has been reached and stop when only 1 action is left
    if i_episode != 0 and len(actions) > 1: 
        if i_episode % (len(actions)*n) == 0:
            floor = np.floor(len(actions)/2) # amount of actions to be removed (half action amount, fractions get floored)
            for x in range(int(floor)): # remove actions that yield lowest avg reward
                remove = rewards_per_action.argmin()
                rewards_per_action = np.delete(rewards_per_action, remove)
                del actions[remove]

    action_index = i_episode % len(actions)
    action = actions[action_index]
    print("action is", action)
    ###########################
        
    # here we taking the next "step" in our environment by taking in our action variable randomly selected above
    observation, reward, done, info = env.step(action) 
    rewards.append(reward)

    ### your code goes here ###
    rewards_per_action[action_index] += reward # used to eliminate action with lowest avg reward
    ###########################

    print("observation space is: ",observation)
    print("reward variable is: ",reward)
    print("done flag is: ",done)
    print("info variable is: ",info)

print("sum of rewards: " + str(np.sum(rewards)))

plt.plot(rewards)
plt.ylabel('rewards')
plt.xlabel('steps')
plt.show()