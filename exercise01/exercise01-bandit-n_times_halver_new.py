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

rewards_per_action = np.zeros(env.action_space.n) # custom
rewards_cum = [0] # custom

actions = list(range(0, env.action_space.n))

# idea: continuously sample arms (n times each) and then progressively eliminate actions with lowest expected return
n = 1 # how often should each arm be sampled between each halving of actions


for i_episode in range(5000):
  
    print("episode Number is", i_episode)  

    if len(actions) > 1 and (i_episode+1) % (len(actions)*n) == 0:
        ceil = np.ceil((len(actions)/2))
        for x in range(int(ceil)):
            remove = rewards_per_action.argmin()
            rewards_per_action = np.delete(rewards_per_action, remove)
            del actions[remove]

    
    #action = env.action_space.sample() # sampling the "action" array which in this case only contains 10 "options" because there is 10 bandits
    action_index = i_episode % len(actions)
    action = actions[action_index]
    print("action is", action)


    ###########################
        
    # here we taking the next "step" in our environment by taking in our action variable randomly selected above
    observation, reward, done, info = env.step(action) 
    rewards_per_action[action_index] += reward # custom
    rewards_cum.append(rewards_cum[-1]+reward)
    rewards.append(reward)

    print("observation space is: ",observation)
    print("reward variable is: ",reward)
    print("done flag is: ",done)
    print("info variable is: ",info)

print("sum of rewards: " + str(np.sum(rewards)))
print("mean reward per action: " + str(np.sum(rewards)/i_episode))


plt.plot(rewards_cum[1:])
plt.ylabel('rewards')
plt.xlabel('steps')
plt.show()