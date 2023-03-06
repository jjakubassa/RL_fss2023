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

for i_episode in range(5000):
  
    print("episode Number is", i_episode)   
    
    if i_episode < 500:
        #action = env.action_space.sample() # sampling the "action" array which in this case only contains 10 "options" because there is 10 bandits
        action = i_episode % env.action_space.n
            
        print("action is", action)

    ### your code goes here ###
    elif i_episode == 500:
        action = rewards_per_action.argmax()
        print("action is", action)
    
    else:
        print("action is", action)

    ###########################
        
    # here we taking the next "step" in our environment by taking in our action variable randomly selected above
    observation, reward, done, info = env.step(action) 
    rewards_per_action[action] += reward # custom
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