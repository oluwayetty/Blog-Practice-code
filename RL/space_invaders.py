import numpy as np
import gym
import random
import time
from IPython.display import clear_output
# -*- coding: utf-8 -*-

env = gym.make("SpaceInvaders-v0")
# env = gym.make("SpaceInvaders-ram-v0")
# env = gym.make("Breakout-v0") # tested and working
import pdb; pdb.set_trace()
# env = gym.make("Breakout-ram-v0")
# env = gym.make("MsPacman-v0")
# env = gym.make("MsPacman-ram-v0")

action_space_size = env.action_space.n # number of actions in the environment
state_space_size = env.observation_space.shape[0] # number of states in the environment
#get action names with env.unwrapped.get_action_meanings()
# env.env.get_action_meanings()

print("action_space_size :", action_space_size) #left, right, up, down.
print("state_space_size :", state_space_size)

'''
initialize Q_table
rows represents the states
columns represents the actions
'''
q_table = np.zeros((state_space_size, action_space_size))
print(q_table) #returns table of 4rows by 16 columns


# Q-learning parameters
num_episodes = 500 #num of episodes agents plays during training
'''
max num of steps to take in each episodeself.
If agent take the 100 steps wihout reaching goal,
game is terminated.
'''
max_steps_per_episode = 100
learning_rate = 0.1
discount_rate = 0.99

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.001 #or 0.01

all_episodes_rewards = []

#Q-Learning algorithm
for each_episode in range(num_episodes):
    state = env.reset() # starting state of game

    done = False # value to express the game is not finished yet
    current_episode_reward = 0

    for step in range(max_steps_per_episode):
        #exploration-exploitation trade off
        exploration_rate_threshold = random.uniform(0,1)
        if exploration_rate_threshold > exploration_rate:
            # choose action that has the high Q value for that state
            action = np.argmax(q_table[state,:])
        else:
            # choose an action randomly
            action = env.action_space.sample()

        '''
        step function initiates the action we choose from the conditional statement above
        step also returns a tuple of the following:
        new_state => the new state the agent is, as a result of the action taken in line 64
        reward => reward of the action agent took
        done => A boolean of whether the action ended our episode/agent is done with the episode
        info => diagnostics information about our environment for debugging purpose
        '''
        new_state, reward, done, info = env.step(action)
        import pdb; pdb.set_trace()

        # update Q-table for Q(s,a)
        old_q_value = (1 - learning_rate) * q_table[state,action]
        learned_q_value = learning_rate * (reward + (discount_rate * np.max(q_table[new_state,:])))
        # import pdb; pdb.set_trace()
        q_table[state,action] = old_q_value + learned_q_value

        state = new_state
        current_episode_reward += reward

        # check if we are done i.e agent steps in a hole or reached the goal
        if done == True:
            print("Episode", each_episode  ,"finished after {} timesteps".format(step+1))
            break; #end the game

    # once an episode is finished
    exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*each_episode)
    all_episodes_rewards.append(current_episode_reward)


# compute average reward per a number of eposides
# EPISODES = 1000
# rewards_per_hundred_episodes = np.split(np.array(all_episodes_rewards), num_episodes/1000)
# count = 1000
# print("*********Average reward per ", 1000 ," episodes ***********\n")
# for r in rewards_per_hundred_episodes:
#     # import pdb; pdb.set_trace()
#     print(count, ": ", str(sum(r/1000)))
#     count += 1000

# updated Q_table
print(q_table)
