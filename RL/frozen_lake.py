import numpy as np
import gym
import random
import time
from IPython.display import clear_output
from gym import wrappers
# -*- coding: utf-8 -*-

env = gym.make("FrozenLake-v0")

action_space_size = env.action_space.n # number of actions in the environment
state_space_size = env.observation_space.n # number of states in the environment


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
num_episodes = 10000 #num of episodes agents plays during training
'''
max num of steps to take in each episodeself.
If agent take the 100 steps wihout reaching goal,
game is terminated.
'''
max_steps_per_episode = 1000
learning_rate = 0.1
discount_rate = 0.99

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.001 #or 0.01

all_episodes_rewards = []

#Q-Learning algorithm
# for each_episode in range(num_episodes):
#     state = env.reset() # starting state of game
#
#     done = False # value to express the game is not finished yet
#     current_episode_reward = 0
#
#     for step in range(max_steps_per_episode):
#         #exploration-exploitation trade off
#         exploration_rate_threshold = random.uniform(0,1)
#         if exploration_rate_threshold > exploration_rate:
#             # choose action that has the high Q value for that state
#             action = np.argmax(q_table[state,:])
#         else:
#             # choose an action randomly
#             action = env.action_space.sample()
#
#         '''
#         step function initiates the action we choose from the conditional statement above
#         step also returns a tuple of the following:
#         new_state => the new state the agent is, as a result of the action taken in line 64
#         reward => reward of the action agent took
#         done => A boolean of whether the action ended our episode/agent is done with the episode
#         info => diagnostics information about our environment for debugging purpose
#         '''
#         new_state, reward, done, info = env.step(action)
#
#         # update Q-table for Q(s,a)
#         # import pdb; pdb.set_trace()
#         old_q_value = (1 - learning_rate) * q_table[state,action]
#         learned_q_value = learning_rate * (reward + (discount_rate * np.max(q_table[new_state,:])))
#
#         q_table[state,action] = old_q_value + learned_q_value
#
#         state = new_state
#         current_episode_reward += reward
#
#         # check if we are done i.e agent steps in a hole or reached the goal
#         if done == True:
#             break; #end the game
#
#     # once an episode is finished
#     exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*each_episode)
#     all_episodes_rewards.append(current_episode_reward)
#
#
# # compute average reward per a number of eposides
# EPISODES = 1000
# rewards_per_hundred_episodes = np.split(np.array(all_episodes_rewards), num_episodes/1000)
# count = 1000
# print("*********Average reward per ", 1000 ," episodes ***********\n")
# for r in rewards_per_hundred_episodes:
#     print(count, ": ", str(sum(r/1000)))
#     count += 1000
#
# # updated Q_table
# print(q_table)
# Watch our agent play Frozen Lake by playing the best action
# from each state according to the Q-table

# for episode in range(3):
#     # initialize new episode params
#     state = env.reset()
#     done = False
#     print("*****EPISODE ", episode+1, "*****\n\n\n\n")
#     time.sleep(1)
#
#     for step in range(max_steps_per_episode):
#         # Show current state of environment on screen
#         clear_output(wait=True)
#         env.render()
#         time.sleep(0.3)
#
#         # Choose action with highest Q-value for current state
#         action = np.argmax(q_table[state,:])
#         # Take new action
#         new_state, reward, done, info = env.step(action)
#
#         if done:
#             clear_output(wait=True)
#             env.render()
#             if reward == 1:
#                 print("****You reached the goal!****")
#                 time.sleep(3)
#             else:
#                 print("****You fell through a hole!****")
#                 time.sleep(3)
#                 clear_output(wait=True)
#             break
#
#         # Set new state
#         state = new_state

# def value_iteration(env, gamma = 1.0):
#     """ Value-iteration algorithm """
#     env = env.unwrapped
#     v = np.zeros(env.nS)  # initialize value-function
#     max_iterations = 100000
#     eps = 1e-20
#     for i in range(max_iterations):
#         prev_v = np.copy(v)
#         for s in range(env.nS):
#             import pdb; pdb.set_trace()
#             q_sa = [sum([p*(r + prev_v[s_]) for p, s_, r, _ in env.P[s][a]]) for a in range(env.nA)]
#             v[s] = max(q_sa)
#         if (np.sum(np.fabs(prev_v - v)) <= eps):
#             print ('Value-iteration converged at iteration# %d.' %(i+1))
#             break
#     return v
#
# value_iteration(env)
# env.close()
