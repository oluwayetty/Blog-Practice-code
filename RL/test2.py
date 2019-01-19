import numpy as np
from copy import deepcopy
import random

ZOMBIE = "Z"
AGENT = "A"
GOLD = "G"
EMPTY = "*"
grid = [
    [GOLD, EMPTY],
    [ZOMBIE, AGENT]
]
# [print(' , '.join(row)) for row in grid]

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
ACTIONS = [UP, DOWN, LEFT, RIGHT]
print(ACTIONS)

class State:
    def __init__(self, grid, agent_pos):
        self.grid = grid
        self.agent_pos = agent_pos

    def __eq__(self, other):
        return isinstance(other, State) and self.grid == other.grid and self.agent_pos == other.agent_pos

    def __hash__(self):
        return hash(str(self.grid) + str(self.agent_pos))

    def __str__(self):
        return f"State(grid={self.grid}, agent_pos={self.agent_pos})"

# start state
initial_state = State(grid=grid, agent_pos=[1, 1])

# agent state and action
def act(state, action):
    def new_agent_pos(state, action):
        pos = deepcopy(state.agent_pos)
        if action == 0: # UP
            # we only update the row position, column position is still the same
            pos[0] = max(0, pos[0] - 1)
        elif action == 1: # DOWN
            # we only update the row position, column position is still the same
            pos[0] = min(len(state.grid) - 1, pos[0] + 1)
        elif action == 2: # LEFT
            # we only update the column position, row position is still the same
            pos[1] = max(0, pos[1] - 1)
        elif action == 3: # RIGHT
            # we only update the column position, row position is still the same
            pos[1] = min(len(state.grid[0]) - 1, pos[1] + 1)
        else:
            raise ValueError(f"Unknown action {action}")
        print(pos)
        return pos

    pos = new_agent_pos(state, action)
    grid_item = state.grid[pos[0]][pos[1]]

    new_grid = deepcopy(state.grid)
    # print("**************", new_grid)
    if grid_item == ZOMBIE:
        reward = -100
        is_done = True
        new_grid[pos[0]][pos[1]] += AGENT
    elif grid_item == GOLD:
        reward = 1000
        is_done = True
        new_grid[pos[0]][pos[1]] += AGENT
    elif grid_item == EMPTY:
        reward = -1
        is_done = False
        old = state.agent_pos
        new_grid[old[0]][old[1]] = EMPTY
        new_grid[pos[0]][pos[1]] = AGENT
    elif grid_item == AGENT:
        reward = -5
        is_done = False
    else:
        raise ValueError(f"Unknown grid item {grid_item}")

    # print(State(grid=new_grid, agent_pos=pos), reward, is_done)
    return State(grid=new_grid, agent_pos=pos), reward, is_done


# print(act(initial_state,2))
random.seed(42) # for reproducibility

'''
initialize Q_table
rows represents the states
columns represents the actions
'''
state_space_size = len(grid)*len(grid[0])
action_space_size =  len(ACTIONS)
q_table = np.zeros((state_space_size, action_space_size))
print(q_table) #returns table of 4rows by 4columns


# # Q-learning parameters
num_episodes = 200 #num of episodes agents plays during training
'''
max num of steps to take in each episodeself.
If agent take the 100 steps wihout reaching goal,
game is terminated.
'''
max_steps_per_episode = 10
learning_rate = 0.1
discount_rate = 0.99

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.001 #or 0.01

all_episodes_rewards = []

# #Q-Learning algorithm
for each_episode in range(num_episodes):
    state = initial_state# starting state of game

    done = False # value to express the game is not finished yet
    current_episode_reward = 0

    for step in range(max_steps_per_episode):
        #exploration-exploitation trade off
        exploration_rate_threshold = random.uniform(0,1)
        if exploration_rate_threshold > exploration_rate:
            # choose action that has the high Q value for that state
            action = np.argmax(q(state))
        else:
            # choose an action randomly
            action = random.choice(ACTIONS)

        '''
        step function initiates the action we choose from the conditional statement above
        step also returns a tuple of the following:
        new_state => the new state the agent is, as a result of the action taken in line 64
        reward => reward of the action agent took
        done => A boolean of whether the action ended our episode/agent is done with the episode
        info => diagnostics information about our environment for debugging purpose
        '''
        new_state, reward, done = act(state, action)
        import pdb; pdb.set_trace()

        # update Q-table for Q(s,a)
        old_q_value = (1 - learning_rate) * q_table[state,action]
        learned_q_value = learning_rate * (reward + (discount_rate * np.max(q_table[new_state,:])))
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
    print(all_episodes_rewards)

#
# # compute average reward per a number of eposides
# # EPISODES = 1000
# # rewards_per_hundred_episodes = np.split(np.array(all_episodes_rewards), num_episodes/1000)
# # count = 1000
# # print("*********Average reward per ", 1000 ," episodes ***********\n")
# # for r in rewards_per_hundred_episodes:
# #     # import pdb; pdb.set_trace()
# #     print(count, ": ", str(sum(r/1000)))
# #     count += 1000
#
# # updated Q_table
# print(q_table)
