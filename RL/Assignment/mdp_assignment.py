import numpy as np
from copy import deepcopy
import random

GHOST = "G"
AGENT = "A"
PILLET = "P"
EMPTY = "*"
grid = [
    [GHOST, EMPTY, EMPTY, EMPTY],
    [EMPTY, EMPTY, GHOST, EMPTY],
    [PILLET,GHOST, EMPTY, GHOST], #6by4
    [EMPTY, EMPTY, EMPTY, AGENT],  #Q_table = 20states by 4actions
    [EMPTY, EMPTY, EMPTY, EMPTY]
]

# [print(' | '.join(row)) for row in grid]

UP = "UP"
DOWN = "DOWN"
LEFT = "LEFT"
RIGHT = "RIGHT"
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
# print(ACTIONS)

num_of_states = len(grid[0])*len(grid)

class State:
    def __init__(self, grid, agent_pos):
        self.grid = grid
        self.agent_pos = agent_pos

    def __str__(self):
        return f"State(grid={self.grid}, agent_pos={self.agent_pos})"

initial_state = State(grid=grid, agent_pos=[3, 3]) #start state
# print(initial_state)

# agent state and action
def act(state, action):
    def new_agent_pos(state, action):
        pos = deepcopy(state.agent_pos)
        if action == "UP": # UP
            # we only update the row position, column position is still the same
            pos[0] = max(0, pos[0] - 1)
        elif action == "DOWN": # DOWN
            # we only update the row position, column position is still the same
            pos[0] = min(len(state.grid) - 1, pos[0] + 1)
        elif action == "LEFT": # LEFT
            # we only update the column position, row position is still the same
            pos[1] = max(0, pos[1] - 1)
        elif action == "RIGHT": # RIGHT
            # we only update the column position, row position is still the same
            pos[1] = min(len(state.grid[0]) - 1, pos[1] + 1)
        else:
            raise ValueError(f"Unknown action {action}")
        return pos

    pos = new_agent_pos(state, action)
    grid_item = state.grid[pos[0]][pos[1]]

    new_grid = deepcopy(state.grid)
    old = state.agent_pos
    if grid_item == GHOST:
        reward = -100
        is_done = True
        new_grid[pos[0]][pos[1]] += AGENT
        new_grid[old[0]][old[1]] = EMPTY
    elif grid_item == PILLET:
        reward = 1000
        is_done = True
        new_grid[pos[0]][pos[1]] += AGENT
        new_grid[old[0]][old[1]] = EMPTY
    elif grid_item == EMPTY:
        reward = -1
        is_done = False
        new_grid[pos[0]][pos[1]] = AGENT
        new_grid[old[0]][old[1]] = EMPTY
    elif grid_item == AGENT:
        reward = -5
        is_done = False
    else:
        raise ValueError(f"Unknown grid item {grid_item}")

    # print(State(grid=new_grid, agent_pos=pos), reward, is_done)
    return State(grid=new_grid, agent_pos=pos), reward, is_done

# act(initial_state,"UP")

N_EPISODES = 1 # number of episodes we want to play
MAX_EPISODE_STEPS = 5 #maximum number of steps per one episode

# randomize our action
def choose_action(state):
    return random.choice(ACTIONS)

for episode in range(N_EPISODES):
    state = initial_state

    for _ in range(MAX_EPISODE_STEPS):
        action = choose_action(state)

        next_state, reward, done = act(state, action)
        state = next_state
        # print("Running episode ::", episode+1)
        # print("Agent selected action ::", action)
        # print("The new state after this action is ::", state.grid)
        if done:
            break

'''VALUE ITERATION ALGORITHM'''
# -5 + 1.0(max([1/4*0,1/4*0,1/4*0,1/4*0]))
v_reward_state = np.array([
        [-5.0, -1.0, -1.0, -1.0],
        [-1.0, -1.0, -5.0, -1.0],
        [100,  -5.0, -1.0, -5.0],
        [-1.0, -1.0, -1.0,  0.0],
        [-1.0, -1.0, -1.0, -1.0]
])

def get_adjacent_indices(i, j, m, n):
  adjacent_indices = []
  if i > 0:
      adjacent_indices.append((v_reward_state[i-1][j]))
  if i+1 < m:
      adjacent_indices.append((v_reward_state[i+1][j]))
  if j > 0:
      adjacent_indices.append((v_reward_state[i][j-1]))
  if j+1 < n:
      adjacent_indices.append((v_reward_state[i][j+1]))
  return adjacent_indices

def update_v_table_states(reward_matrix, zero_matrix, gamma=1):
    for i in range(0,5):
        for j in range(0,4):
            adjacent_values = get_adjacent_indices(i, j, 5, 4)
            length = len(adjacent_values)
            list_of_adjacent_values = [(ele/length)*zero_matrix[i][j] for ele in adjacent_values]

            print(list_of_adjacent_values)
            zero_matrix[i][j] = reward_matrix[i][j] + gamma*(max(list_of_adjacent_values))
            zero_matrix[i][j] = '{:.3g}'.format(zero_matrix[i][j])

""" Value-iteration algorithm """
def value_iteration():
    V_table = np.zeros((len(grid),len(grid[0]))) #initialize value of V table
    for x in range(1):
        update_v_table_states(v_reward_state, V_table, 0.6)
        # print(V_table)

value_iteration()
