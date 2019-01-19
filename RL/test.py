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
# print(ACTIONS)

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


# print(act(initial_state,UP))
random.seed(42) # for reproducibility

N_STATES = 4 # we have 4 states
N_EPISODES = 200 # number of episodes we want to play
MAX_EPISODE_STEPS = 10 #maximum number of steps per one episode
MIN_ALPHA = 0.02
alphas = np.linspace(1.0, MIN_ALPHA, N_EPISODES)
gamma = 1.0
eps = 0.2
q_table = dict()#initialize q_table
# print(q_table)

def q(state, action=None):

    if state not in q_table:
        q_table[state] = np.zeros(len(ACTIONS))

    if action is None:
        return q_table[state]

    return q_table[state][action]

# randomize our action
def choose_action(state):
    if random.uniform(0, 1) < eps:
        return random.choice(ACTIONS)
    else:
        return np.argmax(q(state))

for episode in range(N_EPISODES):
    state = initial_state
    total_reward = 0
    alpha = alphas[episode]

    for _ in range(MAX_EPISODE_STEPS):
        action = choose_action(state)
        # State(grid=new_grid, agent_pos=pos), reward, is_done
        next_state, reward, done = act(state, action)
        total_reward += reward
        # import pdb; pdb.set_trace()
        print(q(state)[action])
        q(state)[action] = q(state, action) + \
                alpha * (reward + gamma *  np.max(q(next_state)) - q(state, action))
        state = next_state
        if done:
            break
    # print(f"Episode {episode + 1}: total reward -> {total_reward}")

for keys, values in q_table.items():
    formatted_q_table = np.array(values)
    # print(formatted_q_table)


# print(q_table)
