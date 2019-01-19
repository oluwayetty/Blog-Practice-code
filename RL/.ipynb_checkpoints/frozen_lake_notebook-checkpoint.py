{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import gym\n",
    "import random\n",
    "import time\n",
    "from IPython.display import clear_output"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "action_space_size : 4\n",
      "state_space_size : 16\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/anaconda3/lib/python3.6/site-packages/gym/envs/registration.py:14: PkgResourcesDeprecationWarning: Parameters to load are deprecated.  Call .resolve and .require separately.\n",
      "  result = entry_point.load(False)\n"
     ]
    }
   ],
   "source": [
    "env = gym.make(\"FrozenLake-v0\")\n",
    "\n",
    "action_space_size = env.action_space.n # number of actions in the environment\n",
    "state_space_size = env.observation_space.n # number of states in the environment\n",
    "\n",
    "print(\"action_space_size :\", action_space_size) #left, right, up, down.\n",
    "print(\"state_space_size :\", state_space_size)\n",
    "\n",
    "'''\n",
    "initialize Q_table\n",
    "rows represents the states\n",
    "columns represents the actions\n",
    "'''\n",
    "q_table = np.zeros((state_space_size, action_space_size))\n",
    "# print(q_table) #returns table of 4rows by 16 columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Q-learning parameters\n",
    "num_episodes = 10000 #num of episodes agents plays during training\n",
    "'''\n",
    "max num of steps to take in each episodeself.\n",
    "If agent take the 100 steps wihout reaching goal,\n",
    "game is terminated.\n",
    "'''\n",
    "max_steps_per_episode = 1000\n",
    "learning_rate = 0.1\n",
    "discount_rate = 0.99\n",
    "\n",
    "exploration_rate = 1\n",
    "max_exploration_rate = 1\n",
    "min_exploration_rate = 0.01\n",
    "exploration_decay_rate = 0.001 #or 0.01\n",
    "\n",
    "all_episodes_rewards = []"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Q-Learning algorithm\n",
    "for each_episode in range(num_episodes):\n",
    "    state = env.reset() # starting state of game\n",
    "\n",
    "    done = False # value to express the game is not finished yet\n",
    "    current_episode_reward = 0\n",
    "\n",
    "    for step in range(max_steps_per_episode):\n",
    "        #exploration-exploitation trade off\n",
    "        exploration_rate_threshold = random.uniform(0,1)\n",
    "        if exploration_rate_threshold > exploration_rate:\n",
    "            # choose action that has the high Q value for that state\n",
    "            action = np.argmax(q_table[state,:])\n",
    "        else:\n",
    "            # choose an action randomly\n",
    "            action = env.action_space.sample()\n",
    "\n",
    "        '''\n",
    "        step function initiates the action we choose from the conditional statement above\n",
    "        step also returns a tuple of the following:\n",
    "        new_state => the new state the agent is, as a result of the action taken in line 64\n",
    "        reward => reward of the action agent took\n",
    "        done => A boolean of whether the action ended our episode/agent is done with the episode\n",
    "        info => diagnostics information about our environment for debugging purpose\n",
    "        '''\n",
    "        new_state, reward, done, info = env.step(action)\n",
    "\n",
    "        # update Q-table for Q(s,a)\n",
    "        old_q_value = (1 - learning_rate) * q_table[state,action]\n",
    "        learned_q_value = learning_rate * (reward + (discount_rate * np.max(q_table[new_state,:])))\n",
    "        q_table[state,action] = old_q_value + learned_q_value\n",
    "\n",
    "        state = new_state\n",
    "        current_episode_reward += reward\n",
    "\n",
    "        # check if we are done i.e agent steps in a hole or reached the goal\n",
    "        if done == True:\n",
    "            break; #end the game\n",
    "\n",
    "    # once an episode is finished\n",
    "    exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*each_episode)\n",
    "    all_episodes_rewards.append(current_episode_reward)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "*********Average reward per  1000  episodes ***********\n",
      "\n",
      "1000 :  0.048000000000000036\n",
      "2000 :  0.21900000000000017\n",
      "3000 :  0.4110000000000003\n",
      "4000 :  0.5790000000000004\n",
      "5000 :  0.6240000000000004\n",
      "6000 :  0.6450000000000005\n",
      "7000 :  0.6840000000000005\n",
      "8000 :  0.6940000000000005\n",
      "9000 :  0.6710000000000005\n",
      "10000 :  0.7300000000000005\n",
      "[[0.57563891 0.54004013 0.55093959 0.5362634 ]\n",
      " [0.2190774  0.31083534 0.35287118 0.55433518]\n",
      " [0.43788597 0.42659008 0.42085692 0.49853028]\n",
      " [0.19521866 0.32002309 0.25170421 0.46946131]\n",
      " [0.60283023 0.41388079 0.40619746 0.45758768]\n",
      " [0.         0.         0.         0.        ]\n",
      " [0.20446942 0.15934835 0.30247505 0.15861493]\n",
      " [0.         0.         0.         0.        ]\n",
      " [0.36212779 0.48931608 0.47069566 0.66215934]\n",
      " [0.37737245 0.74437607 0.34136174 0.49039308]\n",
      " [0.65851488 0.49658683 0.44932121 0.36130406]\n",
      " [0.         0.         0.         0.        ]\n",
      " [0.         0.         0.         0.        ]\n",
      " [0.59683644 0.59942091 0.82398261 0.63050411]\n",
      " [0.79283246 0.94286226 0.79861212 0.75960016]\n",
      " [0.         0.         0.         0.        ]]\n"
     ]
    }
   ],
   "source": [
    "# compute average reward per a number of eposides\n",
    "EPISODES = 1000\n",
    "rewards_per_hundred_episodes = np.split(np.array(all_episodes_rewards), num_episodes/1000)\n",
    "count = 1000\n",
    "print(\"*********Average reward per \", 1000 ,\" episodes ***********\\n\")\n",
    "for r in rewards_per_hundred_episodes:\n",
    "    print(count, \": \", str(sum(r/1000)))\n",
    "    count += 1000\n",
    "\n",
    "# updated Q_table\n",
    "print(q_table)\n",
    "# Watch our agent play Frozen Lake by playing the best action\n",
    "# from each state according to the Q-table"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  (Down)\n",
      "SFFF\n",
      "FHFH\n",
      "FFFH\n",
      "HFF\u001b[41mG\u001b[0m\n",
      "****You reached the goal!****\n"
     ]
    }
   ],
   "source": [
    "maximum_steps_per_episode = 100\n",
    "for episode in range(3):\n",
    "    # initialize new episode params\n",
    "    state = env.reset()\n",
    "    done = False\n",
    "    print(\"*****EPISODE \", episode+1, \"*****\\n\\n\\n\\n\")\n",
    "    time.sleep(1)\n",
    "\n",
    "    for step in range(maximum_steps_per_episode):\n",
    "        # Show current state of environment on screen\n",
    "        clear_output(wait=True)\n",
    "        env.render()\n",
    "        time.sleep(0.3)\n",
    "\n",
    "        # Choose action with highest Q-value for current state\n",
    "        action = np.argmax(q_table[state,:])\n",
    "        # Take new action\n",
    "        new_state, reward, done, info = env.step(action)\n",
    "\n",
    "        if done:\n",
    "            clear_output(wait=True)\n",
    "            env.render()\n",
    "            if reward == 1:\n",
    "                print(\"****You reached the goal!****\")\n",
    "                time.sleep(3)\n",
    "            else:\n",
    "                print(\"****You fell through a hole!****\")\n",
    "                time.sleep(3)\n",
    "                clear_output(wait=True)\n",
    "            break\n",
    "\n",
    "        # Set new state\n",
    "        state = new_state\n",
    "\n",
    "env.close()\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
