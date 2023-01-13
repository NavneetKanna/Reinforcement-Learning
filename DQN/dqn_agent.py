import sys
import os
# os.chdir(r'/mnt/c/Users/navne/Documents/vs_code/Reinforcement_Learning/')
sys.path.append(r'/mnt/c/Users/navne/Documents/vs_code/')
sys.path.append(r'C:\Users\navne\Documents\vs_code')

from numpy.random import default_rng
from DQN.dqnnet import DQNNet
import numpy as np
from replay_memory import ReplayMemory


class DQNAgent:
    def __init__(self, input_shape: int, output_shape: int, episodes=15000, steps=100) -> None:
        self.episodes = episodes
        self.steps = steps
        self.pred_network = DQNNet(input_shape, output_shape)
        self.target_network = DQNNet(input_shape, output_shape)
        self.rng = default_rng()
        self.memory = ReplayMemory()
 
    def update_target_net(self):
        pass
    
    def fill_replay_memory(self, state, reward, action, next_state):
        self.memory.add_experience(state, reward, action, next_state) 

    # Only one state is passed and not a batch, hence one action is returned 
    def get_action(self, state: list) -> int:
        #------- Choose Action --------
        n = self.rng.uniform(0, 1) 
        # Exploitation 
        if n > self.epsilon:
            action = np.argmax(self.pred_network.forward(state))
            # self.exploitation_dict[action] += 1
        # Exploration
        else:
            action = self.action_sapce_sample()
            # self.exploration_dict[action] += 1 
        return action

    def learn(self, batchsize):
        pass 