import sys
import os
# os.chdir(r"C:\\Users\\navne\Documents\\vs_code\\dlgrad_main")
os.chdir(r"/mnt/c/Users/navne/Documents/vs_code/Reinforcement Learning")
sys.path.append(os.getcwd())     

from QLearning.q_learning import Qlearning
# from SARSA.sarsa import sarsa


class FrozenLake(Qlearning):
    def __init__(self) -> None:
        super().__init__(4, 16)
        self.goal = [15]
        self.hole = [5, 7, 11, 12]

        self.right_bounds = [4, 8, 12, 16]
        self.left_bounds = [-1, 3, 7, 11]
        self.first_row = [0, 1, 2, 3]
        self.last_row = [12, 13, 14, 15]

    def step(self, action: int, state: int) -> int | int | bool:
        row = state

        # Up
        if action == 0:
            new_state = row-4 if row not in self.first_row else row
        # Down
        elif action == 1:
            new_state = row+4 if row not in self.last_row else row
        # Left
        elif action == 2:
            new_state = row-1 if row-1 not in self.left_bounds else row
        # Right
        elif action == 3:
            new_state = row+1 if row+1 not in self.right_bounds else row
        
        if new_state in self.goal:
            reward = 1
            done = True
        elif new_state in self.hole:
            reward = 0
            done = True
        else:
            reward = 0
            done = False

        return new_state, reward, done

    def run(self):
        Qlearning.learn(self) 

    def display_graph(self):
        Qlearning.graph(self)

if __name__ == '__main__':
    frozen_lake = FrozenLake()
    frozen_lake.run()
    frozen_lake.display_graph()