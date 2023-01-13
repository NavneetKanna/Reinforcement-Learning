import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng


class SARSA:
    def __init__(self) -> None:
        self.episodes = 15000 
        self.lr = 0.3
        self.steps = 99
        self.gamma = 0.95

        self.epsilon = 1.0
        self.max_epsilon = 1.0        
        self.min_epsilon = 0.01    
        self.decay_rate = 0.005

        self.no_of_actions = 4
        self.no_of_states = 16

        self.total_rewards = [0] * self.episodes
        self.total_steps = [0] * self.episodes
        
        self.rng = default_rng()

        self.goal = [15]
        self.hole = [5, 7, 11, 12]
            
        self.right_bounds = [4, 8, 12, 16]
        self.left_bounds = [-1, 3, 7, 11]
        self.first_row = [0, 1, 2, 3]
        self.last_row = [12, 13, 14, 15]

        self.rewards = []

        # For Plotting
        self.exploration_dict = {
            0: 0,
            1: 0,
            2: 0,
            3: 0
        }
        self.exploitation_dict = {
            0: 0,
            1: 0,
            2: 0,
            3: 0
        }
        self.state_dict = {i: 0 for i in range(16)}
        # self.action = {
        #     0 : 'up',
        #     1 : 'down',
        #     2 : 'left',
        #     3 : 'right'
        # }

        self.q_table = np.zeros((self.no_of_states, self.no_of_actions), dtype=np.float32)
    
    def reset(self) -> int:
        return 0

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
    
    def action_sapce_sample(self) -> int:
        rng = default_rng()
        return rng.integers(0, self.no_of_actions, endpoint=False)

    def learn(self):
        for episode in range(self.episodes):
            if episode % 1000 == 0:
                print(f"episode {episode}")
            state = self.reset()
            stepss = 0
            tr = 0

            n = self.rng.uniform(0, 1) 
            # Exploitation 
            if n > self.epsilon:
                action = np.argmax(self.q_table[state, :])
                self.exploitation_dict[action] += 1
            # Exploration
            else:
                action = self.action_sapce_sample()
                self.exploration_dict[action] += 1

            for _ in range(self.steps):
                self.state_dict[state] += 1
                stepss += 1

                #-------- Perform the action ---------
                new_state, reward, done = self.step(action, state)
                tr += reward

                #--------- Choose Action for next state ------------
                # Exploitation 
                if n > self.epsilon:
                    new_action = np.argmax(self.q_table[new_state, :])
                    self.exploitation_dict[action] += 1
                # Exploration
                else:
                    new_action = self.action_sapce_sample()
                    self.exploration_dict[action] += 1

                #-------- Update the Q-Table ----------
                a = self.q_table[state, action] + self.lr*(reward + self.gamma*(self.q_table[new_state, new_action]) - self.q_table[state, action])
                self.q_table[state, action] = a

                state = new_state
                action = new_action
                
                if done:
                    if reward:
                        self.total_rewards[episode] = reward
                        self.total_steps[episode] = stepss
                    break

            self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon)*np.exp(-self.decay_rate*episode) 
            self.rewards.append(tr)
        print(f"score {sum(self.rewards)/self.episodes}")

    def graph(self):
        print(self.q_table)
        fig, ax = plt.subplots(5, figsize=(8, 8))
        ax[0].bar(range(len(self.total_rewards)), self.total_rewards) 
        ax[1].bar(range(len(self.total_steps)), self.total_steps)
        ax[2].bar(list(self.exploitation_dict.keys()), self.exploitation_dict.values())
        ax[3].bar(list(self.exploration_dict.keys()), self.exploration_dict.values())
        ax[4].bar(list(self.state_dict.keys()), self.state_dict.values())
        plt.close(fig)
        fig.savefig("Sarsa_Metrics.png")

if __name__ == '__main__':
    sarsa = SARSA()
    sarsa.learn()
    sarsa.graph()

