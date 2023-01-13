import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng


class Qlearning:
    def __init__(self, no_of_actions: int, no_of_states: int) -> None:
        self.episodes = 15000 
        self.lr = 0.3
        self.steps = 99
        self.gamma = 0.95

        self.epsilon = 1.0
        self.max_epsilon = 1.0        
        self.min_epsilon = 0.01    
        self.decay_rate = 0.005

        self.total_rewards = [0] * self.episodes
        self.total_steps = [0] * self.episodes
        
        self.no_of_actions = no_of_actions
        self.no_of_states = no_of_states

        self.rng = default_rng()

        self.rewards = []

        # For plotting
        self.exploration_dict = {i: 0 for i in range(self.no_of_actions)}
 
        self.exploitation_dict = {i: 0 for i in range(self.no_of_actions)}
 
        self.state_dict = {i: 0 for i in range(self.no_of_states)}

        # self.action = {
        #     0 : 'up',
        #     1 : 'down',
        #     2 : 'left',
        #     3 : 'right'
        # }

        self.q_table = np.zeros((self.no_of_states, self.no_of_actions), dtype=np.float32)

    # TODO: move to frozen lake 
    def reset(self) -> int:
        return 0

    # TODO: move to frozen lake 
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
            for _ in range(self.steps):
                self.state_dict[state] += 1
                stepss += 1

                #------- Choose Action --------
                n = self.rng.uniform(0, 1) 
                # Exploitation 
                if n > self.epsilon:
                    action = np.argmax(self.q_table[state, :])
                    self.exploitation_dict[action] += 1
                # Exploration
                else:
                    action = self.action_sapce_sample()
                    self.exploration_dict[action] += 1

                #-------- Perform the action ---------
                new_state, reward, done = self.step(action, state)
                tr += reward

                #-------- Update the Q-Table ----------
                a = self.q_table[state, action] + self.lr*(reward + self.gamma*np.max(self.q_table[new_state, :]) - self.q_table[state, action])
                self.q_table[state, action] = a

                state = new_state

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
        fig.savefig("qlearning_Metrics.png")