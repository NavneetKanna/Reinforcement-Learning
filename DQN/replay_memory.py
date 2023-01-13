

class ReplayMemory:
    def __init__(self, capacity=2000) -> None:
        self.state = [] 
        self.reward = [] 
        self.action = [] 
        self.next_state = []
        self.capacity = capacity
        self.idx = 0

    def add_experience(
        self, 
        state,
        reward,
        action,
        next_state
    ):
        # Circular Buffer
        if self.idx < self.capacity:
            self.state.append(state)
            self.reward.append(reward)
            self.action.append(action)
            self.next_state.append(next_state)
        else:
            self.state[self.idx] = state
            self.reward[self.idx] = reward
            self.action[self.idx] = action
            self.next_state[self.idx] = next_state

        self.idx = (self.idx+1) % self.capacity 

    def sample(self):
        pass




