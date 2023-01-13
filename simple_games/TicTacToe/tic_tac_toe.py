from numpy.random import default_rng
from DQN.dqn_agent import DQNAgent

class TicTacToe(DQNAgent):
    def __init__(self, input_shape=9, output_shape=9, episodes=15000, steps=100) -> None:
        super().__init__(input_shape, output_shape, episodes, steps)
        self.rng = default_rng()

        self.player = input("Do you want to play as X or O ?: ")
        # X -> 1
        # O -> 0
        self.board = [
            ['', '', '']
            ['', '', '']
            ['', '', '']
        ]
        self.episodes = episodes
        self.steps = steps

        self.action_space_n = 9

        self.state = []

        self.batchsize = 32
    

    def take_move(self):
        return self.rng.integers(0, 2, size=2, endpoint=True)

    def map_to_cordinates(n):
        row = (n-1) // 3
        col = (n-1) % 3
        return (row, col)

    def action_space_sample(self) -> int:
        # TODO: Batch
        n = self.rng.integers(1, 9, endpoint=True)
        
        row, col = self.map_to_cordinates(n)

        while self.board[row][col]:
            n = self.rng.integers(1, 9, endpoint=True)
            row, col = self.map_to_cordinates(n)

        return n

    # It is the binary representation that the networks learn
    def binary_board(self):
        pass

    def check_winner(self):
        pass


    def display_board(self):
        print(f"\
            {self.board[0][0]} | {self.board[0][1]} | {self.board[0][2]}\
             ---------------------------------------------------------\
            {self.board[1][0]} | {self.board[1][1]} | {self.board[1][1]}\
             ---------------------------------------------------------\
            {self.board[2][0]} | {self.board[2][1]} | {self.board[2][2]}"
        ) 
        self.pos1 = input("Enter pos 1: ")
        self.pos2 = input("Enter pos 2: ")
    
    def get_reward_done(self) -> int | bool:
        pass

    # This repersentation is used for learning (aka) the state
    def board_reper(self):
        # '' -> 0
        # X (1) -> 1
        # O (0) -> 2

        state = []
        for i in self.board:
            for j in i:
                if j == 0:
                    state.append(2)
                elif j == 1:
                    state.append(1)
                elif j == '':
                    state.append(0)
        
        return state
        
    def step(self, action: int, player: str) -> list | int | bool:
        x, y = self.map_to_cordinates(action)
        
        if player == 'X':
            self.board[x][y] = 1
        else:
            self.board[x][y] = 0

        new_state = self.board_reper()

        reward, done = self.get_reward_done()

        return new_state, reward, done

    def reset(self):
        return [0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    def initial_fill(self):
        for i in range(20):
            state = self.reset()
            done = False

            while not done:
                # Epsilon greedy 
                action = DQNAgent.get_action(state)
                next_state, reward, done = self.step(action, 'X')
                DQNAgent.fill_replay_memory(state, reward, action, next_state)

    def x_agent(self):
        self.initial_fill()
        for episode in range(self.episodes):
            state = self.reset()
            done = False

            while not done:
                # Epsilon greedy 
                action = DQNAgent.get_action(state)
                next_state, reward, done = self.step(action, 'X')
                DQNAgent.fill_replay_memory(state, reward, action, next_state)
                DQNAgent.learn(self.batchsize)

    def o_agent(self):
        pass

if __name__ == '__main__':
    tic = TicTacToe()
    tic.fill_replay_memory()

