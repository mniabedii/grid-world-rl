import numpy as np
import random


class QLearningAgent:
    def __init__(self, env):

        self.alpha = 0.1
        self.gamma = 0.99
        self.epsilon = 0.1

        self.q_table = np.zeros((env.rows, env.cols, 4))

    def best_action(self, state):

        row, col = state

        return np.argmax(self.q_table[row, col])

    def choose_action(self, state):

        if random.random() < self.epsilon:
            return random.randint(0, 3)

        return self.best_action(state)

    def learn(self, state, action, reward, next_state, done):

        row, col = state
        n_row, n_col = next_state

        current_q = self.q_table[row, col, action]

        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[n_row, n_col])

        self.q_table[row, col, action] = current_q + self.alpha * (target - current_q)
