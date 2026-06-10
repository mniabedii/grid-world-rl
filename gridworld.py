import numpy as np
import random


class GridWorld:
    def __init__(self):
        self.rows = 5
        self.cols = 5

        self.start = (0, 0)
        self.goal = (4, 4)

        self.agent_pos = self.start

        self.actions = {
            0: (-1, 0),  # up
            1: (1, 0),  # down
            2: (0, -1),  # left
            3: (0, 1),  # right
        }

    def reset(self):
        self.agent_pos = self.start

    def step(self, action):

        dr, dc = self.actions[action]

        row, col = self.agent_pos

        new_row = row + dr
        new_col = col + dc

        reward = -1
        done = False

        if 0 <= new_row < self.rows and 0 <= new_col < self.cols:
            self.agent_pos = (new_row, new_col)
        else:
            reward = -5  # punish if moves out of the gridworld

        if self.agent_pos == self.goal:
            reward = 100
            done = True

        return self.agent_pos, reward, done

    def print_grid(self):
        for r in range(self.rows):
            for c in range(self.cols):

                if (r, c) == self.agent_pos:
                    print("A", end=" ")

                elif (r, c) == self.start:
                    print("S", end=" ")

                elif (r, c) == self.goal:
                    print("G", end=" ")

                else:
                    print(".", end=" ")

            print()

        print()


class QLearningAgent:
    def __init__(self, env):

        self.alpha = 0.1
        self.gamma = 0.99
        self.epsilon = 0.1

        self.q_table = np.zeros((env.rows, env.cols, 4))

    def best_action(self, state):

        if random.random() < self.epsilon:
            return random.randint(0, 3)

        return self.best_action(state)


env = GridWorld()

env.print_grid()

env.step(3)  # right
env.print_grid()

env.step(1)  # down
env.print_grid()
