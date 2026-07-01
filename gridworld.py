import numpy as np
import matplotlib.pyplot as plt


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

        self.walls = {(1, 1), (1, 3), (2, 1), (3, 2)}

    def reset(self):
        self.agent_pos = self.start
        return self.agent_pos

    def get_state(self):
        return np.array(self.agent_pos, dtype=np.float32)

    def step(self, action):

        dr, dc = self.actions[action]

        row, col = self.agent_pos

        new_row = row + dr
        new_col = col + dc

        reward = -1
        done = False

        # check bounds
        if not (0 <= new_row < self.rows and 0 <= new_col < self.cols):
            return self.agent_pos, -5, False

        # check walls
        if (new_row, new_col) in self.walls:
            return self.agent_pos, -10, False

        # move agent
        self.agent_pos = (new_row, new_col)

        # goal
        if self.agent_pos == self.goal:
            return self.agent_pos, 100, True

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

                elif (r, c) in self.walls:
                    print("X", end=" ")

                else:
                    print(".", end=" ")

            print()

        print()
