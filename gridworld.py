import numpy as np
import random
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


env = GridWorld()
agent = QLearningAgent(env)

episodes = 500
episode_steps = []

env.print_grid()

for episode in range(episodes):

    state = env.reset()
    done = False

    steps = 0
    while not done:

        action = agent.choose_action(state)

        next_state, reward, done = env.step(action)

        agent.learn(state, action, reward, next_state, done)

        state = next_state

        steps += 1

    episode_steps.append(steps)
    if episode % 50 == 0:
        print(f"Episode {episode}, steps = {steps}")

plt.plot(episode_steps)
plt.title("GridWorld Q-Learning Progress")
plt.xlabel("Episode")
plt.ylabel("Steps")
plt.show()
