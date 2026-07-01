import matplotlib as plt
import numpy as np
from gridworld import GridWorld
from qlearning_agent import QLearningAgent


def print_policy(env, agent):
    ARROWS = {0: "↑", 1: "↓", 2: "←", 3: "→"}

    for r in range(env.rows):
        for c in range(env.cols):

            pos = (r, c)

            # wall
            if pos in env.walls:
                print("X", end=" ")

            # goal
            elif pos == env.goal:
                print("G", end=" ")

            # start (optional highlight)
            elif pos == env.start:
                print("S", end=" ")

            else:
                best_action = np.argmax(agent.q_table[r, c])
                print(ARROWS[best_action], end=" ")

        print()

    print()


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

print_policy(env, agent)

plt.plot(episode_steps)
plt.title("GridWorld Q-Learning Progress")
plt.xlabel("Episode")
plt.ylabel("Steps")
plt.show()
