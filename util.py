import numpy as np
import torch
import matplotlib.pyplot as plt


def print_policy(env, agent):

    ARROWS = {0: "↑", 1: "↓", 2: "←", 3: "→"}

    for r in range(env.rows):
        for c in range(env.cols):

            pos = (r, c)

            if pos in env.walls:
                print("X", end=" ")

            elif pos == env.goal:
                print("G", end=" ")

            elif pos == env.start:
                print("S", end=" ")

            else:
                if hasattr(agent, "q_table"):
                    best_action = np.argmax(agent.q_table[r, c])

                else:
                    state = torch.FloatTensor([r, c])

                    with torch.no_grad():
                        q_values = agent.online_network(state)

                    best_action = torch.argmax(q_values).item()

                print(ARROWS[best_action], end=" ")

        print()

    print()


def visualize(episode_steps):
    plt.plot(episode_steps)
    plt.title("Steps per episode plot")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.title("DQN Training")
    plt.show()
