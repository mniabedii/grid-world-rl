import matplotlib as plt

from gridworld import GridWorld
from qlearning_agent import QLearningAgent
from util import print_policy

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
