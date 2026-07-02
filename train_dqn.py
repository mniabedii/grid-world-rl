from gridworld import GridWorld
from dqn_agent import DQNAgent

import matplotlib.pyplot as plt

env = GridWorld()
agent = DQNAgent(gamma=0.99, epsilon=0.1, buffer_capacity=10_000)

batch_size = 32
target_update = 20

episodes = 500
episode_steps = []
episode = 0
steps = 0

for episode in range(episodes):

    state = env.reset()
    done = False
    steps = 0

    while not done:
        # agent chooses the action with an epsilon-greedy fashion
        action = agent.choose_action(state)
        # agent takes that actions
        next_state, reward, done = env.step(action)
        # record the experience
        agent.replay_buffer.add((state, action, reward, next_state, done))

        # train after each new experience, but not based on the new experience
        agent.learn(batch_size)

        state = next_state
        steps += 1

    episode_steps.append(steps)
    if episode % 50 == 0:
        print(f"Episode {episode}, Steps: {steps}")

    # update the target network every 20 episodes
    if episode % target_update == 0:
        agent.update_target_network()

print(f"Episode {episode}, Steps: {steps}")

plt.plot(episode_steps)
plt.xlabel("Episode")
plt.ylabel("Steps")
plt.title("DQN Training")
plt.show()
