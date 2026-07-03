from .dqn_agent import DQNAgent

import matplotlib.pyplot as plt
from src.gridworld import GridWorld
from src.util import print_policy, visualize

env = GridWorld()
agent = DQNAgent(gamma=0.99, epsilon=1.0, buffer_capacity=10_000)

batch_size = 32
target_update = 20

episodes = 500
episode_steps = []

epsilon = 1.0  # Start with 100% exploration
epsilon_min = 0.05  # Never decay below 5% exploration
epsilon_decay = 0.985  # Decay rate per episode

train_frequency = 4  # Only run backprop every 4 steps

for episode in range(episodes):

    state = env.reset()
    done = False
    steps = 0
    agent.epsilon = epsilon

    while not done:
        # agent chooses the action with an epsilon-greedy fashion
        action = agent.choose_action(state)
        # agent takes that actions
        next_state, reward, done = env.step(action)
        # record the experience
        agent.replay_buffer.add((state, action, reward, next_state, done))

        # train after each new experience, but not based on the new experience
        if steps % train_frequency == 0:
            agent.learn(batch_size)

        state = next_state
        steps += 1

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    episode_steps.append(steps)
    if episode % 50 == 0:
        print(f"Episode {episode}, Steps: {steps}, Epsilon: {agent.epsilon:.2f}")

    # update the target network every 20 episodes
    if episode % target_update == 0:
        agent.update_target_network()

print_policy(env, agent)
visualize(episode_steps)
