from .network import DQN
from .replay_buffer import ReplayBuffer

import torch
import torch.nn as nn
import torch.optim as optim

import random


class DQNAgent:

    def __init__(self, gamma, epsilon, buffer_capacity):

        self.gamma = gamma
        self.epsilon = epsilon

        self.online_network = DQN()
        self.target_network = DQN()
        self.target_network.load_state_dict(
            self.online_network.state_dict()
        )  # make them identical

        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

        self.optimizer = optim.Adam(self.online_network.parameters(), lr=0.001)
        self.loss_function = nn.MSELoss()

    def choose_action(self, state):
        # explore
        if random.random() < self.epsilon:
            return random.randint(0, 3)

        # exploit
        state = torch.FloatTensor(state)
        q_values = self.online_network(state)
        return torch.argmax(q_values).item()

    def update_target_network(self):
        self.target_network.load_state_dict(self.online_network.state_dict())

    def learn(self, batch_size):
        # check if we have enough experiences
        if len(self.replay_buffer) < batch_size:
            return

        # sample a batch
        batch = self.replay_buffer.sample(batch_size)

        # split the whole experience tuple
        states, actions, rewards, next_states, dones = zip(*batch)

        # convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)  # indices
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # compute current q-values & target q-values
        current_q = self.online_network(states).gather(1, actions.unsqueeze(1))

        # don't build a computation graph and don't compute gradients
        with torch.no_grad():
            next_q = self.target_network(next_states).max(dim=1)[0]

        # bellman equation
        target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = self.loss_function(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
