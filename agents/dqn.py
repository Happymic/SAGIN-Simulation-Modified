import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import numpy as np

class DQNAgent:
    def __init__(self, obs_dim, action_dim, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, learning_rate=0.001):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()
        self.memory = deque(maxlen=2000)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.action_scale = 0.005  # 缩小动作的幅度

    def build_model(self):
        model = nn.Sequential(
            nn.Linear(self.obs_dim, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, self.action_dim)
        )
        return model

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.uniform(-self.action_scale, self.action_scale, size=self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0)
        act_values = self.model(state).detach().numpy()
        return np.clip(act_values[0], -self.action_scale, self.action_scale)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.target_model(torch.FloatTensor(next_state).unsqueeze(0)).detach().numpy()))
            target_f = self.model(torch.FloatTensor(state).unsqueeze(0)).detach().numpy()
            target_f[0] = target
            self.optimizer.zero_grad()
            loss = nn.MSELoss()(self.model(torch.FloatTensor(state).unsqueeze(0)), torch.FloatTensor(target_f))
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
