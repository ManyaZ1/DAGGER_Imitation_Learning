import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

# ----------------------------
# Environment
# ----------------------------
env = gym_super_mario_bros.make('SuperMarioBros-v3')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# ----------------------------
# Expert (Rule-based)
# ----------------------------
class ExpertPolicy:
    def act(self, obs):
        # Naive rule: always move right + run
        return 1  # e.g., SIMPLE_MOVEMENT[1] = ['right', 'A']

# ----------------------------
# Student Model
# ----------------------------
class CNNPolicy(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(26880, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        x = x.float() / 255.0
        return self.net(x)

# ----------------------------
# DAGGER Data Collection
# ----------------------------
def collect_data(policy, expert, use_expert=True):
    data = []
    obs = env.reset()
    done = False

    while not done:
        obs_tensor = torch.from_numpy(obs.copy().transpose(2, 0, 1)).unsqueeze(0)
        with torch.no_grad():
            action = expert.act(obs) if use_expert else policy(obs_tensor).argmax().item()

        expert_action = expert.act(obs)
        data.append((obs.copy(), expert_action))  # store a copy

        obs, reward, done, info = env.step(action)

    return data

# ----------------------------
# Train
# ----------------------------
def train(model, dataset):
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(2):
        random.shuffle(dataset)
        for obs, act in dataset:
            x = torch.from_numpy(obs.copy().transpose(2, 0, 1)).unsqueeze(0).float()
            y = torch.tensor([act])
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# ----------------------------
# Main DAGGER Loop
# ----------------------------
def main():
    student = CNNPolicy(num_actions=len(SIMPLE_MOVEMENT))
    expert = ExpertPolicy()
    dataset = []

    for i in range(5):
        print(f"[DAGGER] Iteration {i+1}")
        new_data = collect_data(student, expert, use_expert=(i == 0))
        dataset.extend(new_data)
        train(student, dataset)

    torch.save(student.state_dict(), "mario_dagger_policy.pth")
    env.close()

if __name__ == "__main__":
    main()
