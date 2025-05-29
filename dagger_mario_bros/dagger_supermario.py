import retro
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from utils.wrappers import MarioWrapper
from utils.expert import ExpertPolicy
import cv2

# ---- Student Policy Model ----
class CNNPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 9 * 9, 128),
            nn.ReLU(),
            nn.Linear(128, 9),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.float() / 255.0
        return self.net(x)

# ---- Dataset Aggregation (DAGGER) ----
def collect_data(policy, expert, env, use_expert=True):
    data = []
    obs = env.reset()
    done = False
    while not done:
        ram = env.unwrapped.get_ram()
        obs_tensor = torch.from_numpy(obs.transpose(2, 0, 1)).unsqueeze(0)
        with torch.no_grad():
            act = policy(obs_tensor).squeeze().numpy() if not use_expert else expert.act(ram)
        expert_action = expert.act(ram)
        data.append((obs, expert_action))
        obs, _, done, _ = env.step(np.round(act).astype(np.uint8))
    return data

def train(model, dataset):
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss()
    for epoch in range(3):
        random.shuffle(dataset)
        for obs, act in dataset:
            x = torch.from_numpy(obs.transpose(2, 0, 1)).unsqueeze(0).float()
            y = torch.tensor(act).unsqueeze(0).float()
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# ---- Main Loop ----
def main():
    env = MarioWrapper(retro.make(game='SuperMarioBros-Nes'))
    expert = ExpertPolicy(env)
    student = CNNPolicy()

    dataset = []

    for i in range(5):
        print(f"[DAGGER] Iteration {i+1}")
        new_data = collect_data(student, expert, env, use_expert=(i == 0))
        dataset.extend(new_data)
        train(student, dataset)

    torch.save(student.state_dict(), 'models/mario_dagger.pth')

if __name__ == "__main__":
    main()