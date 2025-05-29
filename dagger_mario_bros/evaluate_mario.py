import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
import torch
import torch.nn as nn
import numpy as np

# Env setup
env = gym_super_mario_bros.make('SuperMarioBros-v3')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# CNN model
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

# Load model
model = CNNPolicy(len(SIMPLE_MOVEMENT))
model.load_state_dict(torch.load("mario_dagger_policy.pth"))
model.eval()

# Evaluation loop
obs = env.reset()
done = False
total_reward = 0

while not done:
    obs_tensor = torch.from_numpy(obs.copy().transpose(2, 0, 1)).unsqueeze(0).float()
    with torch.no_grad():
        action = model(obs_tensor).argmax().item()

    obs, reward, done, info = env.step(action)
    total_reward += reward
    env.render()

env.close()
print(f"Total reward: {total_reward}")
