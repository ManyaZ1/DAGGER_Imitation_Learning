import os
import sys
import time
import torch

# Setup paths
base_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(base_dir, 'expert-SMB_DQN', 'super_dqn'))

# Imports
from dagger_agent import DaggerMarioAgent
from env_wrappers import MarioPreprocessor
from visual_utils import MarioRenderer

import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace


def main():
    model_path = os.path.join(
        base_dir, 'models_dagger', 'mario_FLAG_iter105_ep12_3346_20250608_154904.pth'
    )

    # Init environment
    env_raw = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    env = MarioPreprocessor(JoypadSpace(env_raw, SIMPLE_MOVEMENT))
    env._max_episode_steps = 3000

    # Init agent
    agent = DaggerMarioAgent(env.observation_space.shape, env.action_space.n)
    agent.load_model(model_path)

    # Renderer
    renderer = MarioRenderer(env, scale=3.0)

    # Run one episode
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        renderer.render()
        time.sleep(1 / 60)

        action = agent.act(state, training=False)
        state, reward, done, info = env.step(action)
        total_reward += reward

        if done:
            print(f"Episode finished. X: {info.get('x_pos')} | Score: {total_reward:.2f} | Flag: {info.get('flag_get', False)}")
            break

    env.close()


if __name__ == '__main__':
    main()
