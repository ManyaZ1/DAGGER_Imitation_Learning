# utils/wrappers.py

import gym
import cv2
import numpy as np

class MarioWrapper(gym.ObservationWrapper):
    """
    Preprocesses Super Mario observations:
    - Converts RGB to grayscale
    - Resizes to 84x84
    - Adds channel dimension
    """
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized[:, :, None]
