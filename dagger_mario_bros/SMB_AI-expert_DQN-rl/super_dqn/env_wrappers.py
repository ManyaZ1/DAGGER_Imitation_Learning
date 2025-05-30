import gym
import cv2
import numpy as np
from collections import deque

class MarioPreprocessor(gym.Wrapper):
    """Preprocessing wrapper for Mario environment"""
    def __init__(self, env, stack_frames=4, skip_frames=4):
        super().__init__(env)
        self.stack_frames = stack_frames
        self.skip_frames = skip_frames
        self.frames = deque(maxlen=stack_frames)
        
        # New observation space: grayscale stacked frames
        self.observation_space = gym.spaces.Box(
            low=0, high=255, 
            shape=(stack_frames, 84, 84), 
            dtype=np.uint8
        )
    
    def preprocess_frame(self, frame):
        """Convert frame to grayscale and resize"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # Resize to 84x84
        resized = cv2.resize(gray, (84, 84))
        return resized
    
    def reset(self, **kwargs):
        # Handle different gym versions
        reset_result = self.env.reset(**kwargs)
        if isinstance(reset_result, tuple):
            obs, info = reset_result
        else:
            obs = reset_result
            info = {}
        
        # Initialize frame stack
        processed = self.preprocess_frame(obs)
        for _ in range(self.stack_frames):
            self.frames.append(processed)
        
        if isinstance(reset_result, tuple):
            return np.array(self.frames), info
        else:
            return np.array(self.frames)
    
    def step(self, action):
        total_reward = 0
        for _ in range(self.skip_frames):
            step_result = self.env.step(action)
            if len(step_result) == 4:  # Old gym format
                obs, reward, done, info = step_result
                truncated = False
            else:  # New gym format
                obs, reward, done, truncated, info = step_result
            
            total_reward += reward
            if done or truncated:
                break
        
        processed = self.preprocess_frame(obs)
        self.frames.append(processed)
        
        # Return in new format consistently
        return np.array(self.frames), total_reward, done, truncated, info
