import gym
import cv2
import numpy as np
from collections import deque

# ----- Wrapper για το περιβάλλον του Mario ----- {grayscale, resize, stacking}
class EnhancedMarioPreprocessor(gym.Wrapper):
    '''Enhanced preprocessing wrapper that includes structured state info'''

    def __init__(self, env: gym.Env, stack_frames: int = 4, skip_frames: int = 4) -> None:
        super().__init__(env)
        
        self.stack_frames = stack_frames
        self.skip_frames = skip_frames
        self.frames = deque(maxlen=stack_frames)
        
        # Track additional state information
        self.prev_x_pos = 0
        self.prev_score = 0
        self.prev_coins = 0
        self.prev_time = 400  # Starting time in Mario
        
        # Enhanced observation space includes visual + structured data
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255, 
            shape=(stack_frames, 84, 84), 
            dtype=np.uint8
        )
        
        # Additional structured state size (we'll concatenate this to the CNN features)
        self.structured_state_size = 8  # x_pos, y_pos, score_delta, coins_delta, time_delta, lives, world, stage

    def get_structured_state(self, info):
        '''Extract structured information from the environment'''
        x_pos = info.get('x_pos', 0)
        y_pos = info.get('y_pos', 0)
        score = info.get('score', 0)
        coins = info.get('coins', 0)
        time = info.get('time', 400)
        life = info.get('life', 3)
        world = info.get('world', 1)
        stage = info.get('stage', 1)
        
        # Calculate deltas (changes from previous step)
        x_delta = x_pos - self.prev_x_pos
        score_delta = score - self.prev_score
        coins_delta = coins - self.prev_coins
        time_delta = time - self.prev_time
        
        # Update previous values
        self.prev_x_pos = x_pos
        self.prev_score = score
        self.prev_coins = coins
        self.prev_time = time
        
        # Normalize values to reasonable ranges
        structured_state = np.array([
            x_pos / 3000.0,        # Normalize x position (levels are ~3000 units wide)
            y_pos / 240.0,         # Normalize y position (screen height)
            x_delta / 10.0,        # Movement delta
            score_delta / 1000.0,  # Score change
            coins_delta,           # Coin change
            time / 400.0,          # Time remaining (normalized)
            life / 3.0,            # Lives remaining
            world,                 # World number
        ], dtype=np.float32)
        
        return structured_state

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        
        # Reset tracking variables
        self.prev_x_pos = 0
        self.prev_score = 0
        self.prev_coins = 0
        self.prev_time = 400
        
        processed = self.preprocess_frame(obs)
        for _ in range(self.stack_frames):
            self.frames.append(processed)
        
        return np.array(self.frames)

    def step(self, action):
        total_reward = 0
        obs = None
        done = False
        info = {}

        for _ in range(self.skip_frames):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break

        processed = self.preprocess_frame(obs)
        self.frames.append(processed)
        
        return np.array(self.frames), total_reward, done, info

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        '''Convert frame to grayscale and resize'''
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84))
        return resized
