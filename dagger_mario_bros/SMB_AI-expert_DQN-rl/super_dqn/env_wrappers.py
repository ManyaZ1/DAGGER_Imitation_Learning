import gym
import cv2
import numpy as np
from collections import deque

# ----- Wrapper για το περιβάλλον του Mario ----- {grayscale, resize, stacking}
class MarioPreprocessor(gym.Wrapper):
    '''Preprocessing wrapper για το περιβάλλον Mario'''
    def __init__(self,
                 env:          gym.Env,
                 stack_frames: int = 4,
                 skip_frames:  int = 4) -> None:
        super().__init__(env)

        # Πλήθος καρέ που διατηρούνται στη στοίβα (default: 4)
        self.stack_frames = stack_frames

        # frame skipping (default: 4)
        self.skip_frames = skip_frames

        # Κυκλική ουρά που κρατάει τα τελευταία frames!
        self.frames = deque(maxlen=stack_frames)
        
        # New observation space: grayscale stacked frames
        # Ορίζεται το observation space του Agent:
        # grayscale εικόνα διαστάσεων 84x84 και stack_frames κανάλια
        self.observation_space = gym.spaces.Box(
            low = 0,
            high = 255, 
            shape = (stack_frames, 84, 84), 
            dtype = np.uint8
        )

        return
    
    def preprocess_frame(self, frame):
        '''Μετατροπή frame σε grayscale και αλλαγή μεγέθους'''
        # Μετατροπή RGB σε grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Resize σε 84x84 για να ταιριάζει με το DQN input!
        resized = cv2.resize(gray, (84, 84))

        return resized
    
    def reset(self, **kwargs):
        '''Επαναφορά περιβάλλοντος και αρχικοποίηση stack'''
        reset_result = self.env.reset(**kwargs)
        if isinstance(reset_result, tuple):
            obs, info = reset_result
        else:
            obs = reset_result
            info = {}
        
        # Προεπεξεργασία του αρχικού frame
        processed = self.preprocess_frame(obs)
        # Αρχικοποίηση του stack με το ίδιο frame N φορές
        for _ in range(self.stack_frames):
            self.frames.append(processed)
        
        if isinstance(reset_result, tuple):
            return np.array(self.frames), info
        else:
            return np.array(self.frames)
    
    def step(self, action):
        '''Εκτέλεση action με skip και επιστροφή stack εικόνων'''

        total_reward = 0  # Συνολική ανταμοιβή για το action
        obs          = None
        done         = False
        truncated    = False
        info         = {}

        # Επαναληπτική εκτέλεση του ίδιου action (frame skipping)
        for _ in range(self.skip_frames):
            step_result = self.env.step(action)

            # ψηλοέλεγχος ασφάλειας για αντιμετώπιση διαφορετικών εκδόσεων gym!
            if len(step_result) == 4:
                obs, reward, done, info = step_result
                truncated = False
            else:
                obs, reward, done, truncated, info = step_result

            total_reward += reward

            # Αν έχει τελειώσει το επεισόδιο, σταμάτα την επανάληψη!
            if done or truncated:
                break

        # Προεπεξεργασία τελευταίου frame (μετά από skip)
        processed = self.preprocess_frame(obs)

        # Ενημέρωση της στοίβας με το νέο frame
        self.frames.append(processed)

        # Επιστρέφεται η στοίβα από καρέ, η συνολική ανταμοιβή,
        # και flags για τερματισμό επεισοδίου
        return np.array(self.frames), total_reward, done, truncated, info
