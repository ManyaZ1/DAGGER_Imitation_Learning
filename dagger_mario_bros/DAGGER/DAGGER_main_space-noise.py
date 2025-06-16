import os
import sys
import time
import cv2
import numpy as np
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

# Paths
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(
    os.path.join(base_dir, '..', 'expert-SMB_DQN', 'super_dqn'))
)
from visual_utils import NESControllerOverlay

# Προσθήκη του observation wrapper
base_dir = os.path.dirname(__file__)
temp     = os.path.abspath(os.path.join(base_dir, '..'))
sys.path.append(temp)
from observation_wrapper import PartialObservationWrapper

from dagger_trainer import DaggerTrainer, DaggerConfig

class MarioRenderer:
    '''
    Ορίστηκε νέα κλάση MarioRenderer ειδικά για το DAGGER, ώστε ο χρήστης να μπορεί
    κατά τη διάρκεια του testing να εναλλάσσεται δυναμικά μεταξύ της καθαρής και της
    θορυβώδους εικόνας που βλέπει ο Agent στην πράξη.
    '''
    def __init__(self, env, scale = 3.):
        self.env        = env
        self.scale      = scale
        self.show_noise = False

        return

    def toggle_noise(self):
        self.show_noise = not self.show_noise

        return

    def render(self, frame_clean, frame_noisy=None):
        frame = frame_noisy if (self.show_noise and frame_noisy is not None) else frame_clean
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.resize(frame, (0, 0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_NEAREST)
        
        # Add visual indicator of current mode
        mode_text = 'NOISY VIEW' if self.show_noise else 'CLEAN VIEW'
        color = (0, 0, 255) if self.show_noise else (0, 255, 0) # Red for noisy, green for clean
        cv2.putText(frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, 'Press SPACE to toggle', (
            10, frame.shape[0] - 20
        ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Mario DAGGER Viewer', frame)
        key = cv2.waitKey(1)
        if key == 32:  # SPACE key toggles
            self.toggle_noise()
        time.sleep(1 / 60)

        return

def create_noisy_frame_from_obs(obs_wrapper, clean_frame):
    '''
    Create a proper noisy frame visualization by applying noise to the clean frame
    instead of trying to reconstruct from preprocessed observation.
    '''
    if obs_wrapper.obs_type != 'noisy':
        return clean_frame
    
    # Convert clean frame to the same format as the observation preprocessing
    # This simulates what the agent actually sees
    
    # Convert to grayscale (like in preprocessing)
    gray_frame = cv2.cvtColor(clean_frame, cv2.COLOR_RGB2GRAY)
    
    # Resize to observation size (usually 84x84)
    obs_size = 84  # Assuming standard Atari preprocessing
    resized_frame = cv2.resize(
        gray_frame, (obs_size, obs_size), interpolation = cv2.INTER_AREA
    )
    
    # Add noise similar to what's applied during training
    noise_level = getattr(obs_wrapper, 'noise_level', 0.2)
    noise = np.random.normal(0, noise_level, resized_frame.shape)
    noisy_frame_small = np.clip(resized_frame.astype(np.float32) / 255. + noise, 0, 1)
    
    # Convert back to uint8 and resize to original dimensions
    noisy_frame_small = (noisy_frame_small * 255).astype(np.uint8)
    
    # Resize back to original frame size
    noisy_frame_resized = cv2.resize(
        noisy_frame_small, 
        (clean_frame.shape[1], clean_frame.shape[0]), 
        interpolation = cv2.INTER_NEAREST
    )
    
    # Convert grayscale back to RGB for display
    noisy_frame_rgb = cv2.cvtColor(noisy_frame_resized, cv2.COLOR_GRAY2RGB)
    
    return noisy_frame_rgb

def main():
    model_path = os.path.join(
        base_dir, 'SUCCESS',
        # 'mario_FLAG_iter300_ep1_3437_20250616_194752_noisy.pth'
        'mario_FLAG_iter427_ep2_3434_20250616_233100_noisy.pth'
    )

    # DAGGER config (minimal so we can use the test method!!!)
    config = DaggerConfig(
        observation_type          = 'noisy',
        noise_level               = 0.2,
        iterations                = 1,
        episodes_per_iter         = 1,
        training_batches_per_iter = 1,
        expert_model_path         = ' ',
        render                    = True,
        only_for_testing          = True
    )

    # Load trainer & wrapper
    trainer     = DaggerTrainer(config)
    obs_wrapper = PartialObservationWrapper(
        obs_type    = config.observation_type,
        noise_level = config.noise_level
    )
    renderer    = MarioRenderer(trainer.env, scale = 3.)

    trainer.learner.load_model(model_path)
    trainer.learner.epsilon = 0

    print('\n-> Testing DAGGER model. Press SPACE to toggle between clean and noisy view.\n')
    print('   CLEAN VIEW: What humans see')
    print('   NOISY VIEW: Approximation of what the agent sees (with noise and preprocessing)')

    controller_overlay = NESControllerOverlay()
    for ep in range(3):
        state              = trainer.env.reset()
        done               = False
        total_reward       = 0
        trainer.prev_x_pos = 40

        # Variables to track repeated actions and Mario's position
        action_history   = []
        position_history = []
        force_no_action  = False

        while not done:
            # Render clean frame
            clean_frame = trainer.env.render(mode = 'rgb_array')

            # Prepare noisy observation for the model
            noisy_obs = obs_wrapper.transform_observation(state)
            
            # Create a better noisy frame visualization
            noisy_frame = create_noisy_frame_from_obs(obs_wrapper, clean_frame)

            # Render with toggle support
            renderer.render(clean_frame, noisy_frame)

            current_x_pos = trainer.prev_x_pos # Track current position
            # Check if Mario is stuck (no response from environment) - Ξεκολλάμε το gym!!!
            if len(action_history) >= 10:
                # Check if last 10 actions are the same
                last_10_actions = action_history[-10:]
                if len(set(last_10_actions)) == 1:
                    # Check if Mario's position hasn't changed significantly - Env froze!
                    if len(position_history) >= 10:
                        last_10_positions = position_history[-10:]
                        position_change   = max(last_10_positions) - min(last_10_positions)
                        if position_change <= 5: # Mario hasn't moved much (adjust threshold as needed)
                            force_no_action = True
            
            if force_no_action:
                # Force "no action" - Help agent with the gym bs!
                action          = 0
                force_no_action = False
                # Clear history to restart detection
                action_history   = []
                position_history = []
            else:
                action = trainer.learner.act(noisy_obs)

            # Track action and position history
            action_history.append(action)
            position_history.append(current_x_pos)
            
            # Show controller input
            buttons = SIMPLE_MOVEMENT[action]
            controller_overlay.show(buttons)

            next_state, reward, done, info = trainer.env.step(action)
            reward = trainer.shape_reward(reward, info, done)

            state         = next_state
            total_reward += reward

        print(f"\nEpisode {ep+1} completed! Score: {total_reward:.2f}, Final X: {info.get('x_pos', 0)}")
        time.sleep(2)

    trainer.env.close()
    cv2.destroyAllWindows()

    return

if __name__ == '__main__':
    main()
