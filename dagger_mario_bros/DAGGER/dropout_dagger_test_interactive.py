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
temp = os.path.abspath(os.path.join(base_dir, '..'))
sys.path.append(temp)
from observation_wrapper import PartialObservationWrapper

from dropout_dagger_trainer import DropoutDaggerTrainer, DropoutDaggerConfig

class MarioRenderer:
    '''
    Ορίστηκε νέα κλάση MarioRenderer ειδικά για το DropoutDAGGER, ώστε ο χρήστης να μπορεί
    κατά τη διάρκεια του testing να εναλλάσσεται δυναμικά μεταξύ της καθαρής και της
    θορυβώδους/partial εικόνας που βλέπει ο Agent στην πράξη.
    '''
    def __init__(self, env, scale=3.):
        self.env = env
        self.scale = scale
        self.show_processed = False

        return

    def toggle_view(self):
        self.show_processed = not self.show_processed

        return

    def render(self, frame_clean, frame_processed=None):
        frame = frame_processed if (self.show_processed and frame_processed is not None) else frame_clean
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.resize(frame, (0, 0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_NEAREST)
        
        # Add visual indicator of current mode
        mode_text = 'PROCESSED VIEW' if self.show_processed else 'CLEAN VIEW'
        color = (0, 0, 255) if self.show_processed else (0, 255, 0)  # Red for processed, green for clean
        cv2.putText(frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, 'Press SPACE to toggle', (
            10, frame.shape[0] - 20
        ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Mario DropoutDAGGER Viewer', frame)
        key = cv2.waitKey(1)
        if key == 32:  # SPACE key toggles
            self.toggle_view()
        time.sleep(1 / 60)

        return

def create_processed_frame_from_obs(obs_wrapper, clean_frame):
    '''
    Create a visualization of what the agent actually sees by applying
    the observation transformation to the clean frame.
    '''
    if obs_wrapper is None:
        return clean_frame
    
    # Convert clean frame to the same format as the observation preprocessing
    gray_frame = cv2.cvtColor(clean_frame, cv2.COLOR_RGB2GRAY)
    
    # Resize to observation size (usually 84x84)
    obs_size = 84
    resized_frame = cv2.resize(
        gray_frame, (obs_size, obs_size), interpolation=cv2.INTER_AREA
    )
    
    # Apply the observation transformation
    if obs_wrapper.obs_type == 'noisy':
        noise_level = getattr(obs_wrapper, 'noise_level', 0.2)
        noise = np.random.normal(0, noise_level, resized_frame.shape)
        processed_frame_small = np.clip(resized_frame.astype(np.float32) / 255. + noise, 0, 1)
        processed_frame_small = (processed_frame_small * 255).astype(np.uint8)
    elif obs_wrapper.obs_type == 'partial':
        # For partial, just show the reduced version
        processed_frame_small = resized_frame
    elif obs_wrapper.obs_type == 'downsampled':
        # Further downsample
        small_frame = cv2.resize(resized_frame, (42, 42), interpolation=cv2.INTER_AREA)
        processed_frame_small = cv2.resize(small_frame, (obs_size, obs_size), interpolation=cv2.INTER_NEAREST)
    else:
        processed_frame_small = resized_frame
    
    # Resize back to original frame size
    processed_frame_resized = cv2.resize(
        processed_frame_small, 
        (clean_frame.shape[1], clean_frame.shape[0]), 
        interpolation=cv2.INTER_NEAREST
    )
    
    # Convert grayscale back to RGB for display
    processed_frame_rgb = cv2.cvtColor(processed_frame_resized, cv2.COLOR_GRAY2RGB)
    
    return processed_frame_rgb

def test_dropout_dagger_model(model_path, observation_type='partial', noise_level=0.2, dropout_rate=0.5):
    '''Test a specific DropoutDAgger model'''
    
    print(f"\n🎮 Testing DropoutDAgger Model 🎮")
    print(f"Model: {model_path}")
    print(f"Observation type: {observation_type}")
    print(f"Dropout rate: {dropout_rate}")
    if observation_type == 'noisy':
        print(f"Noise level: {noise_level}")
    
    # DAGGER config (minimal so we can use the test method!!!)
    config = DropoutDaggerConfig(
        observation_type=observation_type,
        noise_level=noise_level,
        dropout_rate=dropout_rate,
        iterations=1,
        episodes_per_iter=1,
        training_batches_per_iter=1,
        expert_model_path=' ',
        render=True,
        only_for_testing=True
    )

    # Load trainer & wrapper
    trainer = DropoutDaggerTrainer(config)
    obs_wrapper = PartialObservationWrapper(
        obs_type=config.observation_type,
        noise_level=config.noise_level
    ) if config.observation_type else None
    
    renderer = MarioRenderer(trainer.env, scale=3.)

    trainer.learner.load_model(model_path)
    trainer.learner.epsilon = 0

    print('\n-> Testing DropoutDAgger model. Press SPACE to toggle between clean and processed view.\n')
    print('   CLEAN VIEW: What humans see')
    print('   PROCESSED VIEW: Approximation of what the agent sees (with transformations)')

    controller_overlay = NESControllerOverlay()
    for ep in range(3):
        state = trainer.env.reset()
        done = False
        total_reward = 0
        trainer.prev_x_pos = 40

        # Variables to track repeated actions and Mario's position
        action_history = []
        position_history = []
        force_no_action = False

        while not done:
            # Render clean frame
            clean_frame = trainer.env.render(mode='rgb_array')

            # Prepare observation for the model
            if obs_wrapper:
                processed_obs = obs_wrapper.transform_observation(state)
            else:
                processed_obs = state
            
            # Create a visualization of what the agent sees
            processed_frame = create_processed_frame_from_obs(obs_wrapper, clean_frame)

            # Render with toggle support
            renderer.render(clean_frame, processed_frame)

            current_x_pos = trainer.prev_x_pos  # Track current position
            # Check if Mario is stuck (no response from environment) - Ξεκολλάμε το gym!!!
            if len(action_history) >= 10:
                # Check if last 10 actions are the same
                last_10_actions = action_history[-10:]
                if len(set(last_10_actions)) == 1:
                    # Check if Mario's position hasn't changed significantly - Env froze!
                    if len(position_history) >= 10:
                        last_10_positions = position_history[-10:]
                        position_change = max(last_10_positions) - min(last_10_positions)
                        if position_change <= 5:  # Mario hasn't moved much
                            force_no_action = True
            
            if force_no_action:
                # Force "no action" - Help agent with the gym bs!
                action = 0
                force_no_action = False
                # Clear history to restart detection
                action_history = []
                position_history = []
            else:
                action = trainer.learner.act(processed_obs)

            # Track action and position history
            action_history.append(action)
            position_history.append(current_x_pos)
            
            # Show controller input
            buttons = SIMPLE_MOVEMENT[action]
            controller_overlay.show(buttons)

            next_state, reward, done, info = trainer.env.step(action)
            reward = trainer.shape_reward(reward, info, done)

            state = next_state
            total_reward += reward

        print(f"\nEpisode {ep+1} completed! Score: {total_reward:.2f}, Final X: {info.get('x_pos', 0)}")
        time.sleep(2)

    trainer.env.close()
    cv2.destroyAllWindows()

    return

def main():
    """Main testing interface for DropoutDAgger models"""
    
    print("🎮 DROPOUT DAGGER MODEL TESTER 🎮")
    print("\nAvailable test configurations:")
    print("1. Test Full State Model")
    print("2. Test Partial Observations Model") 
    print("3. Test Noisy Model (σ=0.1)")
    print("4. Test Noisy Model (σ=0.2)")
    print("5. Test Downsampled Model")
    print("6. Custom Model Path")
    
    choice = input("\nSelect test configuration (1-6): ").strip()
    
    base_dir = os.path.dirname(__file__)
    results_dir = os.path.join(base_dir, 'dropout_dagger_results')
    
    try:
        if choice == '1':
            # You'll need to update these paths with actual trained models
            model_path = os.path.join(results_dir, 'full_state_model.pth')
            test_dropout_dagger_model(model_path, observation_type=None, dropout_rate=0.5)
            
        elif choice == '2':
            model_path = os.path.join(results_dir, 'partial_obs_model.pth') 
            test_dropout_dagger_model(model_path, observation_type='partial', dropout_rate=0.5)
            
        elif choice == '3':
            model_path = os.path.join(results_dir, 'noisy_01_model.pth')
            test_dropout_dagger_model(model_path, observation_type='noisy', noise_level=0.1, dropout_rate=0.5)
            
        elif choice == '4':
            model_path = os.path.join(results_dir, 'noisy_02_model.pth')
            test_dropout_dagger_model(model_path, observation_type='noisy', noise_level=0.2, dropout_rate=0.5)
            
        elif choice == '5':
            model_path = os.path.join(results_dir, 'downsampled_model.pth')
            test_dropout_dagger_model(model_path, observation_type='downsampled', dropout_rate=0.5)
            
        elif choice == '6':
            model_path = input("Enter full path to model file: ").strip()
            observation_type = input("Enter observation type (partial/noisy/downsampled or leave empty for full): ").strip()
            if observation_type == '':
                observation_type = None
            
            noise_level = 0.1
            if observation_type == 'noisy':
                noise_input = input("Enter noise level (default 0.1): ").strip()
                if noise_input:
                    noise_level = float(noise_input)
            
            dropout_input = input("Enter dropout rate (default 0.5): ").strip()
            dropout_rate = float(dropout_input) if dropout_input else 0.5
            
            test_dropout_dagger_model(model_path, observation_type, noise_level, dropout_rate)
            
        else:
            print("Invalid choice. Please select 1-6.")
            return
            
    except FileNotFoundError as e:
        print(f"\n❌ Model file not found: {e}")
        print("Make sure you have trained the DropoutDAgger models first using dropout_dagger_main.py")
    except KeyboardInterrupt:
        print("\n\nTesting interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()