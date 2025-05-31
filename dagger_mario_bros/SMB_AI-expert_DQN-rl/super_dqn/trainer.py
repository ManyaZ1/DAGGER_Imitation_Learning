import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from .agent import MarioAgent
from .env_wrappers import MarioPreprocessor

class MarioTrainer:
    """Training manager for Mario AI"""
    def __init__(self, world='1-1', stage='1', action_type='simple'):
        # Create environment
        env_name = f'SuperMarioBros-{world}-{stage}-v0'
        self.env = gym_super_mario_bros.make(env_name)
        
        # Action space
        if action_type == 'simple':
            actions = SIMPLE_MOVEMENT
        elif action_type == 'complex':
            actions = COMPLEX_MOVEMENT
        else:
            actions = RIGHT_ONLY
        
        self.env = JoypadSpace(self.env, actions)
        self.env = MarioPreprocessor(self.env)
        
        # Agent
        state_shape = self.env.observation_space.shape
        n_actions = self.env.action_space.n
        self.agent = MarioAgent(state_shape, n_actions)
        
        print(f"Environment: {env_name}")
        print(f"State shape: {state_shape}")
        print(f"Number of actions: {n_actions}")
        print(f"Actions: {actions}")
    
    def train(self, episodes=1000, save_freq=100, render=False):
        """Train the Mario agent"""
        print(f"Starting training for {episodes} episodes...")
        
        for episode in range(episodes):
            reset_result = self.env.reset()
            if isinstance(reset_result, tuple):
                state, _ = reset_result
            else:
                state = reset_result
            
            total_reward = 0
            steps = 0
            done = False
            
            while not done:
                if render:
                    self.env.render()
                
                action = self.agent.act(state)
                step_result = self.env.step(action)
                
                # Handle different return formats
                if len(step_result) == 5:
                    next_state, reward, done, truncated, info = step_result
                else:
                    next_state, reward, done, info = step_result
                    truncated = False
                
                # Reward shaping
                reward = self.shape_reward(reward, info, done)
                
                self.agent.remember(state, action, reward, next_state, done or truncated)
                state = next_state
                total_reward += reward
                steps += 1
                
                # Train the agent
                self.agent.replay()
                
                if done or truncated:
                    break
            
            # Track scores
            self.agent.scores.append(total_reward)
            avg_score = np.mean(self.agent.scores[-100:])
            self.agent.avg_scores.append(avg_score)
            
            # Print progress
            if episode % 10 == 0:
                print(f"Episode {episode}/{episodes}")
                print(f"Score: {total_reward:.2f}, Avg Score: {avg_score:.2f}")
                print(f"Epsilon: {self.agent.epsilon:.4f}, Steps: {steps}")
                print(f"World Position: {info.get('x_pos', 0)}")
                print("-" * 50)
            
            # Save model
            if episode % save_freq == 0 and episode > 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = f"models/mario_model_ep{episode}_{timestamp}.pth"  # Custom directory
                os.makedirs("models", exist_ok=True)  # Create directory if it doesn't exist
                self.agent.save_model(save_path)
        
        # Final save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_save_path = f"models/mario_model_final_{timestamp}.pth"
        os.makedirs("models", exist_ok=True)
        self.agent.save_model(final_save_path)
        
        self.plot_training_progress()
        print("Training completed!")
    
    def shape_reward(self, reward, info, done):
        """Custom reward shaping for better learning"""
        shaped_reward = reward
        
        # Encourage moving right
        if 'x_pos' in info:
            shaped_reward += info['x_pos'] * 0.01
        
        # Penalize death
        if done and info.get('life', 3) < 3:
            shaped_reward -= 50
        
        # Bonus for completing level
        if info.get('flag_get', False):
            shaped_reward += 500
        
        return shaped_reward
    
    def test(self, model_path, episodes=5, render=True):
        """Test a trained model"""
        self.agent.load_model(model_path)
        self.agent.epsilon = 0  # No exploration during testing
        
        print(f"Testing model for {episodes} episodes...")
        
        test_scores = []
        
        for episode in range(episodes):
            reset_result = self.env.reset()
            if isinstance(reset_result, tuple):
                state, _ = reset_result
            else:
                state = reset_result
            
            total_reward = 0
            steps = 0
            done = False
            
            while not done:
                if render:
                    self.env.render()
                
                action = self.agent.act(state, training=False)
                step_result = self.env.step(action)
                
                # Handle different return formats
                if len(step_result) == 5:
                    next_state, reward, done, truncated, info = step_result
                else:
                    next_state, reward, done, info = step_result
                    truncated = False
                
                state = next_state
                total_reward += reward
                steps += 1
                
                if done or truncated:
                    break
            
            test_scores.append(total_reward)
            print(f"Test Episode {episode + 1}: Score = {total_reward}, Steps = {steps}")
            print(f"Final Position: {info.get('x_pos', 0)}, Lives: {info.get('life', 3)}")
        
        avg_test_score = np.mean(test_scores)
        print(f"\nAverage Test Score: {avg_test_score:.2f}")
        
        self.env.close()
        return test_scores
    
    def plot_training_progress(self):
        """Plot training progress"""
        if len(self.agent.scores) == 0:
            return
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.agent.scores)
        plt.title('Training Scores')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        
        plt.subplot(1, 2, 2)
        plt.plot(self.agent.avg_scores)
        plt.title('Average Scores (100 episodes)')
        plt.xlabel('Episode')
        plt.ylabel('Average Score')
        
        plt.tight_layout()
        plt.savefig(f'mario_training_progress_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.show()
