#!/usr/bin/env python3
"""
DAGGER Implementation for Super Mario Bros using the imitation library
Author: AI Assistant
Date: 2025

This implementation uses the professional imitation learning library to train
a Mario agent using the DAGGER (Dataset Aggregation) algorithm.
"""

import os
import sys
import torch
import numpy as np
import gymnasium as gym
import gym as old_gym  # Import old gym for compatibility
from typing import Dict, Any, Optional, Callable
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Imitation learning imports
from imitation.algorithms import dagger
from imitation.data import rollout
from imitation.policies.serialize import load_policy
from imitation.util import util
from imitation.data.wrappers import RolloutInfoWrapper

# Stable Baselines3 imports
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.callbacks import BaseCallback

# Mario environment imports
try:
    import gym_super_mario_bros
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
    from nes_py.wrappers import JoypadSpace
    MARIO_AVAILABLE = True
except ImportError:
    print("Warning: gym_super_mario_bros not available. Using CartPole for demonstration.")
    MARIO_AVAILABLE = False


class GymnasiumWrapper(gym.Wrapper):
    """Convert old gym environment to gymnasium format"""
    
    def __init__(self, env):
        # Convert old gym spaces to gymnasium spaces
        if hasattr(env.observation_space, 'dtype'):
            if isinstance(env.observation_space, old_gym.spaces.Box):
                obs_space = gym.spaces.Box(
                    low=env.observation_space.low,
                    high=env.observation_space.high,
                    shape=env.observation_space.shape,
                    dtype=env.observation_space.dtype
                )
            else:
                obs_space = env.observation_space
        else:
            obs_space = env.observation_space
            
        if isinstance(env.action_space, old_gym.spaces.Discrete):
            action_space = gym.spaces.Discrete(env.action_space.n)
        else:
            action_space = env.action_space
            
        # Initialize with converted spaces
        self.env = env
        self.observation_space = obs_space
        self.action_space = action_space
        
    def reset(self, **kwargs):
        try:
            obs, info = self.env.reset(**kwargs)
            return obs, info
        except (TypeError, ValueError):
            obs = self.env.reset()
            return obs, {}
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        terminated = done
        truncated = False
        return obs, reward, terminated, truncated, info
    
    def render(self, mode="human"):
        return self.env.render()
    
    def close(self):
        return self.env.close()


class MarioWrapper(GymnasiumWrapper):
    """Custom wrapper for Mario environment to work with imitation library"""
    
    def __init__(self, env):
        super().__init__(env)


class PreprocessFrame(gym.ObservationWrapper):
    """Preprocess frames for Mario - grayscale and resize"""
    
    def __init__(self, env, height=84, width=84):
        super().__init__(env)
        self.height = height
        self.width = width
        # Use gymnasium Box space
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(height, width, 1), dtype=np.uint8
        )
    
    def observation(self, obs):
        # Convert to grayscale and resize
        import cv2
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (self.width, self.height))
        return np.expand_dims(resized, axis=-1)


class ExpertPolicy:
    """Wrapper for expert policy to work with imitation library"""
    
    def __init__(self, expert_model_path: str, device: str = "auto"):
        self.device = device
        self.expert_model = None
        self.load_expert(expert_model_path)
    
    def load_expert(self, model_path: str):
        """Load the expert DQN model"""
        if model_path is None:
            print("No expert model provided - using random policy")
            self.expert_model = None
            return
            
        try:
            # Check if file exists
            if not os.path.exists(model_path):
                print(f"Expert model file not found: {model_path}")
                self.expert_model = None
                return
                
            # Try to load as Stable Baselines3 model first
            self.expert_model = DQN.load(model_path, device=self.device)
            print(f"Loaded expert DQN model from: {model_path}")
        except Exception as e:
            try:
                # Try PPO if DQN fails
                self.expert_model = PPO.load(model_path, device=self.device)
                print(f"Loaded expert PPO model from: {model_path}")
            except Exception as e2:
                print(f"Error loading expert model as DQN: {e}")
                print(f"Error loading expert model as PPO: {e2}")
                print("Using random policy as expert (for demonstration)")
                self.expert_model = None
    
    def predict(self, obs, deterministic=True):
        """Predict action using expert policy"""
        if self.expert_model is not None:
            try:
                action, _ = self.expert_model.predict(obs, deterministic=deterministic)
                return action
            except Exception as e:
                print(f"Expert prediction error: {e}")
                return np.random.randint(0, 7)
        else:
            # Random policy fallback
            return np.random.randint(0, 7)  # SIMPLE_MOVEMENT has 7 actions
    
    def __call__(self, obs):
        """Make the policy callable"""
        return self.predict(obs)


def create_mario_env(world: str = "1", stage: str = "1", render_mode: Optional[str] = None):
    """Create and configure Mario environment"""
    if not MARIO_AVAILABLE:
        # Fallback to CartPole for demonstration
        print("Using CartPole environment for demonstration")
        env = gym.make("CartPole-v1")
        return env
    
    # Create Mario environment - older gym_super_mario_bros doesn't support render_mode parameter
    env_name = f'SuperMarioBros-{world}-{stage}-v0'
    try:
        # Try new API first
        env = gym_super_mario_bros.make(env_name, render_mode=render_mode, apply_api_compatibility=True)
    except TypeError:
        # Fallback to old API without render_mode
        env = gym_super_mario_bros.make(env_name)
        if render_mode == "human":
            env.render_mode = "human"
    
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = MarioWrapper(env)  # This now converts to gymnasium format
    env = PreprocessFrame(env)
    
    return env


def create_vec_env(env_fn: Callable, n_envs: int = 1):
    """Create vectorized environment"""
    def make_env():
        return env_fn()
    
    env = DummyVecEnv([make_env for _ in range(n_envs)])
    env = VecFrameStack(env, n_stack=4)  # Stack 4 frames
    return env


class DAGGERTrainingCallback(BaseCallback):
    """Custom callback to monitor DAGGER training progress"""
    
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        # Log episode statistics
        if self.locals.get('dones', [False])[0]:
            episode_reward = self.locals.get('episode_rewards', [0])[0]
            episode_length = self.locals.get('episode_lengths', [0])[0]
            
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            if self.verbose >= 1:
                print(f"Episode reward: {episode_reward:.2f}, Length: {episode_length}")
        
        return True


def train_dagger_mario(
    expert_model_path: str,
    world: str = "1",
    stage: str = "1",
    total_timesteps: int = 100000,
    dagger_rounds: int = 10,
    episodes_per_round: int = 10,
    render: bool = False,
    save_dir: str = "./dagger_models"
):
    """
    Train a Mario agent using DAGGER algorithm
    
    Args:
        expert_model_path: Path to expert model
        world: Mario world (default: "1")
        stage: Mario stage (default: "1") 
        total_timesteps: Total training timesteps
        dagger_rounds: Number of DAGGER rounds
        episodes_per_round: Episodes per DAGGER round
        render: Whether to render environment
        save_dir: Directory to save models
    """
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    
    # Create environment
    env_fn = lambda: create_mario_env(world, stage, render_mode=None)  # Don't pass render_mode to avoid issues
    vec_env = create_vec_env(env_fn, n_envs=1)
    
    # Wrap for rollout collection
    rollout_env = RolloutInfoWrapper(vec_env)
    
    # Create expert policy
    expert_policy = ExpertPolicy(expert_model_path)
    
    # Create learner policy (PPO works better with DAGGER than DQN)
    learner = PPO(
        "CnnPolicy" if MARIO_AVAILABLE else "MlpPolicy",
        rollout_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log=f"{save_dir}/tensorboard/"
    )
    
    # Collect initial expert demonstrations
    print("Collecting initial expert demonstrations...")
    
    def expert_policy_fn(obs):
        """Expert policy function for rollout collection"""
        actions = []
        for o in obs:
            action = expert_policy.predict(o, deterministic=True)
            actions.append(action)
        return np.array(actions)
    
    # Generate expert trajectories
    expert_trajectories = rollout.rollout(
        expert_policy_fn,
        rollout_env,
        rollout.make_sample_until(min_timesteps=episodes_per_round * 200),
        rng=np.random.default_rng(42)
    )
    
    print(f"Collected {len(expert_trajectories)} expert trajectories")
    
    # Create DAGGER trainer
    dagger_trainer = dagger.SimpleDAggerTrainer(
        venv=rollout_env,
        scratch_dir=save_dir,
        expert_policy=expert_policy_fn,
        bc_trainer=None,  # Will be created automatically
        rng=np.random.default_rng(42)
    )
    
    # Training loop
    print(f"Starting DAGGER training for {dagger_rounds} rounds...")
    
    training_stats = {
        'round_rewards': [],
        'round_losses': [],
        'expert_agreement': []
    }
    
    for round_num in range(dagger_rounds):
        print(f"\n=== DAGGER Round {round_num + 1}/{dagger_rounds} ===")
        
        # Train current policy
        if round_num == 0:
            # Initial behavioral cloning
            print("Performing initial behavioral cloning...")
            dagger_trainer.train(
                total_timesteps=total_timesteps // dagger_rounds,
                bc_train_kwargs={"n_epochs": 10}
            )
        else:
            # DAGGER training round
            print(f"DAGGER training round {round_num + 1}")
            dagger_trainer.train(
                total_timesteps=total_timesteps // dagger_rounds
            )
        
        # Evaluate current policy
        print("Evaluating current policy...")
        current_policy = dagger_trainer.policy
        
        # Collect rollouts with current policy
        eval_trajectories = rollout.rollout(
            current_policy,
            rollout_env,
            rollout.make_sample_until(min_episodes=5),
            rng=np.random.default_rng(42 + round_num)
        )
        
        # Calculate statistics
        round_rewards = [traj.rewards.sum() for traj in eval_trajectories]
        avg_reward = np.mean(round_rewards)
        
        training_stats['round_rewards'].append(avg_reward)
        
        print(f"Round {round_num + 1} Results:")
        print(f"  Average Reward: {avg_reward:.2f}")
        print(f"  Episode Count: {len(eval_trajectories)}")
        
        # Save intermediate model
        model_path = os.path.join(save_dir, f"dagger_mario_round_{round_num + 1}.zip")
        if hasattr(dagger_trainer, 'policy') and hasattr(dagger_trainer.policy, 'save'):
            dagger_trainer.policy.save(model_path)
            print(f"Model saved: {model_path}")
    
    # Final evaluation
    print("\n=== Final Evaluation ===")
    final_trajectories = rollout.rollout(
        dagger_trainer.policy,
        rollout_env,
        rollout.make_sample_until(min_episodes=10),
        rng=np.random.default_rng(123)
    )
    
    final_rewards = [traj.rewards.sum() for traj in final_trajectories]
    final_avg_reward = np.mean(final_rewards)
    
    print(f"Final Average Reward: {final_avg_reward:.2f}")
    print(f"Final Reward Std: {np.std(final_rewards):.2f}")
    
    # Save final model
    final_model_path = os.path.join(save_dir, "dagger_mario_final.zip")
    if hasattr(dagger_trainer, 'policy') and hasattr(dagger_trainer.policy, 'save'):
        dagger_trainer.policy.save(final_model_path)
        print(f"Final model saved: {final_model_path}")
    
    # Cleanup
    rollout_env.close()
    
    return {
        'final_reward': final_avg_reward,
        'training_stats': training_stats,
        'final_model_path': final_model_path,
        'trajectories': len(final_trajectories)
    }

base_dir = os.path.dirname(os.path.abspath(__file__))
def main():
    """Main training function"""
    
    # Configuration
    config = {
        'expert_model_path': os.path.join(
            base_dir, '..', 'expert-SMB_DQN', 'models', 'ep30000_MARIO_EXPERT.pth'
        ),
        'world': '1',
        'stage': '1', 
        'total_timesteps': 50000,
        'dagger_rounds': 5,
        'episodes_per_round': 10,
        'render': False,
        'save_dir': './dagger_mario_models'
    }
    
    print("DAGGER Mario Training with Imitation Library")
    print("=" * 50)
    print(f"Configuration: {config}")
    
    # Check if expert model exists
    if not os.path.exists(config['expert_model_path']):
        print(f"Warning: Expert model not found at {config['expert_model_path']}")
        print("Training will use random policy as expert (for demonstration)")
        # Create a dummy expert model file path for the demo
        config['expert_model_path'] = None
    
    try:
        # Run DAGGER training
        results = train_dagger_mario(**config)
        
        print("\n" + "=" * 50)
        print("DAGGER Training Complete!")
        print(f"Final Performance: {results['final_reward']:.2f}")
        print(f"Model saved at: {results['final_model_path']}")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()