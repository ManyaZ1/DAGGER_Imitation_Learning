import os
import numpy as np
import torch
import logging
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
import matplotlib.pyplot as plt
from collections import deque

from super_dqn.agent import MarioAgent
from super_dqn.env_wrappers import MarioPreprocessor
from super_dqn.visual_utils import MarioRenderer
import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace


@dataclass
class DaggerConfig:
    """Configuration class for DAGGER training parameters."""
    iterations: int = 5
    episodes_per_iter: int = 5
    training_batches_per_iter: int = 20
    expert_model_path: str = 'models/WORKING_MARIO_AGENT.pth'
    world: str = '1'
    stage: str = '1'
    render: bool = False
    save_frequency: int = 1
    early_stopping_threshold: float = 0.95
    max_episode_steps: int = 4000
    log_level: str = 'INFO'


class DaggerTrainer:
    """
    Enhanced DAGGER (Dataset Aggregation) trainer for Mario AI agent.
    
    Features:
    - Comprehensive logging and metrics tracking
    - Early stopping based on expert agreement
    - Configurable training parameters
    - Robust error handling
    - Model checkpointing and recovery
    - Performance visualization
    """
    
    def __init__(self, config: Optional[DaggerConfig] = None):
        self.config = config or DaggerConfig()
        self._setup_logging()
        self._setup_environment()
        self._setup_agents()
        self._setup_directories()
        self._setup_metrics()
        
    def _setup_logging(self):
        """Configure logging system."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'dagger_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def _setup_environment(self):
        """Initialize the Mario environment with proper error handling."""
        try:
            env_name = f'SuperMarioBros-{self.config.world}-{self.config.stage}-v0'
            self.logger.info(f"Initializing environment: {env_name}")
            
            raw_env = gym_super_mario_bros.make(env_name)
            wrapped_env = JoypadSpace(raw_env, SIMPLE_MOVEMENT)
            self.env = MarioPreprocessor(wrapped_env)
            
            # Set maximum episode steps
            self.env._max_episode_steps = self.config.max_episode_steps
            
            self.state_shape = self.env.observation_space.shape
            self.n_actions = self.env.action_space.n
            
            self.logger.info(f"Environment initialized - State shape: {self.state_shape}, Actions: {self.n_actions}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize environment: {e}")
            raise
    
    def _setup_agents(self):
        """Initialize expert and learner agents."""
        try:
            # Initialize agents
            self.expert = MarioAgent(self.state_shape, self.n_actions)
            self.learner = MarioAgent(self.state_shape, self.n_actions)
            
            # Load expert model
            if not os.path.exists(self.config.expert_model_path):
                raise FileNotFoundError(f"Expert model not found: {self.config.expert_model_path}")
            
            self.expert.load_model(self.config.expert_model_path)
            self.logger.info(f"Expert model loaded from: {self.config.expert_model_path}")
            
            # Initialize renderer if needed
            if self.config.render:
                self.renderer = MarioRenderer(self.env, scale=3.0)
                
        except Exception as e:
            self.logger.error(f"Failed to initialize agents: {e}")
            raise
    
    def _setup_directories(self):
        """Create necessary directories for saving models and logs."""
        base_dir = Path(__file__).parent.parent
        self.save_dir = base_dir / 'models' / 'dagger_checkpoints'
        self.plots_dir = base_dir / 'plots'
        
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Save directory: {self.save_dir}")
        self.logger.info(f"Plots directory: {self.plots_dir}")
    
    def _setup_metrics(self):
        """Initialize metrics tracking."""
        self.metrics = {
            'iteration_rewards': [],
            'episode_rewards': [],
            'expert_agreement': [],
            'training_losses': [],
            'episode_lengths': []
        }
        self.best_reward = float('-inf')
        self.expert_agreement_window = deque(maxlen=100)
    
    def _calculate_expert_agreement(self, learner_actions: List[int], expert_actions: List[int]) -> float:
        """Calculate agreement rate between learner and expert actions."""
        if not learner_actions or not expert_actions:
            return 0.0
        
        agreements = sum(1 for la, ea in zip(learner_actions, expert_actions) if la == ea)
        return agreements / len(learner_actions)
    
    def _run_episode(self, iteration: int, episode: int) -> Dict:
        """Run a single episode and collect data."""
        state = self.env.reset()
        done = False
        total_reward = 0
        step_count = 0
        learner_actions = []
        expert_actions = []
        
        try:
            while not done and step_count < self.config.max_episode_steps:
                if self.config.render:
                    self.renderer.render()
                
                # Get actions from both agents
                learner_action = self.learner.act(state)
                expert_action = self.expert.act(state, training=False)
                
                # Store actions for agreement calculation
                learner_actions.append(learner_action)
                expert_actions.append(expert_action)
                
                # Execute learner's action in environment
                next_state, reward, done, info = self.env.step(learner_action)
                
                # Store experience with expert's action as label
                self.learner.remember(state, expert_action, reward, next_state, done)
                
                state = next_state
                total_reward += reward
                step_count += 1
                
                # Early termination if Mario dies or gets stuck
                if info.get('life', 2) < 2:  # Mario died
                    break
                    
        except Exception as e:
            self.logger.error(f"Error during episode {episode} of iteration {iteration}: {e}")
            return {'reward': 0, 'steps': 0, 'agreement': 0.0}
        
        # Calculate expert agreement
        agreement = self._calculate_expert_agreement(learner_actions, expert_actions)
        self.expert_agreement_window.append(agreement)
        
        episode_info = {
            'reward': total_reward,
            'steps': step_count,
            'agreement': agreement,
            'final_x_pos': info.get('x_pos', 0)
        }
        
        self.logger.info(
            f"Episode {episode+1}: Reward={total_reward:.2f}, "
            f"Steps={step_count}, Agreement={agreement:.3f}, "
            f"X-pos={info.get('x_pos', 0)}"
        )
        
        return episode_info
    
    def _train_learner(self, iteration: int) -> float:
        """Train the learner agent on collected data."""
        total_loss = 0.0
        batch_count = 0
        
        try:
            self.logger.info(f"Training learner for iteration {iteration+1}...")
            
            for batch in range(self.config.training_batches_per_iter):
                loss = self.learner.replay()
                if loss is not None:
                    total_loss += loss
                    batch_count += 1
                    
            avg_loss = total_loss / max(batch_count, 1)
            self.logger.info(f"Average training loss: {avg_loss:.6f}")
            
            return avg_loss
            
        except Exception as e:
            self.logger.error(f"Error during training: {e}")
            return float('inf')
    
    def _save_checkpoint(self, iteration: int, metrics: Dict):
        """Save model checkpoint and training metrics."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save model
            model_path = self.save_dir / f'dagger_mario_iter{iteration+1}_{timestamp}.pth'
            self.learner.save_model(str(model_path))
            
            # Save metrics
            metrics_path = self.save_dir / f'metrics_iter{iteration+1}_{timestamp}.json'
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            self.logger.info(f"Checkpoint saved: {model_path}")
            
            return str(model_path)
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
            return None
    
    def _plot_training_progress(self):
        """Generate training progress plots."""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Episode rewards
            ax1.plot(self.metrics['episode_rewards'])
            ax1.set_title('Episode Rewards')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Reward')
            ax1.grid(True)
            
            # Iteration average rewards
            if self.metrics['iteration_rewards']:
                ax2.plot(range(1, len(self.metrics['iteration_rewards']) + 1), 
                        self.metrics['iteration_rewards'], 'o-')
                ax2.set_title('Average Reward per Iteration')
                ax2.set_xlabel('Iteration')
                ax2.set_ylabel('Average Reward')
                ax2.grid(True)
            
            # Expert agreement
            if self.metrics['expert_agreement']:
                ax3.plot(self.metrics['expert_agreement'])
                ax3.set_title('Expert Agreement Rate')
                ax3.set_xlabel('Episode')
                ax3.set_ylabel('Agreement Rate')
                ax3.grid(True)
            
            # Training losses
            if self.metrics['training_losses']:
                ax4.plot(self.metrics['training_losses'])
                ax4.set_title('Training Loss')
                ax4.set_xlabel('Iteration')
                ax4.set_ylabel('Loss')
                ax4.grid(True)
            
            plt.tight_layout()
            
            plot_path = self.plots_dir / f'training_progress_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Training plots saved: {plot_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate plots: {e}")
    
    def _check_early_stopping(self) -> bool:
        """Check if early stopping criteria are met."""
        if len(self.expert_agreement_window) < 50:  # Need sufficient data
            return False
        
        recent_agreement = np.mean(list(self.expert_agreement_window)[-50:])
        
        if recent_agreement >= self.config.early_stopping_threshold:
            self.logger.info(f"Early stopping triggered! Agreement rate: {recent_agreement:.3f}")
            return True
        
        return False
    
    def train(self) -> Dict:
        """
        Main training loop for DAGGER algorithm.
        
        Returns:
            Dict containing final training metrics and best model path.
        """
        self.logger.info("=" * 60)
        self.logger.info("STARTING DAGGER TRAINING")
        self.logger.info("=" * 60)
        self.logger.info(f"Configuration: {self.config}")
        
        best_model_path = None
        
        try:
            for iteration in range(self.config.iterations):
                self.logger.info(f"\n{'='*20} ITERATION {iteration+1}/{self.config.iterations} {'='*20}")
                
                iteration_rewards = []
                iteration_agreements = []
                
                # Collect episodes for this iteration
                for episode in range(self.config.episodes_per_iter):
                    episode_info = self._run_episode(iteration, episode)
                    
                    # Track metrics
                    iteration_rewards.append(episode_info['reward'])
                    iteration_agreements.append(episode_info['agreement'])
                    self.metrics['episode_rewards'].append(episode_info['reward'])
                    self.metrics['expert_agreement'].append(episode_info['agreement'])
                    self.metrics['episode_lengths'].append(episode_info['steps'])
                
                # Calculate iteration statistics
                avg_reward = np.mean(iteration_rewards)
                avg_agreement = np.mean(iteration_agreements)
                
                self.metrics['iteration_rewards'].append(avg_reward)
                
                self.logger.info(f"Iteration {iteration+1} Summary:")
                self.logger.info(f"  Average Reward: {avg_reward:.2f}")
                self.logger.info(f"  Average Agreement: {avg_agreement:.3f}")
                self.logger.info(f"  Best Reward So Far: {max(iteration_rewards):.2f}")
                
                # Train the learner
                avg_loss = self._train_learner(iteration)
                self.metrics['training_losses'].append(avg_loss)
                
                # Save checkpoint
                if (iteration + 1) % self.config.save_frequency == 0:
                    checkpoint_path = self._save_checkpoint(iteration, {
                        'iteration': iteration + 1,
                        'avg_reward': avg_reward,
                        'avg_agreement': avg_agreement,
                        'avg_loss': avg_loss
                    })
                    
                    if avg_reward > self.best_reward:
                        self.best_reward = avg_reward
                        best_model_path = checkpoint_path
                
                # Check early stopping
                if self._check_early_stopping():
                    self.logger.info(f"Early stopping at iteration {iteration+1}")
                    break
            
            # Generate final plots
            self._plot_training_progress()
            
            # Final summary
            self.logger.info("\n" + "="*60)
            self.logger.info("DAGGER TRAINING COMPLETE")
            self.logger.info("="*60)
            self.logger.info(f"Best Average Reward: {self.best_reward:.2f}")
            self.logger.info(f"Final Expert Agreement: {np.mean(list(self.expert_agreement_window)[-10:]):.3f}")
            self.logger.info(f"Best Model: {best_model_path}")
            
            return {
                'best_reward': self.best_reward,
                'best_model_path': best_model_path,
                'final_metrics': self.metrics,
                'total_episodes': len(self.metrics['episode_rewards'])
            }
            
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
            return {'status': 'interrupted', 'best_model_path': best_model_path}
            
        except Exception as e:
            self.logger.error(f"Training failed with error: {e}")
            raise
        
        finally:
            if hasattr(self, 'env'):
                self.env.close()
            if self.config.render and hasattr(self, 'renderer'):
                self.renderer.close()

import os.path as path
def main():
    parent_dir = Path(__file__).parent

    # Create custom configuration
    config = DaggerConfig(
        iterations=10,
        episodes_per_iter=3,
        training_batches_per_iter=25,
        expert_model_path= path.join(
            parent_dir, 'models', 'WORKING_MARIO_AGENT.pth'
        ),
        world='1',
        stage='1',
        render=False,
        save_frequency=2,
        early_stopping_threshold=0.9,
        max_episode_steps=5000,
        log_level='INFO'
    )
    
    # Initialize and run trainer
    trainer = DaggerTrainer(config)
    results = trainer.train()
    
    print(f"\nTraining Results: {results}")


if __name__ == '__main__':
    main()
    