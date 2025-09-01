import os
import sys
import numpy as np
from datetime import datetime
from typing import Dict, List
from dataclasses import dataclass
import json
from collections import deque
from typing import Optional
import matplotlib.pyplot as plt
import seaborn as sns

# Για το environment access violation reading 0x000000000003C200!
import time
import gc

base_dir = os.path.dirname(__file__)              
pkg_parent = os.path.abspath(os.path.join(base_dir, '..', 'expert-SMB_DQN'))
sys.path.insert(0, pkg_parent)   
super_dqn_path = os.path.abspath(
    os.path.join(base_dir, '..', 'expert-SMB_DQN', 'super_dqn')
) # …/expert-SMB_DQN/super_dqn
sys.path.append(super_dqn_path) # add to PYTHONPATH
from agent import MarioAgent
from env_wrappers import MarioPreprocessor
from visual_utils import MarioRenderer
from trainer import MarioTrainer

from dropout_dagger_agent import DropoutDaggerMarioAgent

import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from gym.wrappers import TimeLimit

# Προσθήκη του observation wrapper
temp = os.path.abspath(os.path.join(base_dir, '..'))
sys.path.append(temp)
from observation_wrapper import PartialObservationWrapper

# Γιατί μας τα ζάλιζε ένα gym...
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='gym')

@dataclass
class DropoutDaggerConfig:
    '''
    Configuration κλάση για τις παραμέτρους εκπαίδευσης DropoutDAGGER.
    Επεκτείνει το DaggerConfig με dropout_rate παράμετρο.
    '''
    iterations: int
    episodes_per_iter: int
    training_batches_per_iter: int
    expert_model_path: str
    dropout_rate: float = 0.5  # Νέα παράμετρος για dropout
    observation_type: Optional[str] = None  # partial, noisy, downsampled...
    noise_level: float = 0.1  # Χρησιμοποιείται για το noisy observation_type!
    world: str = '1'
    stage: str = '1'
    render: bool = False
    save_frequency: int = 1
    max_episode_steps: int = 1000
    only_for_testing: bool = False  # Όταν θέλουμε να κάνουμε απλά testing

class DropoutDaggerTrainer(MarioTrainer):  # Κληρονομεί κυρίως για το test method
    ''' DropoutDAGGER [Dataset Aggregation with Dropout] trainer για τον Mario AI agent. '''
    
    def __init__(self, config: DropoutDaggerConfig):
        self.config = config

        self._setup_environment()
        
        if self.config.only_for_testing:
            print(f'\n-> DropoutDAGGER Trainer initialized in testing mode (dropout_rate={config.dropout_rate}). No training will be performed.\n')
            self._handle_observation_type_settings()
            
            self.learner = DropoutDaggerMarioAgent(self.state_shape, self.n_actions, self.config.dropout_rate)
            self.agent = self.learner
            self.actions = SIMPLE_MOVEMENT
            
            return

        self._setup_agents()
        self._setup_directories()
        self._setup_metrics()

        return
    
    def _handle_observation_type_settings(self):
        if self.config.observation_type == 'partial':
            self.state_shape = list(self.state_shape)
            self.state_shape[0] = 2
            self.state_shape = tuple(self.state_shape)
            print(f'-> State shape for learner: {self.state_shape} - PARTIAL\n')

        self.observation_wrapper = None
        if self.config.observation_type:
            self.observation_wrapper = PartialObservationWrapper(
                obs_type=self.config.observation_type,
                noise_level=self.config.noise_level
            )

        return
    
    def _setup_environment(self):
        '''Δημιουργία environment'''
        env = gym_super_mario_bros.make(f'SuperMarioBros-{self.config.world}-{self.config.stage}-v0')
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        env = TimeLimit(env, max_episode_steps=self.config.max_episode_steps)
        env = MarioPreprocessor(env)

        self.env = env
        self.state_shape = env.observation_space.shape
        self.n_actions = env.action_space.n
        self.prev_x_pos = 40

        print(f'Environment setup: World {self.config.world}-{self.config.stage}')
        print(f'State shape: {self.state_shape}')
        print(f'Number of actions: {self.n_actions}')

        return
    
    def _setup_agents(self):
        '''Δημιουργία expert και learner agents'''
        
        # Handle observation type settings
        self._handle_observation_type_settings()
        
        # Expert agent (πάντα βλέπει full state)
        self.expert = MarioAgent(self.env.observation_space.shape, self.n_actions)
        self.expert.load_model(self.config.expert_model_path)
        self.expert.epsilon = 0  # Δεν θέλουμε exploration από expert
        
        # Learner agent με dropout
        self.learner = DropoutDaggerMarioAgent(self.state_shape, self.n_actions, self.config.dropout_rate)
        
        print(f'Expert agent loaded from: {self.config.expert_model_path}')
        print(f'Learner agent created with dropout_rate={self.config.dropout_rate}')
        
        return
    
    def _setup_directories(self):
        '''Δημιουργία φακέλων για αποθήκευση'''
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        obs_suffix = f"_{self.config.observation_type}" if self.config.observation_type else ""
        dropout_suffix = f"_dropout{self.config.dropout_rate}"
        
        self.save_dir = os.path.join(
            'dropout_dagger_results', 
            f'experiment_{timestamp}{obs_suffix}{dropout_suffix}'
        )
        self.plots_dir = os.path.join(self.save_dir, 'plots')
        
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        
        print(f'Results will be saved to: {self.save_dir}')
        
        return
    
    def _setup_metrics(self):
        '''Αρχικοποίηση metrics για tracking'''
        self.metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'expert_agreements': [],
            'training_losses': [],
            'iteration_metrics': []
        }
        
        return
    
    def _calculate_expert_agreement(self,
                                    learner_actions: List[int],
                                    expert_actions: List[int]) -> float:
        '''Υπολογισμός agreement μεταξύ learner και expert'''
        if len(learner_actions) != len(expert_actions):
            return 0.0
        
        agreements = sum(1 for l, e in zip(learner_actions, expert_actions) if l == e)
        return agreements / len(learner_actions) if learner_actions else 0.0
    
    def _run_episode(self, iteration: int, episode: int) -> Dict:
        '''Εκτέλεση ενός episode με συλλογή δεδομένων'''
        state_full = self.env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        learner_actions = []
        expert_actions = []
        
        self.prev_x_pos = 40  # Reset position tracking
        
        while not done and steps < self.config.max_episode_steps:
            # Expert παίρνει action με full state
            expert_action = self.expert.act(state_full, training=False)
            expert_actions.append(expert_action)
            
            # Μετατροπή observation για learner (αν χρειάζεται)
            if self.observation_wrapper:
                state_partial = self.observation_wrapper.transform_observation(state_full)
            else:
                state_partial = state_full
            
            # Learner παίρνει action με partial/noisy state
            learner_action = self.learner.act(state_partial)
            learner_actions.append(learner_action)
            
            # Αποθήκευση στη DAGGER memory (state_partial, expert_action)
            self.learner.remember(state_partial, expert_action)
            
            # Environment step με expert action
            next_state, reward, done, info = self.env.step(expert_action)
            shaped_reward = self.shape_reward(reward, info, done)
            total_reward += shaped_reward
            
            if self.config.render:
                self.env.render()
            
            state_full = next_state
            steps += 1
        
        # Υπολογισμός expert agreement
        agreement = self._calculate_expert_agreement(learner_actions, expert_actions)
        
        return {
            'reward': total_reward,
            'steps': steps,
            'expert_agreement': agreement,
            'final_x_pos': info.get('x_pos', 0)
        }
    
    def _train_learner(self, iteration: int) -> float:
        '''Εκπαίδευση learner με dropout'''
        if len(self.learner.dagger_memory) < self.learner.batch_size:
            return 0.0
        
        total_loss = 0.0
        num_batches = self.config.training_batches_per_iter
        
        for _ in range(num_batches):
            loss = self.learner.replay()
            if loss is not None:
                total_loss += loss
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def _train_learner_immediate(self, num_batches: int = 3) -> float:
        """Train learner immediately with specified number of batches."""
        if len(self.learner.dagger_memory) < self.learner.batch_size:
            return 0.0
        
        total_loss = 0.0
        successful_batches = 0
        
        for _ in range(num_batches):
            loss = self.learner.replay()
            if loss is not None:
                total_loss += loss
                successful_batches += 1
        
        return total_loss / successful_batches if successful_batches > 0 else 0.0
    
    def _save_checkpoint(self, iteration: int, metrics: Dict):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        obs_suffix = f"_{self.config.observation_type}" if self.config.observation_type else ""
        dropout_suffix = f"_dropout{self.config.dropout_rate}"
        
        # Save model
        model_path = os.path.join(
            self.save_dir, f'dropout_dagger_mario_iter{iteration+1}_{timestamp}{obs_suffix}{dropout_suffix}.pth'
        )
        self.learner.save_model(str(model_path))
        
        # Save στατιστικά
        metrics_path = os.path.join(
            self.save_dir, f'metrics_iter{iteration+1}_{timestamp}.json'
        )
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f'Checkpoint saved: {model_path}')
        print(f'Metrics saved: {metrics_path}')

        return str(model_path)
    
    def train(self) -> Dict:
        '''Κυρίως training loop για DropoutDAGGER'''
        print(f'\n=== DropoutDAGGER Training Started (dropout_rate={self.config.dropout_rate}) ===')
        print(f'Observation type: {self.config.observation_type or "full_state"}')
        if self.config.observation_type == 'noisy':
            print(f'Noise level: {self.config.noise_level}')
        print(f'Iterations: {self.config.iterations}')
        print(f'Episodes per iteration: {self.config.episodes_per_iter}')
        print('=' * 60)
        
        best_score = float('-inf')
        best_model_path = None
        
        for iteration in range(self.config.iterations):
            print(f'\n--- Iteration {iteration+1}/{self.config.iterations} ---')
            
            # Συλλογή episodes
            iteration_rewards = []
            iteration_agreements = []
            iteration_lengths = []
            
            for episode in range(self.config.episodes_per_iter):
                episode_data = self._run_episode(iteration, episode)
                
                iteration_rewards.append(episode_data['reward'])
                iteration_agreements.append(episode_data['expert_agreement'])
                iteration_lengths.append(episode_data['steps'])
                
                print(f'Episode {episode+1}: Reward={episode_data["reward"]:.2f}, '
                      f'Agreement={episode_data["expert_agreement"]:.3f}, '
                      f'Steps={episode_data["steps"]}, '
                      f'X_pos={episode_data["final_x_pos"]}')
                
                # Immediate training after successful episodes
                if episode_data['reward'] > 500:  # Threshold for "successful" episode
                    immediate_loss = self._train_learner_immediate(num_batches=3)
                    print(f'  -> Immediate training loss: {immediate_loss:.6f}')
                    
                    # Save flag model for good performance
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    obs_suffix = f"_{self.config.observation_type}" if self.config.observation_type else "normal"
                    dropout_suffix = f"_dropout{self.config.dropout_rate}"
                    flag_model_path = os.path.join(
                        self.save_dir,
                        f'mario_FLAG_iter{iteration+1}_ep{episode+1}_{int(episode_data["reward"])}_{timestamp}_{obs_suffix}{dropout_suffix}.pth'
                    )
                    self.learner.save_model(flag_model_path)
                    print(f'-> FLAG MODEL SAVED: {flag_model_path}')
            
            # Training του learner στο τέλος κάθε iteration
            print(f'Final training with {len(self.learner.dagger_memory)} experiences...')
            avg_loss = self._train_learner(iteration)
            
            # Metrics για αυτή την iteration
            avg_reward = np.mean(iteration_rewards)
            avg_agreement = np.mean(iteration_agreements)
            avg_length = np.mean(iteration_lengths)
            
            # Update global metrics
            self.metrics['episode_rewards'].extend(iteration_rewards)
            self.metrics['episode_lengths'].extend(iteration_lengths)
            self.metrics['expert_agreements'].extend(iteration_agreements)
            self.metrics['training_losses'].append(avg_loss)
            
            self.metrics['iteration_metrics'].append({
                'iteration': iteration + 1,
                'avg_reward': avg_reward,
                'avg_agreement': avg_agreement,
                'avg_length': avg_length,
                'training_loss': avg_loss,
                'memory_size': len(self.learner.dagger_memory)
            })
            
            print(f'\nIteration {iteration+1} Summary:')
            print(f'  Average Reward: {avg_reward:.2f}')
            print(f'  Average Agreement: {avg_agreement:.3f}')
            print(f'  Average Length: {avg_length:.1f}')
            print(f'  Training Loss: {avg_loss:.6f}')
            print(f'  Memory Size: {len(self.learner.dagger_memory)}')
            
            # Save checkpoint
            if (iteration + 1) % self.config.save_frequency == 0:
                model_path = self._save_checkpoint(iteration, self.metrics)
                
                # Track best model
                if avg_reward > best_score:
                    best_score = avg_reward
                    best_model_path = model_path
        
        # Final training plots
        self._plot_training_results()
        
        print(f'\n=== DropoutDAGGER Training Completed ===')
        print(f'Best average reward: {best_score:.2f}')
        
        return {
            'best_model_path': best_model_path,
            'final_metrics': self.metrics,
            'total_episodes': len(self.metrics['episode_rewards']),
            'best_score': best_score
        }
    
    def _plot_training_results(self):
        """Create comprehensive plots of training results and save to plots directory."""
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'DropoutDAGGER Training Results (dropout_rate={self.config.dropout_rate})', fontsize=16)
        
        # 1. Episode Rewards
        axes[0, 0].plot(self.metrics['episode_rewards'], 'b-', alpha=0.7)
        axes[0, 0].set_title('Episode Rewards Over Time')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add moving average
        if len(self.metrics['episode_rewards']) > 10:
            window = min(50, len(self.metrics['episode_rewards']) // 4)
            moving_avg = np.convolve(self.metrics['episode_rewards'], 
                                   np.ones(window)/window, mode='valid')
            axes[0, 0].plot(range(window-1, len(self.metrics['episode_rewards'])), 
                          moving_avg, 'r-', label=f'Moving Avg ({window})')
            axes[0, 0].legend()
        
        # 2. Expert Agreement
        axes[0, 1].plot(self.metrics['expert_agreements'], 'g-', alpha=0.7)
        axes[0, 1].set_title('Expert Agreement Over Time')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Agreement Ratio')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(0, 1)
        
        # 3. Training Loss per Iteration
        if self.metrics['training_losses']:
            iterations = range(1, len(self.metrics['training_losses']) + 1)
            axes[1, 0].plot(iterations, self.metrics['training_losses'], 'purple', marker='o')
            axes[1, 0].set_title('Training Loss per Iteration')
            axes[1, 0].set_xlabel('Iteration')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Episode Lengths
        axes[1, 1].plot(self.metrics['episode_lengths'], 'orange', alpha=0.7)
        axes[1, 1].set_title('Episode Lengths Over Time')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Steps')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        obs_suffix = f"_{self.config.observation_type}" if self.config.observation_type else ""
        dropout_suffix = f"_dropout{self.config.dropout_rate}"
        plot_path = os.path.join(self.plots_dir, f'training_results_{timestamp}{obs_suffix}{dropout_suffix}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f'Training plots saved to: {plot_path}')
        
        return
