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

base_dir   = os.path.dirname(__file__)              
pkg_parent = os.path.abspath(os.path.join(base_dir, '..', 'expert-SMB_DQN'))
sys.path.insert(0, pkg_parent)   
super_dqn_path  = os.path.abspath(
    os.path.join(base_dir, '..', 'expert-SMB_DQN', 'super_dqn')
) # …/expert-SMB_DQN/super_dqn
sys.path.append(super_dqn_path) # add to PYTHONPATH
from agent import MarioAgent
from env_wrappers import MarioPreprocessor
from visual_utils import MarioRenderer
from trainer import MarioTrainer

from dagger_agent import DaggerMarioAgent

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
warnings.filterwarnings('ignore', category = UserWarning, module = 'gym')

@dataclass
class DaggerConfig:
    '''
    Configuration κλάση για τις παραμέτρους εκπαίδευσης DAGGER.
    Χρησιμοποιείται αυτή η τεχνική για αποφυγή global μεταβλητών!
    '''
    iterations:                int
    episodes_per_iter:         int
    training_batches_per_iter: int
    expert_model_path:         str
    observation_type:          Optional[str] = None # partial, noisy, downsampled...
    noise_level:               float = 0.1 # Χρησιμοποιείται για το noisy observation_type!
    world:                     str = '1'
    stage:                     str = '1'
    render:                    bool = False
    save_frequency:            int = 1
    max_episode_steps:         int = 1000
    only_for_testing:          bool = False # Όταν θέλουμε κυρίως να κάνουμε απλά testing!

class DaggerTrainer(MarioTrainer): # Κληρονομεί κυρίως για το test method!!!
    ''' DAGGER [Dataset Aggregation] trainer για τον Mario AI agent. '''
    
    def __init__(self, config: DaggerConfig):
        self.config = config

        self._setup_environment()
        
        if self.config.only_for_testing:
            print('\n-> DAGGER Trainer initialized in testing mode. No training will be performed.\n')
            self._handle_observation_type_settings()
            
            self.learner = DaggerMarioAgent(self.state_shape, self.n_actions)
            self.agent   = self.learner
            self.actions = SIMPLE_MOVEMENT
            
            return

        self._setup_agents()
        self._setup_directories()
        self._setup_metrics()

        return
    
    def _handle_observation_type_settings(self):
        if self.config.observation_type == 'partial':
            self.state_shape    = list(self.state_shape)
            self.state_shape[0] = 2
            self.state_shape    = tuple(self.state_shape)
            print(f'-> State shape for learner: {self.state_shape} - PARTIAL\n')

        self.observation_wrapper = None
        if self.config.observation_type:
            self.observation_wrapper = PartialObservationWrapper(
                obs_type    = self.config.observation_type,
                noise_level = self.config.noise_level
            )

        return
        
    def _setup_environment(self, printless: bool = False):
        env_name = f'SuperMarioBros-{self.config.world}-{self.config.stage}-v0'
        
        raw_env     = gym_super_mario_bros.make(env_name)
        wrapped_env = JoypadSpace(raw_env, SIMPLE_MOVEMENT)
        self.env    = MarioPreprocessor(wrapped_env)
        
        # Ορισμός max αριθμού βημάτων ανά επεισόδιο
        self.env = TimeLimit( # TimeLimit για memory safety!!!
            MarioPreprocessor(wrapped_env), max_episode_steps=self.config.max_episode_steps
        )
        
        self.state_shape = self.env.observation_space.shape
        self.n_actions   = self.env.action_space.n
        
        if not printless:
            print(f'Αρχικοποίηση περιβάλλοντος: {env_name}')
            print(f'State shape: {self.state_shape} - Actions: {self.n_actions}')

        return
    
    def _setup_agents(self):
        ''' Αρχικοποίηση του expert και learner agent '''
        # Οι agent μας
        print('\nExpert:')
        self.expert = MarioAgent(self.state_shape, self.n_actions)
        # Φόρτωση του expert μοντέλου
        self.expert.load_model(self.config.expert_model_path)

        # Διαχείριση observation type settings - Μέθοδος ώστε να
        # υπάρχει και όταν γίνεται χρήση μόνο για testing!
        self._handle_observation_type_settings()

        print('\nLearner/DAGGER:')
        self.learner = DaggerMarioAgent(self.state_shape, self.n_actions)

        # Οπτικοποίηση αν ζητηθεί
        if self.config.render:
            self.renderer = MarioRenderer(self.env, scale = 3.)
                
        return
    
    def _setup_directories(self):
        ''' Δημιουργία των απαραίτητων dirs '''
        self.save_dir  = os.path.join(
            base_dir, 'models_dagger'
        )
        os.makedirs(self.save_dir,  exist_ok = True)

        return
    
    def _setup_metrics(self):
        ''' Στατιστικά για σπασίκλες '''
        self.metrics = {
            'iteration_rewards': [],
            'episode_rewards':   [],
            'expert_agreement':  [],
            'training_losses':   [],
            'episode_lengths':   []
        }

        self.best_reward = float('-inf')

        # Μνημονικό παράθυρο για συμφωνία expert-learner!
        self.expert_agreement_window = deque(maxlen = 100)

        return
    
    def _calculate_expert_agreement(self,
                                    learner_actions: List[int],
                                    expert_actions:  List[int]) -> float:
        ''' Υπολογισμός του agreement rate μεταξύ των ενεργειών learner & expert '''
        if not learner_actions or not expert_actions:
            return 0.
        
        agreements = sum(1 for la, ea in zip(learner_actions, expert_actions) if la == ea)
        
        return agreements / len(learner_actions)
    
    def _run_episode(self, iteration: int, episode: int) -> Dict:
        ''' Τρέχει 1 επεισόδιο και συλλέγει δεδομένα '''

        # Fix για το access violation reading 0x000000000003C200...
        for attempt in range(3):
            try:
                state = self.env.reset()
                break
            except OSError:
                try:
                    self.env.close()
                except Exception:
                    pass
                gc.collect()
                time.sleep(1)
                self._setup_environment(printless = True) # Recreate the env from scratch!!!
        else:
            raise RuntimeError(f"[Episode {episode + 1:03}] env.reset() failed 3 times. Aborting.")

        self.prev_x_pos = 40 # Αρχική θέση Mario

        done            = False
        total_reward    = 0
        step_count      = 0
        learner_actions = []
        expert_actions  = []

        while not done and step_count < self.config.max_episode_steps:
            if self.config.render:
                self.renderer.render()
            


            # --- Ενέργειες από τον learner και τον expert ---
            # Apply wrapper only for learner
            observed_state = self.observation_wrapper.transform_observation(state) \
                             if self.observation_wrapper else state
            # Learner acts on the partial/noisy observation
            learner_action = self.learner.act(observed_state)
            # Expert acts on the full observation
            expert_action = self.expert.act(state, training = False)
            


            # Διατήρηση των ενεργειών για agreement calculation!
            learner_actions.append(learner_action)
            expert_actions.append(expert_action)
            
            # Εκτέλεση της ενέργειας του learner στο περιβάλλον
            next_state, reward, done, info = self.env.step(learner_action)
            reward = self.shape_reward(reward, info, done)
            
            # Θυμήσου την αλληλεπίδραση expert-περιβάλλοντος, δηλαδή:
            # Expert provides correct labels for those states
            self.learner.remember(observed_state, expert_action)
            
            state         = next_state
            total_reward += reward
            step_count   += 1

            if done or info.get('life', 2) < 2: # Τέλος επεισοδίου!
                break
        
        # Υπολογισμός expert agreement
        agreement = self._calculate_expert_agreement(learner_actions, expert_actions)
        self.expert_agreement_window.append(agreement)
        
        episode_info = {
            'reward':      total_reward,
            'steps':       step_count,
            'agreement':   agreement,
            'final_x_pos': info.get('x_pos', 0),
            'flag_get':    info.get('flag_get', False)
        }
        
        print(f'[Episode {episode + 1:03}] '
              f'Reward: {total_reward:8.2f} | '
              f'Steps: {step_count:4} | '
              f'Agreement: {agreement:.3f} | '
              f'X-pos: {info.get("x_pos", 0):4}'
        )
        
        return episode_info
    
    def _train_learner(self, iteration: int) -> float:
        ''' Εκπαίδευση του learner agent με τα δεδομένα που συγκεντρώθηκαν! '''
        total_loss  = 0.
        batch_count = 0
        
        for batch in range(self.config.training_batches_per_iter):
            loss = self.learner.replay()
            if loss is not None:
                total_loss  += loss
                batch_count += 1
                
        avg_loss = total_loss / max(batch_count, 1)
        print(f'Μέσος όρος loss εκπαίδευσης: {avg_loss:.6f}')

        return avg_loss
    
    def _train_learner_immediate(self, num_batches: int = 3) -> float:
        """Train learner immediately with specified number of batches."""
        total_loss  = 0.
        batch_count = 0
        
        for batch in range(num_batches):
            loss = self.learner.replay()
            if loss is not None:
                total_loss  += loss
                batch_count += 1
        
        avg_loss = total_loss / max(batch_count, 1)
        return avg_loss
    
    def _save_checkpoint(self, iteration: int, metrics: Dict):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save model
        model_path = os.path.join(
            self.save_dir, f'dagger_mario_iter{iteration+1}_{timestamp}.pth'
        )
        self.learner.save_model(str(model_path))
        
        # Save στατιστικά
        metrics_path = os.path.join(
            self.save_dir, f'metrics_iter{iteration+1}_{timestamp}.json'
        )
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent = 2)
        
        print(f'Checkpoint saved: {model_path}')
        print(f'Metrics saved: {metrics_path}')

        return str(model_path)
    
    def train(self) -> Dict:
        '''Βασική συνάρτηση/loop εκπαίδευσης DAGGER.'''
        print('-> Ενάρξη εκπαίδευσης DAGGER για Mario AI agent...')

        best_model_path = None
        for iteration in range(self.config.iterations):
            print(f'\nIteration: {iteration+1}/{self.config.iterations}')

            iteration_rewards    = []
            iteration_agreements = []
            
            # Run episodes one by one and immediately save flag completions
            for episode in range(self.config.episodes_per_iter):
                episode_info = self._run_episode(iteration, episode)
                
                # Στατιστικά
                reward_temp = episode_info['reward']
                iteration_rewards.append(reward_temp)
                iteration_agreements.append(episode_info['agreement'])
                self.metrics['episode_rewards'].append(reward_temp)
                self.metrics['expert_agreement'].append(episode_info['agreement'])
                self.metrics['episode_lengths'].append(episode_info['steps'])

                # IMMEDIATE FLAG SAVE - Right after episode completion
                if episode_info['flag_get']:    
                    success = self.test(
                        test_agent = self.learner, render = False, episodes = 3,
                        observation_wrapper = self.observation_wrapper,
                        env_unresponsive = False
                    )
                    if not success:
                        continue

                    print(f'* FLAG CAPTURED in episode {episode+1}! Score: {reward_temp:.2f}')
                    
                    # Train immediately with current memory
                    print(f'Training learner immediately with '
                        f'{len(getattr(self.learner, "dagger_memory", self.learner.dagger_memory))} experiences...')
                    immediate_loss = self._train_learner_immediate()
                    
                    # Save the model that just learned from the successful episode
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    temp = self.config.observation_type if self.config.observation_type is not None else 'normal'
                    flag_model_path = os.path.join(
                        self.save_dir,
                        f'mario_FLAG_iter{iteration+1}_ep{episode+1}_{int(reward_temp)}_{timestamp}_{temp}.pth'
                    )
                    self.learner.save_model(flag_model_path)
                    print(f'-> FLAG MODEL SAVED IMMEDIATELY: {flag_model_path}')
                    print(f'   Score: {reward_temp:.2f}, Loss: {immediate_loss:.6f}')
            
            # Regular training after all episodes (additional training)
            print(f'Final training with {len(self.learner.dagger_memory)} experiences...')
            avg_loss = self._train_learner(iteration)
            self.metrics['training_losses'].append(avg_loss)
            
            # Iteration summary
            avg_reward    = np.mean(iteration_rewards)
            avg_agreement = np.mean(iteration_agreements)
            
            self.metrics['iteration_rewards'].append(avg_reward)
            
            print(f'Iteration {iteration+1} Summary:')
            print(f'  Average Reward:    {avg_reward:.2f}')
            print(f'  Average Agreement: {avg_agreement:.3f}')
            print(f'  Best Iter Reward:  {max(iteration_rewards):.2f}')
            print(f'  Final Training Loss: {avg_loss:.6f}')

            # Regular checkpoint saves
            if (iteration + 1) % self.config.save_frequency == 0:
                checkpoint_path = self._save_checkpoint(iteration, {
                    'iteration':     iteration + 1,
                    'avg_reward':    avg_reward,
                    'avg_agreement': avg_agreement,
                    'avg_loss':      avg_loss
                })
                
                if avg_reward > self.best_reward:
                    self.best_reward = avg_reward
                    best_model_path  = checkpoint_path
        
        # Final results
        print(f'Best Average Reward: {self.best_reward:.2f}')
        print(f'Final Expert Agreement: {np.mean(list(self.expert_agreement_window)[-10:]):.3f}')
        print(f'Best Model: {best_model_path}')
        
        # Generate comprehensive plots
        print('\n-> Generating training plots...')
        plot_paths = self._plot_training_results()
        print(f'All plots and summary saved in: {os.path.join(base_dir, "plots")}')

        return {
            'best_reward': self.best_reward,
            'best_model_path': best_model_path,
            'final_metrics': self.metrics,
            'total_episodes': len(self.metrics['episode_rewards']),
            'plot_paths': plot_paths
        }
    
    def _plot_training_results(self):
        """Create comprehensive plots of training results and save to plots directory."""
        # Create plots directory
        plots_dir = os.path.join(base_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create a large figure with multiple subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('DAGGER Training Results - Mario AI Agent', fontsize=16, fontweight='bold')
        
        # 1. Episode Rewards over Time
        axes[0, 0].plot(self.metrics['episode_rewards'], alpha=0.6, linewidth=0.8)
        # Add moving average
        if len(self.metrics['episode_rewards']) > 10:
            window_size = min(50, len(self.metrics['episode_rewards']) // 10)
            moving_avg = np.convolve(self.metrics['episode_rewards'], 
                                   np.ones(window_size)/window_size, mode='valid')
            axes[0, 0].plot(range(window_size-1, len(self.metrics['episode_rewards'])), 
                          moving_avg, color='red', linewidth=2, label=f'Moving Avg ({window_size})')
            axes[0, 0].legend()
        axes[0, 0].set_title('Episode Rewards Over Time')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Iteration Average Rewards
        axes[0, 1].plot(self.metrics['iteration_rewards'], marker='o', linewidth=2, markersize=4)
        axes[0, 1].set_title('Average Reward per Iteration')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Average Reward')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Expert Agreement over Time
        axes[0, 2].plot(self.metrics['expert_agreement'], alpha=0.6, linewidth=0.8)
        # Add moving average for agreement
        if len(self.metrics['expert_agreement']) > 10:
            window_size = min(50, len(self.metrics['expert_agreement']) // 10)
            moving_avg = np.convolve(self.metrics['expert_agreement'], 
                                   np.ones(window_size)/window_size, mode='valid')
            axes[0, 2].plot(range(window_size-1, len(self.metrics['expert_agreement'])), 
                          moving_avg, color='red', linewidth=2, label=f'Moving Avg ({window_size})')
            axes[0, 2].legend()
        axes[0, 2].set_title('Expert Agreement Over Time')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Agreement Rate')
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].set_ylim(0, 1)
        
        # 4. Training Loss over Iterations
        axes[1, 0].plot(self.metrics['training_losses'], marker='s', linewidth=2, markersize=4, color='orange')
        axes[1, 0].set_title('Training Loss per Iteration')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Average Loss')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_yscale('log')  # Log scale for better visualization
        
        # 5. Episode Lengths Distribution
        axes[1, 1].hist(self.metrics['episode_lengths'], bins=30, alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(np.mean(self.metrics['episode_lengths']), color='red', linestyle='--', 
                         linewidth=2, label=f'Mean: {np.mean(self.metrics["episode_lengths"]):.1f}')
        axes[1, 1].set_title('Episode Length Distribution')
        axes[1, 1].set_xlabel('Episode Length (Steps)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Reward vs Agreement Scatter
        axes[1, 2].scatter(self.metrics['expert_agreement'], self.metrics['episode_rewards'], 
                         alpha=0.6, s=20)
        axes[1, 2].set_title('Reward vs Expert Agreement')
        axes[1, 2].set_xlabel('Expert Agreement')
        axes[1, 2].set_ylabel('Episode Reward')
        axes[1, 2].grid(True, alpha=0.3)
        
        # Add correlation coefficient
        if len(self.metrics['expert_agreement']) > 1:
            corr = np.corrcoef(self.metrics['expert_agreement'], self.metrics['episode_rewards'])[0, 1]
            axes[1, 2].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                          transform=axes[1, 2].transAxes, fontsize=10,
                          bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        
        # Save the main plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_path = os.path.join(plots_dir, f'dagger_training_results_{timestamp}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f'Main training plots saved: {plot_path}')
        
        # Create a separate figure for detailed reward analysis
        plt.figure(figsize=(15, 10))
        
        # Subplot 3: Performance improvement over iterations
        plt.subplot(2, 2, 3)
        if len(self.metrics['iteration_rewards']) > 1:
            plt.plot(self.metrics['iteration_rewards'], marker='o', linewidth=2)
            plt.fill_between(range(len(self.metrics['iteration_rewards'])), 
                           self.metrics['iteration_rewards'], alpha=0.3)
        plt.title('Performance Improvement Over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Average Reward')
        plt.grid(True, alpha=0.3)
        
        # Subplot 4: Training efficiency (reward vs loss)
        plt.subplot(2, 2, 4)
        if len(self.metrics['training_losses']) > 0 and len(self.metrics['iteration_rewards']) > 0:
            # Normalize both metrics for comparison
            norm_rewards = np.array(self.metrics['iteration_rewards']) / max(self.metrics['iteration_rewards'])
            norm_losses = np.array(self.metrics['training_losses']) / max(self.metrics['training_losses'])
            
            plt.plot(norm_rewards, label='Normalized Rewards', linewidth=2)
            plt.plot(norm_losses, label='Normalized Loss', linewidth=2)
            plt.title('Training Efficiency: Rewards vs Loss')
            plt.xlabel('Iteration')
            plt.ylabel('Normalized Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the detailed analysis plot
        detailed_plot_path = os.path.join(plots_dir, f'dagger_detailed_analysis_{timestamp}.png')
        plt.savefig(detailed_plot_path, dpi=300, bbox_inches='tight')
        print(f'Detailed analysis plots saved: {detailed_plot_path}')
        
        # Create a summary statistics text file
        stats_path = os.path.join(plots_dir, f'training_summary_{timestamp}.txt')
        with open(stats_path, 'w') as f:
            f.write("DAGGER Training Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total Episodes: {len(self.metrics['episode_rewards'])}\n")
            f.write(f"Total Iterations: {len(self.metrics['iteration_rewards'])}\n")
            f.write(f"Best Episode Reward: {max(self.metrics['episode_rewards']):.2f}\n")
            f.write(f"Average Episode Reward: {np.mean(self.metrics['episode_rewards']):.2f}\n")
            f.write(f"Final Iteration Reward: {self.metrics['iteration_rewards'][-1]:.2f}\n")
            f.write(f"Average Expert Agreement: {np.mean(self.metrics['expert_agreement']):.3f}\n")
            f.write(f"Final Expert Agreement: {np.mean(self.metrics['expert_agreement'][-10:]):.3f}\n")
            f.write(f"Average Training Loss: {np.mean(self.metrics['training_losses']):.6f}\n")
            f.write(f"Final Training Loss: {self.metrics['training_losses'][-1]:.6f}\n")
            f.write(f"Average Episode Length: {np.mean(self.metrics['episode_lengths']):.1f}\n")
            
        print(f'Training summary saved: {stats_path}')
        
        # Close all figures to free memory
        plt.close('all')
        
        return plot_path, detailed_plot_path, stats_path

def main():
    config = DaggerConfig(
        observation_type          = 'noisy',  # partial, noisy, downsampled...
        noise_level               = 0.1,      # Χρησιμοποιείται μόνο για noisy observation_type!
        iterations                = 300,
        episodes_per_iter         = 3,
        training_batches_per_iter = 300,
        expert_model_path         = os.path.join(
            base_dir, '..',
            'expert-SMB_DQN', 'models', 'ep30000_MARIO_EXPERT.pth'
        ),
        world                    = '1',
        stage                    = '1',
        render                   = False,
        save_frequency           = 400,
        max_episode_steps        = 600, # Στα 300 κάπου τερματίζεται η πίστα!
    )
    
    trainer = DaggerTrainer(config)
    trainer.train()

    return

if __name__ == '__main__':
    main()
