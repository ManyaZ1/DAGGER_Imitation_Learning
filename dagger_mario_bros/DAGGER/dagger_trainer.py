import os
import sys
import json
import numpy as np
from datetime import datetime
from typing import Dict, List
from dataclasses import dataclass
from collections import deque

base_dir = os.path.dirname(__file__)

# ----- Append paths for Mario dependencies -----
lib_path = os.path.abspath(os.path.join(base_dir, '..', 'expert-SMB_DQN', 'super_dqn'))
sys.path.append(lib_path)

from agent import MarioAgent
from env_wrappers import MarioPreprocessor
from visual_utils import MarioRenderer
from trainer import MarioTrainer
from dagger_agent import DaggerMarioAgent

import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='gym')

@dataclass
class DaggerConfig:
    """Base configuration class (kept for compatibility)."""
    iterations: int
    episodes_per_iter: int
    training_batches_per_iter: int
    expert_model_path:         str
    world:                     str = '1'
    stage:                     str = '1'
    render:                    bool = False
    save_frequency:            int = 1
    max_episode_steps:         int = 4000
    noise_probability:         float = 0.2 # 0.1?
      
class TrueDaggerConfig(DaggerConfig):
    """Configuration for TRUE DAGGER (no mixed policy, pure learner execution)."""
    # Nothing extra for now – we just alias for clarity.
    pass

class DaggerTrainer:
    """TRUE DAGGER trainer for Mario AI agent."""

    # ------------------------------------------------------------------ INIT
    def __init__(self, config: DaggerConfig):
        self.config = config

        self._setup_environment()
        self._setup_agents()
        self._setup_directories()
        self._setup_metrics()

    # ---------------------------------------------------------- ENVIRONMENT
    def _setup_environment(self):
        env_name = f"SuperMarioBros-{self.config.world}-{self.config.stage}-v0"
        print(f"Initialising environment: {env_name}")

        raw_env = gym_super_mario_bros.make(env_name)
        wrapped_env = JoypadSpace(raw_env, SIMPLE_MOVEMENT)
        self.env = MarioPreprocessor(wrapped_env)

        # Cap episode length
        self.env._max_episode_steps = self.config.max_episode_steps

        self.state_shape = self.env.observation_space.shape
        self.n_actions = self.env.action_space.n

        print(f"State shape: {self.state_shape} | Actions: {self.n_actions}")

    # -------------------------------------------------------------- AGENTS
    def _setup_agents(self):
        print("\nExpert:")
        self.expert = MarioAgent(self.state_shape, self.n_actions)
        self.expert.load_model(self.config.expert_model_path)

        print("\nLearner (DAGGER):")
        self.learner = DaggerMarioAgent(self.state_shape, self.n_actions)

        if self.config.render:
            self.renderer = MarioRenderer(self.env, scale=3.0)

    # ----------------------------------------------------------- DIRECTORIES
    def _setup_directories(self):
        self.save_dir = os.path.join(base_dir, 'models_dagger')
        os.makedirs(self.save_dir, exist_ok=True)

    # --------------------------------------------------------------- METRICS
    def _setup_metrics(self):
        self.metrics = {
            'iteration_rewards': [],
            'training_losses': [],
        }

        # Dataset storage per iteration
        self.iteration_datasets: Dict[int, List[Dict]] = {}

    # ------------------------------------------------------ REWARD SHAPING
    def _shape_reward(self, reward: float, info: dict, done: bool) -> float:
        shaped_reward = reward

        # Encourage rightward progress
        x_pos = info.get('x_pos')
        if x_pos is not None:
            progress = max(0, x_pos - self.prev_x_pos)
            shaped_reward += progress * 0.1
            self.prev_x_pos = x_pos

        # Small living penalty
        shaped_reward -= 0.1

        # Penalise death
        if done and info.get('life', 3) < 3:
            shaped_reward -= 10

        # Bonus for flag
        if info.get('flag_get', False):
            shaped_reward += 100

        return shaped_reward

    # --- DATASET HELPERS
    def _add_to_aggregated_dataset(self, episode_data: List[Dict], iteration: int):
        """Store episode data and rebuild aggregated memory."""
        self.iteration_datasets.setdefault(iteration, []).extend(episode_data)
        self._update_aggregated_training_set()

    def _update_aggregated_training_set(self):
        """Combine all data from every iteration into learner memory."""
        self.learner.dagger_memory.clear()

        aggregated = []
        for data_list in self.iteration_datasets.values():
            aggregated.extend(data_list)

        for dp in aggregated:
            self.learner.remember(
                dp['state'],
                dp['expert_action'],
                dp['reward'],
                dp['next_state'],
                dp['done']
            )

        print(f"Aggregated dataset size: {len(aggregated)} transitions")

    # --- RUN EPISODE
    def _run_episode(self, iteration: int, episode: int) -> Dict:
        """True DAGGER episode: learner acts, expert labels."""
        state = self.env.reset()
        self.prev_x_pos = 40  # Reset position
        max_x = 0             # Track max x_pos reached
        x_positions = []      # Track x_pos progression for validation

        done = False
        total_reward = 0
        step_count = 0
        learner_actions = []
        expert_actions = []

        while not done and step_count < self.config.max_episode_steps:
            if self.config.render:
                self.renderer.render()

            learner_action = self.learner.act(state)

            # Optional noise injection
            if np.random.random() < self.config.noise_probability:
                learner_action = np.random.randint(self.n_actions)

            expert_action = self.expert.act(state, training=False)

            learner_actions.append(learner_action)
            expert_actions.append(expert_action)

            next_state, reward, done, info = self.env.step(learner_action)
            reward = self._shape_reward(reward, info, done)

            # Track x position progression
            x_pos = info.get("x_pos", 0)
            max_x = max(max_x, x_pos)
            x_positions.append(x_pos)

            self.learner.remember(state, expert_action, reward, next_state, done)

            state = next_state
            total_reward += reward

            # Early stop if Mario dies
            if done or info.get("life", 2) < 2:
                break

        # Final expert agreement
        agreement = self._calculate_expert_agreement(learner_actions, expert_actions)
        self.expert_agreement_window.append(agreement)

        # ROBUST FLAG DETECTION - Multiple criteria must be met
        flag_captured = self._detect_flag_completion(max_x, x_positions, info, total_reward, done)
        
        # Debug information for flag area
        if max_x > 3100 or info.get('flag_get', False):
            print(f"DEBUG FLAG AREA:")
            print(f"  Max X: {max_x}")
            print(f"  Environment flag_get: {info.get('flag_get', False)}")
            print(f"  Total reward: {total_reward}")
            print(f"  Done: {done}, Life: {info.get('life', 2)}")
            print(f"  Last 10 X positions: {x_positions[-10:] if len(x_positions) >= 10 else x_positions}")
            print(f"  FINAL FLAG DECISION: {flag_captured}")

        episode_info = {
            'reward': total_reward,
            'steps': step_count,
            'agreement': agreement,
            'final_x_pos': max_x,
            'flag_get': flag_captured
        }

        flag_symbol = "🏁" if flag_captured else "  "
        print(f'[Episode {episode + 1:03}] {flag_symbol} '
            f'Reward: {total_reward:8.2f} | '
            f'Steps: {step_count:4} | '
            f'Agreement: {agreement:.3f} | '
            f'Max X: {max_x:4} | '
            f'Flag: {flag_captured}')

        return episode_info

    def _detect_flag_completion(self, max_x: int, x_positions: List[int], 
                            info: dict, total_reward: float, done: bool) -> bool:
        """
        Robust flag detection using multiple criteria.
        A flag is only considered captured if MULTIPLE conditions are met.
        """
        
        # Criterion 1: X position threshold (conservative)
        x_threshold_met = max_x >= 3161
        
        # Criterion 2: Reward threshold (flag completion gives big reward boost)
        # Adjust this based on your reward shaping - flag completion should give significant reward
        reward_threshold_met = total_reward >= 3000  # Adjust based on your typical successful runs
        
        # Criterion 3: X position progression (Mario should have progressed significantly)
        progression_valid = len(x_positions) > 50 and max_x > 3000  # Must have made significant progress
        
        # Criterion 4: Episode ended naturally (not due to death/timeout)
        natural_completion = done and info.get('life', 2) >= 2  # Didn't die
        # Criterion 5
        environmentflag= info.get('flag_get', False)  # Check if environment flag is set
        
        # Criterion six (΄έχει χαλάσει το πληκτρολόγιο μου...): X position stability near flag (Mario should stay near flag area when captured)
        x_stability = False
        if len(x_positions) >= 20:
            recent_x = x_positions[-20:]  # Last 20 positions
            high_x_count = sum(1 for x in recent_x if x >= 3100)
            x_stability = high_x_count >= 10  # At least half of recent positions in flag area
        
        # ALL criteria for high confidence flag detection
        strict_flag = (x_threshold_met and reward_threshold_met and 
                    progression_valid and natural_completion and environmentflag) # and x_stability) απορριφθηκε σαν εγκυρο κριτηριο
        
        # RELAXED criteria (at least 3 out of 5 must be true)
        criteria_met = sum([
            x_threshold_met,
            reward_threshold_met, 
            progression_valid,
            natural_completion#,
            #x_stability
        ])
        
        relaxed_flag = criteria_met >= 3 and x_threshold_met  # X threshold is mandatory
        
        # Use strict criteria for now, can switch to relaxed if too conservative
        flag_decision = strict_flag
        
        # Debug the decision process
        if max_x > 3000:  # Only debug when close to flag area
            print(f"    FLAG CRITERIA CHECK:")
            print(f"      X threshold (>= 3161): {x_threshold_met} (max_x: {max_x})")
            print(f"      Reward threshold (>= 2500): {reward_threshold_met} (reward: {total_reward:.1f})")
            print(f"      Progression valid: {progression_valid}")
            print(f"      Natural completion: {natural_completion}")
            print(f"      X stability: {x_stability}")
            print(f"      Criteria met: {criteria_met}/5")
            print(f"      STRICT FLAG: {strict_flag}")
            print(f"      RELAXED FLAG: {relaxed_flag}")
        
        return flag_decision
    def _run_episode2(self, iteration: int, episode: int) -> Dict:
        ''' Τρέχει 1 επεισόδιο και συλλέγει δεδομένα '''
        state = self.env.reset()
        self.prev_x_pos = 40  # Reset position
        max_x = 0             # NEW: track max x_pos reached

        done = False
        total_reward = 0
        step_count = 0
        learner_actions = []
        expert_actions = []

        while not done and step_count < self.config.max_episode_steps:
            if self.config.render:
                self.renderer.render()

            learner_action = self.learner.act(state)

            # Optional noise injection
            if np.random.random() < self.config.noise_probability:
                learner_action = np.random.randint(self.n_actions)

            expert_action = self.expert.act(state, training=False)

            learner_actions.append(learner_action)
            expert_actions.append(expert_action)

            next_state, reward, done, info = self.env.step(learner_action)
            reward = self._shape_reward(reward, info, done)

            # Track max x position
            x_pos = info.get("x_pos", 0)
            max_x = max(max_x, x_pos)

            self.learner.remember(state, expert_action, reward, next_state, done)

            state = next_state
            total_reward += reward
            step_count += 1

            # Early stop if Mario dies
            if done or info.get("life", 2) < 2:
                break

        # Final expert agreement
        agreement = self._calculate_expert_agreement(learner_actions, expert_actions)
        self.expert_agreement_window.append(agreement)

        flag_captured = max_x >= 3150  # STRONG flag detection

        episode_info = {
            'reward': total_reward,
            'steps': step_count,
            'agreement': agreement,
            'final_x_pos': max_x,
            'flag_get': flag_captured
        }

        print(f'[Episode {episode + 1:03}] '
            f'Reward: {total_reward:8.2f} | '
            f'Steps: {step_count:4} | '
            f'Agreement: {agreement:.3f} | '
            f'Max X: {max_x:4} | '
            f'Flag: {flag_captured}')

        return episode_info
        
    def _run_episode1(self, iteration: int, episode: int) -> Dict:
        ''' Τρέχει 1 επεισόδιο και συλλέγει δεδομένα '''
        state = self.env.reset()
        self.prev_x_pos = 40 # Αρχική θέση Mario

        done = False
        total_reward = 0.0
        step_count = 0
        episode_data: List[Dict] = []

        while not done and step_count < self.config.max_episode_steps:
            if self.config.render:
                self.renderer.render()

            # Learner chooses action
            learner_action = self.learner.act(state)
            # Expert provides correct label
            expert_action = self.expert.act(state, training=False)

            # Execute learner's action
            next_state, reward, done, info = self.env.step(learner_action)
            reward = self._shape_reward(reward, info, done)

            # Store transition (state, expert label)
            episode_data.append({
                'state': state.copy(),
                'expert_action': expert_action,
                'reward': reward,
                'next_state': next_state.copy(),
                'done': done,
            })

            state = next_state
            total_reward += reward
            step_count += 1

            if done or info.get('life', 2) < 2:
                break

        # Add collected data to aggregated dataset
        self._add_to_aggregated_dataset(episode_data, iteration)

        temp = False # Τερμάτισε όντως;
        if info.get('flag_get', False):
            print('\n-> Testing model with MarioTrainer...\n')
            trainer = MarioTrainer()
            if trainer.test(
                render = False, test_agent = self.learner,
                env_unresponsive = False
            ):
                temp = True
            print(f'\n-> Model testing complete! {temp}\n')

        return {
            'reward':      total_reward,
            'steps':       step_count,
            'data_points': len(episode_data),
            'final_x_pos': info.get('x_pos', 0),
            'flag_get':    temp,
        }

    # ----------------------------------------------------------- TRAIN STEP
    def _train_learner(self, iteration: int) -> float:
        """Train learner using entire aggregated dataset."""
        dataset_size = len(self.learner.dagger_memory)
        if dataset_size == 0:
            return 0.0

        batches_needed = max(self.config.training_batches_per_iter, dataset_size // 32)

        total_loss, batch_count = 0.0, 0
        for _ in range(batches_needed):
            loss = self.learner.replay()
            if loss is not None:
                total_loss += loss
                batch_count += 1

        avg_loss = total_loss / max(batch_count, 1)
        print(f"Trained on {dataset_size} samples | Avg loss: {avg_loss:.6f}")
        return avg_loss

    # ---------------------------------------------------------------- TRAIN
    def train(self) -> Dict:
        """TRUE DAGGER training loop."""
        print("-> Starting TRUE DAGGER training...")
        best_reward = float('-inf')
        best_model_path = None

        for iteration in range(self.config.iterations):
            print(f"\n=== Iteration {iteration + 1}/{self.config.iterations} ===")
            iteration_rewards = []

            # 1. Collect data
            for episode in range(self.config.episodes_per_iter):
                ep_info = self._run_episode(iteration, episode)
                iteration_rewards.append(ep_info['reward'])

                print(f"[Ep {episode + 1:02}] Reward: {ep_info['reward']:7.2f} | "
                      f"Steps: {ep_info['steps']:4} | Data: {ep_info['data_points']:3}")
                
                if ep_info['flag_get']:
                    print('Ο Mario τερμάτισε!')
                    self._save_checkpoint(iteration, {
                        'iteration':    iteration + 1,
                        'avg_reward':   np.mean(iteration_rewards),
                        'dataset_size': len(self.learner.dagger_memory),
                        'avg_loss':     0.0, # Δεν έχει υπολογιστεί ακόμα!
                    })

                # Στατιστικά
                reward_temp = episode_info['reward']
                iteration_rewards.append(reward_temp)
                iteration_agreements.append(episode_info['agreement'])
                self.metrics['episode_rewards'].append(reward_temp)
                self.metrics['expert_agreement'].append(episode_info['agreement'])
                self.metrics['episode_lengths'].append(episode_info['steps'])
                if reward_temp > 3400: #save immediately!
                    print(f'-> Episode {episode+1} Reward: {reward_temp:.2f} - '
                          f'Agreement: {episode_info["agreement"]:.3f} - '
                          f'Steps: {episode_info["steps"]}')
                    # Save the model that just learned from the successful episode
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    flag_model_path = os.path.join(
                        self.save_dir,
                        f'mario_FLAG_iter{iteration+1}_ep{episode+1}_{int(reward_temp)}_{timestamp}.pth'
                    )
                    self.learner.save_model(flag_model_path)
                    
                
                # IMMEDIATE FLAG SAVE - Right after episode completion
                if episode_info['flag_get']:
                    print(f'* FLAG CAPTURED in episode {episode+1}! Score: {reward_temp:.2f}')
                    
                    # Train immediately with current memory
                    print(f'Training learner immediately with '
                          f'{len(getattr(self.learner, "dagger_memory", self.learner.memory))} experiences...')
                    immediate_loss = self._train_learner_immediate()
                    
                    # Save the model that just learned from the successful episode
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    flag_model_path = os.path.join(
                        self.save_dir,
                        f'mario_FLAG_iter{iteration+1}_ep{episode+1}_{int(reward_temp)}_{timestamp}.pth'
                    )
                    self.learner.save_model(flag_model_path)
                    print(f'-> FLAG MODEL SAVED IMMEDIATELY: {flag_model_path}')
                    print(f'   Score: {reward_temp:.2f}, Loss: {immediate_loss:.6f}')
            
            # Regular training after all episodes (additional training)
            print(f'Final training with {len(self.learner.memory)} experiences...')
            avg_loss = self._train_learner(iteration)
            avg_reward = np.mean(iteration_rewards)
            dataset_size = len(self.learner.dagger_memory)

            # 3. Metrics & logging
            self.metrics['iteration_rewards'].append(avg_reward)
            self.metrics['training_losses'].append(avg_loss)

            print("Iteration Summary:")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Dataset Size: {dataset_size}")
            print(f"  Avg Loss: {avg_loss:.6f}")

            # 4. Save checkpoints
            if (iteration + 1) % self.config.save_frequency == 0:
                cp_path = self._save_checkpoint(iteration, {
                    'iteration': iteration + 1,
                    'avg_reward': avg_reward,
                    'dataset_size': dataset_size,
                    'avg_loss': avg_loss,
                })

                if avg_reward > best_reward:
                    best_reward = avg_reward
                    best_model_path = cp_path

        print(f"Best Avg Reward: {best_reward:.2f}")
        print(f"Best Model: {best_model_path}")

        return {
            'best_reward': best_reward,
            'best_model_path': best_model_path,
            'metrics': self.metrics,
        }

    # --- SAVE CHECKPOINT
    def _save_checkpoint(self, iteration: int, metrics: Dict) -> str:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        model_path = os.path.join(
            self.save_dir,
            f'dagger_mario_iter{iteration + 1}_{timestamp}.pth'
        )
        self.learner.save_model(model_path)

        metrics_path = os.path.join(
            self.save_dir,
            f'metrics_iter{iteration + 1}_{timestamp}.json'
        )
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"Checkpoint saved → {model_path}")
        return model_path

def main():
    config = DaggerConfig(
        iterations                = 400, #800,200
        episodes_per_iter         = 20,
        training_batches_per_iter = 200,
        expert_model_path=os.path.join(
            base_dir,
            '..',
            'expert-SMB_DQN',
            'models',
            'ep30000_MARIO_EXPERT.pth',
        ),
        world             = '1',
        stage             = '1',
        render            = False,
        save_frequency    = 25,
        max_episode_steps = 800,
    )

    trainer = DaggerTrainer(config)
    trainer.train()

if __name__ == '__main__':
    main()
