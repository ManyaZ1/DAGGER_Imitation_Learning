import os
import sys
import numpy as np
from datetime import datetime
from typing import Dict, List
from dataclasses import dataclass
import json
from collections import deque

base_dir = os.path.dirname(__file__)
temp = os.path.abspath(
    os.path.join(base_dir, '..', 'expert-SMB_DQN', 'super_dqn')
)
sys.path.append(temp)
from agent import MarioAgent
from env_wrappers import MarioPreprocessor
from visual_utils import MarioRenderer
from dagger_agent import DaggerMarioAgent

import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

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
    world:                     str = '1'
    stage:                     str = '1'
    render:                    bool = False
    save_frequency:            int = 1
    max_episode_steps:         int = 4000
    noise_probability:         float = 0.2 # 0.1?

class DaggerTrainer:
    ''' DAGGER [Dataset Aggregation] trainer για τον Mario AI agent. '''
    
    def __init__(self, config: DaggerConfig):
        self.config = config
        self._setup_environment()
        self._setup_agents()
        self._setup_directories()
        self._setup_metrics()

        return
        
    def _setup_environment(self):
        env_name = f'SuperMarioBros-{self.config.world}-{self.config.stage}-v0'
        print(f'Αρχικοποίηση περιβάλλοντος: {env_name}')
        
        raw_env     = gym_super_mario_bros.make(env_name)
        wrapped_env = JoypadSpace(raw_env, SIMPLE_MOVEMENT)
        self.env    = MarioPreprocessor(wrapped_env)
        
        # Ορισμός max αριθμού βημάτων ανά επεισόδιο
        self.env._max_episode_steps = self.config.max_episode_steps
        
        self.state_shape = self.env.observation_space.shape
        self.n_actions   = self.env.action_space.n
        
        print(f'State shape: {self.state_shape} - Actions: {self.n_actions}')

        return
    
    def _setup_agents(self):
        ''' Αρχικοποίηση του expert και learner agent '''
        # Οι agent μας
        print('\nExpert:')
        self.expert  = MarioAgent(self.state_shape, self.n_actions)
        # Φόρτωση του expert μοντέλου
        self.expert.load_model(self.config.expert_model_path)

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
    
    def _shape_reward(self, reward: float, info: dict, done: bool) -> float:
        '''
        Custom reward διαμόρφωση για καλύτερη
        εκπαίδευση - copy/paste από trainer.py
        '''
        shaped_reward = reward
        
        # Θέλουμε να πάει προς τα δεξιά!
        x_pos = info.get('x_pos')
        if x_pos is not None:
            progress        = max(0, x_pos - self.prev_x_pos)
            shaped_reward  += progress * 0.1
            self.prev_x_pos = x_pos

        # Time-based penalty
        shaped_reward -= 0.1
        
        # Penalize death
        if done and info.get('life', 3) < 3:
            shaped_reward -= 10
        
        # Bonus για την ολοκλήρωση της πίστας
        if info.get('flag_get', False):
            shaped_reward += 100
        
        return shaped_reward
    def _run_episode(self, iteration: int, episode: int) -> Dict:
        ''' Τρέχει 1 επεισόδιο και συλλέγει δεδομένα '''
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

        done            = False
        total_reward    = 0
        step_count      = 0
        learner_actions = []
        expert_actions  = []

        while not done and step_count < self.config.max_episode_steps:
            if self.config.render:
                self.renderer.render()
            
            # --- Ενέργειες από τον learner και τον expert ---
            learner_action = self.learner.act(state)
            # Random actions during training from learner - NOISE INJECTION!
            if np.random.random() < self.config.noise_probability:
                learner_action = np.random.randint(self.n_actions)

            expert_action  = self.expert.act(state, training = False)
            
            # Διατήρηση των ενεργειών για agreement calculation!
            learner_actions.append(learner_action)
            expert_actions.append(expert_action)
            
            # Εκτέλεση της ενέργειας του learner στο περιβάλλον
            next_state, reward, done, info = self.env.step(learner_action)
            reward = self._shape_reward(reward, info, done)
            
            # Θυμήσου την αλληλεπίδραση expert-περιβάλλοντος, δηλαδή:
            # Expert provides correct labels for those states
            self.learner.remember(state, expert_action, reward, next_state, done)
            
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
        total_loss = 0.
        batch_count = 0
        
        for batch in range(num_batches):
            loss = self.learner.replay()
            if loss is not None:
                total_loss += loss
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
            flag_count = sum(1 for r in iteration_rewards if r > 3000)  # Approximate flag count
            print(f'  Flag Completions:  {flag_count}')

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

        return {
            'best_reward': self.best_reward,
            'best_model_path': best_model_path,
            'final_metrics': self.metrics,
            'total_episodes': len(self.metrics['episode_rewards'])
        }

def main():
    config = DaggerConfig(
        iterations                = 400, #800,200
        episodes_per_iter         = 20,
        training_batches_per_iter = 200,
        expert_model_path= os.path.join(
            base_dir, '..',
            'expert-SMB_DQN', 'models', 'ep30000_MARIO_EXPERT.pth'
        ),
        world                    = '1',
        stage                    = '1',
        render                   = False,
        save_frequency           = 25,
        max_episode_steps        = 800, # Στα 300 κάπου τερματίζεται η πίστα!
    )
    
    trainer = DaggerTrainer(config)
    trainer.train()

    return

if __name__ == '__main__':
    main()
