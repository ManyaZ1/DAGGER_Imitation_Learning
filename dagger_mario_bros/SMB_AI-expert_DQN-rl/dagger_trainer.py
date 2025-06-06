import os
import numpy as np
from datetime import datetime
from typing import Dict, List
from dataclasses import dataclass
import json
import matplotlib.pyplot as plt
from collections import deque

from super_dqn.agent import MarioAgent
from super_dqn.env_wrappers import MarioPreprocessor
from super_dqn.visual_utils import MarioRenderer
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

# Γιατί μας τα ζάλιζε ένα gym...
import warnings
warnings.filterwarnings('ignore', category = UserWarning, module = 'gym')

base_dir = os.path.dirname(__file__)

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
    early_stopping_threshold:  float = 0.95
    max_episode_steps:         int = 4000

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
        
        print(f'State shape: {self.state_shape}, Actions: {self.n_actions}')

        return
    
    def _setup_agents(self):
        ''' Αρχικοποίηση του expert και learner agent '''
        # Οι agent μας
        self.expert  = MarioAgent(self.state_shape, self.n_actions)
        self.learner = MarioAgent(self.state_shape, self.n_actions)
        
        # Φόρτωση του expert μοντέλου
        self.expert.load_model(self.config.expert_model_path)
        
        # Οπτικοποίηση αν ζητηθεί
        if self.config.render:
            self.renderer = MarioRenderer(self.env, scale = 3.)
                
        return
    
    def _setup_directories(self):
        ''' Δημιουργία των απαραίτητων dirs '''
        self.save_dir  = os.path.join(
            base_dir, 'models', 'dagger_checkpoints'
        )
        self.plots_dir = os.path.join(
            base_dir, 'models', 'dagger_checkpoints', 'plots'
        )

        os.makedirs(self.save_dir,  exist_ok = True)
        os.makedirs(self.plots_dir, exist_ok = True)

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
            return 0.;
        
        agreements = sum(1 for la, ea in zip(learner_actions, expert_actions) if la == ea)
        
        return agreements / len(learner_actions)
    
    def _run_episode(self, iteration: int, episode: int) -> Dict:
        ''' Τρέχει 1 επεισόδιο και συλλέγει δεδομένα '''
        state = self.env.reset()

        done            = False
        total_reward    = 0
        step_count      = 0
        learner_actions = []
        expert_actions  = []

        while not done and step_count < self.config.max_episode_steps:
            if self.config.render:
                self.renderer.render()
            
            # Ενέργειες από τον learner και τον expert
            learner_action = self.learner.act(state)
            expert_action  = self.expert.act(state, training = False)
            
            # Διατήρηση των ενεργειών για agreement calculation!
            learner_actions.append(learner_action)
            expert_actions.append(expert_action)
            
            # Εκτέλεση της ενέργειας του learner στο περιβάλλον
            next_state, reward, done, info = self.env.step(learner_action)
            
            # Θυμήσου την αλληλεπίδραση expert-περιβάλλοντος, δηλαδή:
            # Expert provides correct labels for those states
            self.learner.remember(state, expert_action, reward, next_state, done)
            
            state         = next_state
            total_reward += reward
            step_count   += 1
            
            if info.get('life', 2) < 2: # Mario died
                break
        
        # Υπολογισμός expert agreement
        agreement = self._calculate_expert_agreement(learner_actions, expert_actions)
        self.expert_agreement_window.append(agreement)
        
        episode_info = {
            'reward':      total_reward,
            'steps':       step_count,
            'agreement':   agreement,
            'final_x_pos': info.get('x_pos', 0)
        }
        
        print(
            f'Episode {episode + 1}: Reward={total_reward:.2f}, '
            f'Steps={step_count}, Agreement={agreement:.3f}, '
            f'X-pos={info.get("x_pos", 0)}'
        )
        
        return episode_info
    
    def _train_learner(self, iteration: int) -> float:
        ''' Εκπαίδευση του learner agent με τα δεδομένα που συγκεντρώθηκαν! '''
        total_loss  = 0.
        batch_count = 0
        
        print(f'Εκπαίδευση learner για επανάληψη {iteration+1}')
        for batch in range(self.config.training_batches_per_iter):
            loss = self.learner.replay()
            if loss is not None:
                total_loss  += loss
                batch_count += 1
                
        avg_loss = total_loss / max(batch_count, 1)
        print(f"Μέσος όρος loss εκπαίδευσης: {avg_loss:.6f}")

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

    def _check_early_stopping(self) -> bool:
        ''' Ελέγχει αν πληρούνται τα κριτήρια early stopping... '''
        if len(self.expert_agreement_window) < 50: # Φυσικά, πρέπει να υπάρχουν δεδομένα...
            return False
        
        recent_agreement = np.mean(list(self.expert_agreement_window)[-50:])
        
        if recent_agreement >= self.config.early_stopping_threshold:
            print(f"Early stopping! Agreement rate: {recent_agreement:.3f}")
            return True
        
        return False
    
    def train(self) -> Dict:
        '''Βασική συνάρτηση/loop εκπαίδευσης DAGGER.'''
        print('\nΕνάρξη εκπαίδευσης DAGGER για Mario AI agent...')

        best_model_path = None
        for iteration in range(self.config.iterations):
            print(f'\nIteration: {iteration+1}/{self.config.iterations}')

            iteration_rewards    = []
            iteration_agreements = []
            
            # Συλλογή επεισοδίων για την συγκεκριμένη επανάληψη
            for episode in range(self.config.episodes_per_iter):
                episode_info = self._run_episode(iteration, episode)
                
                # Στατιστικά
                iteration_rewards.append(episode_info['reward'])
                iteration_agreements.append(episode_info['agreement'])
                self.metrics['episode_rewards'].append(episode_info['reward'])
                self.metrics['expert_agreement'].append(episode_info['agreement'])
                self.metrics['episode_lengths'].append(episode_info['steps'])
            
            # Στατιστικά για την τρέχουσα επανάληψη
            avg_reward    = np.mean(iteration_rewards)
            avg_agreement = np.mean(iteration_agreements)
            
            self.metrics['iteration_rewards'].append(avg_reward)
            
            print(f"Iteration {iteration+1} Summary:")
            print(f"  Average Reward:     {avg_reward:.2f}")
            print(f"  Average Agreement:  {avg_agreement:.3f}")
            print(f"  Best Reward So Far: {max(iteration_rewards):.2f}")

            # Εκπαίδευση του learner agent με τα δεδομένα της επανάληψης
            avg_loss = self._train_learner(iteration)
            self.metrics['training_losses'].append(avg_loss)
            
            # Save checkpoint
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
            
            # Early stopping έλεγχος!
            if self._check_early_stopping():
                break
        
        # Τελική αναφορά αποτελεσμάτων
        print(f"Best Average Reward: {self.best_reward:.2f}")
        print(f"Final Expert Agreement: {np.mean(list(self.expert_agreement_window)[-10:]):.3f}")
        print(f"Best Model: {best_model_path}")

        return {
            'best_reward': self.best_reward,
            'best_model_path': best_model_path,
            'final_metrics': self.metrics,
            'total_episodes': len(self.metrics['episode_rewards'])
        }

def main():
    config = DaggerConfig(
        iterations                = 100,
        episodes_per_iter         = 3,
        training_batches_per_iter = 25,
        expert_model_path= os.path.join(
            base_dir, 'models', 'WORKING_MARIO_AGENT.pth'
        ),
        world                    = '1',
        stage                    = '1',
        render                   = False,
        save_frequency           = 25,
        early_stopping_threshold = 0.9,
        max_episode_steps        = 5000,
    )
    
    trainer = DaggerTrainer(config)
    results = trainer.train()
    
    print(f"\nTraining Results: {results}")

    return

if __name__ == '__main__':
    main()
