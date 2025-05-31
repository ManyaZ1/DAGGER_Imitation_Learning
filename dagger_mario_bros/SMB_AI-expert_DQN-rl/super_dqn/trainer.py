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

# Κλάση που διαχειρίζεται την εκπαίδευση και αξιολόγηση του Mario agent!
class MarioTrainer:
    '''Διαχειριστής εκπαίδευσης για τον Agent Mario AI'''

    def __init__(self,
                 world:       str = '1',
                 stage:       str = '1',
                 action_type: str = 'simple') -> None:
        # Δημιουργία περιβάλλοντος Gym για συγκεκριμένο επίπεδο
        env_name = f'SuperMarioBros-{world}-{stage}-v0'
        self.env = gym_super_mario_bros.make(env_name)
        
        # Action space (οι ενέργειες που μπορεί να εκτελέσει ο Mario!)
        if action_type == 'simple':
            actions = SIMPLE_MOVEMENT
        elif action_type == 'complex':
            actions = COMPLEX_MOVEMENT
        else:
            actions = RIGHT_ONLY
        
        # Εφαρμογή wrapper NES για περιορισμό ενεργειών!
        # Αναλυτικότερα, ο Mario έχει αρχικά 1 binary action για κάθε
        # κουμπί του NES controller! Δηλαδή, υπάρχουν 6 κουμπιά [D-Pad & A, B]
        # τα οποία μπορούν να πατηθούν ή όχι, δίνοντας 2^6 = 64 actions!
        # Επομένως, περνάμε μία λίστα από έγκυρους, προκαθορισμένους συνδυασμούς
        # ενεργειών, με αποτέλεσμα την μείωση του action space.
        self.env = JoypadSpace(self.env, actions)

        # Προσθήκη του custom wrapper μας για preprocessing (resize, grayscale, stack)
        self.env = MarioPreprocessor(self.env)
        
        # Δημιουργία agent με βάση τα χαρακτηριστικά του περιβάλλοντος
        state_shape = self.env.observation_space.shape
        n_actions   = self.env.action_space.n
        self.agent  = MarioAgent(state_shape, n_actions)
        
        print(f'Environment:       {env_name}')
        print(f'State shape:       {state_shape}')
        print(f'Number of actions: {n_actions}')
        print(f'Actions:           {actions}')

        return
    
    def train(self,
              episodes:  int = 1000,
              save_freq: int = 100,
              render:    bool = False) -> None:
        '''Εκπαίδευση του Mario agent'''
        print(f'Έναρξη εκπαίδευσης για {episodes} επεισόδια...')
        
        for episode in range(episodes):
            # Επαναφορά περιβάλλοντος και αρχικής κατάστασης
            state = self.env.reset()
            
            total_reward = 0
            steps        = 0
            done         = False
            
            while not done:
                if render:
                    self.env.render()
                
                # Επιλογή action μέσω πολιτικής του agent
                action = self.agent.act(state)

                # Εκτέλεση action στο περιβάλλον
                next_state, reward, done, info = self.env.step(action)
                
                reward = self.shape_reward(reward, info, done)
                
                # Αποθήκευση 'εμπειρίας' και μετάβαση στην επόμενη κατάσταση
                self.agent.remember(state, action, reward, next_state, done)
                state         = next_state
                total_reward += reward
                steps        += 1
                
                # Εκπαίδευση με εμπειρίες (Replay)
                self.agent.replay()
                
                if done:
                    break
            
            # Καταγραφή score για το επεισόδιο
            self.agent.scores.append(total_reward)
            avg_score = np.mean(self.agent.scores[-100:])
            self.agent.avg_scores.append(avg_score)
            
            if episode % 10 == 0:
                print(f'Episode {episode}/{episodes}')
                print(f'Score: {total_reward:.2f}, Avg Score: {avg_score:.2f}')
                print(f'Epsilon: {self.agent.epsilon:.4f}, Steps: {steps}')
                print(f'World Position: {info.get('x_pos', 0)}')
                print('-' * 50)
            
            # Αποθήκευση μοντέλου ανά save_freq επεισόδια
            if episode % save_freq == 0 and episode > 0:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                save_path = f'models/mario_model_ep{episode}_{timestamp}.pth'
                os.makedirs('models', exist_ok = True)
                self.agent.save_model(save_path)
        
        # Final save
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        final_save_path = f'models/mario_model_final_{timestamp}.pth'
        os.makedirs('models', exist_ok=True)
        self.agent.save_model(final_save_path)
        
        # Ζωγραφικηηηηηή!
        self._plot_training_progress()
        print('Η εκπαίδευση ολοκληρώθηκε!')

        return
    
    def shape_reward(self,
                     reward: float,
                     info:   dict,
                     done:   bool) -> float:
        '''Custom reward διαμόρφωση για καλύτερη εκπαίδευση'''
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
    
    def test(self,
             model_path: str,
             episodes: int = 5,
             render: bool = True) -> list:
        '''Δοκιμή μοντέλου'''
        self.agent.load_model(model_path)
        self.agent.epsilon = 0  # Όχι exploration κατά το testing!
        
        print(f'Δοκιμή μοντέλου για {episodes} επεισόδια...')
        test_scores = []
        
        for episode in range(episodes):
            state = self.env.reset()
            
            total_reward = 0
            steps        = 0
            done         = False
            
            while not done:
                if render:
                    self.env.render()
                
                # Action βάσει του εκπαιδευμένου μοντέλου (ΧΩΡΙΣ RANDOM)!
                action = self.agent.act(state, training=False)
                next_state, reward, done, info = self.env.step(action)
                
                state         = next_state
                total_reward += reward
                steps        += 1
                
                if done:
                    break
            
            test_scores.append(total_reward)
            print(f'Test Episode {episode + 1}: Score = {total_reward}, Steps = {steps}')
            print(f'Final Position: {info.get('x_pos', 0)}, Lives: {info.get('life', 3)}')
        
        avg_test_score = np.mean(test_scores)
        print(f'\nAverage Test Score: {avg_test_score:.2f}')
        
        self.env.close()

        return test_scores
    
    def _plot_training_progress(self) -> None:
        '''Γραφική για την πρόοδο της εκπαίδευσης'''
        if len(self.agent.scores) == 0:
            return
        
        plt.figure(figsize = (12, 4))
        
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
        plt.savefig(f'mario_training_progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png')
        plt.show()

        return
