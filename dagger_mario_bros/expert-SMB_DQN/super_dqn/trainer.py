import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

import os
import cv2
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from agent import MarioAgent
from env_wrappers import MarioPreprocessor
from visual_utils import MarioRenderer, NESControllerOverlay

# Κλάση που διαχειρίζεται την εκπαίδευση και αξιολόγηση του Mario agent!
class MarioTrainer:
    '''Διαχειριστής εκπαίδευσης για τον Agent Mario AI'''

    def __init__(self,
                 world:       str = '1',
                 stage:       str = '1',
                 action_type: str = 'simple',
                 max_steps:   int = 800,
                 printless:   bool = False) -> None:
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
        self.actions = actions
        
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
        self.agent  = MarioAgent(state_shape, n_actions, printless = True)

        if not printless:
            print(f'Environment:       {env_name}')
            print(f'State shape:       {state_shape}')
            print(f'Number of actions: {n_actions}')
            print(f'Actions:           {actions}')

        self.best_avg_score    = float('-inf') # Track best average score
        self.max_episode_steps = max_steps # Μέγιστο πλήθος βημάτων ανά επεισόδιο

        return
    
    def train(self,
              episodes:  int = 1000,
              save_freq: int = 100,
              render:    bool = False) -> None:
        '''Εκπαίδευση του Mario agent'''
        print(f'Έναρξη εκπαίδευσης για {episodes} επεισόδια...')

        # Dir αποθήκευσης μοντέλων
        save_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        os.makedirs(save_dir, exist_ok = True)
        
        for episode in range(episodes):
            # Επαναφορά περιβάλλοντος και αρχικής κατάστασης
            state = self.env.reset()
            self.prev_x_pos = 40 # Βάση δοκιμών, ο Mario ξεκινάει από x = 40!
            
            total_reward = 0
            steps        = 0
            done         = False
            
            while not done and steps < self.max_episode_steps:
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
            if avg_score > self.best_avg_score:
                self.best_avg_score = avg_score
            
            if episode % 10 == 0:
                print(f'Episode {episode}/{episodes}')
                print(f'Score: {total_reward:.2f}, Avg Score: {avg_score:.2f}')
                print(f'Epsilon: {self.agent.epsilon:.4f}, Steps: {steps}')
                print(f"World Position: {info.get('x_pos', 0)}")
                print('-' * 50)
            
            # Αποθήκευση μοντέλου ανά save_freq επεισόδια
            if episode % save_freq == 0 and episode > 0:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                save_path = os.path.join(save_dir, f'mario_model_ep{episode}_{timestamp}.pth')
                self.agent.save_model(save_path)
                print(f'Best Average Score so far: {self.best_avg_score:.2f}')

            if done and info.get('flag_get', False) and total_reward > 3400:
                # Το score πρέπει να είναι πάνω από 3400
                # για να θεωρηθεί expert! Βάση δοκιμών!
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                model_path = os.path.join(
                    save_dir,
                    f'mario_model_FLAG_ep{episode}_{int(total_reward)}_{timestamp}.pth'
                )
                self.agent.save_model(model_path)
                print(f'Model που τερμάτισε: {model_path}')
        
        # Final save
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        final_save_path = os.path.join(save_dir, f'mario_model_final_{timestamp}.pth')
        self.agent.save_model(final_save_path)
        
        # Ζωγραφικηηηηηή!
        self._plot_training_progress()
        print('Η εκπαίδευση ολοκληρώθηκε!')

        return
    
    def shape_reward(self, reward: float, info: dict, done: bool) -> float:
        '''Custom reward διαμόρφωση για καλύτερη εκπαίδευση'''
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
    
    def test(self,
             model_path:       str = None,
             episodes:         int = 1,
             render:           bool = True,
             show_controller:  bool = False,
             env_unresponsive: bool = False,
             test_agent:       MarioAgent = None) -> bool:
        '''
        Δοκιμή μοντέλου
        
        Parameters:
         - env_unresponsive_: True -> ελέγχει αν το περιβάλλον είναι "κολλημένο" - bugged.
        '''
        if model_path is None:
            self.agent = test_agent
        else:
            self.agent.load_model(model_path)
        self.agent.epsilon = 0 # Όχι exploration κατά το testing!
        controller_overlay = NESControllerOverlay()
        
        print(f'Δοκιμή μοντέλου για {episodes} επεισόδια...')
        test_scores = []
        
        for episode in range(episodes):
            state = self.env.reset()
            self.prev_x_pos = 40
            renderer = MarioRenderer(self.env, scale = 3.)
            
            total_reward = 0
            steps        = 0
            done         = False
            
            # Variables to track repeated actions and Mario's position
            action_history   = []
            position_history = []
            stuck_counter    = 0
            force_no_action  = False
            
            while not done:
                if render:
                    renderer.render()
                
                current_x_pos = self.prev_x_pos # Track current position
                
                # Check if Mario is stuck (no response from environment) - Ξεκολλάμε το gym!!!
                if env_unresponsive:
                    if len(action_history) >= 10:
                        # Check if last 10 actions are the same
                        last_10_actions = action_history[-10:]
                        if len(set(last_10_actions)) == 1:
                            # Check if Mario's position hasn't changed significantly - Env froze!
                            if len(position_history) >= 10:
                                last_10_positions = position_history[-10:]
                                position_change   = max(last_10_positions) - min(last_10_positions)
                                if position_change <= 5: # Mario hasn't moved much (adjust threshold as needed)
                                    if model_path is not None:
                                        print('\nEnvironment froze / became unresponsive.')
                                        print('Taking emergency recovery step!')
                                    force_no_action = True
                                    stuck_counter  += 1
                
                # Action selection
                if force_no_action:
                    # Force "no action"
                    action = 0
                    force_no_action = False
                    # Clear history to restart detection
                    action_history = []
                    position_history = []
                else:
                    # Normal action selection from trained model
                    action = self.agent.act(state, training = False)
                
                # Track action and position history
                action_history.append(action)
                position_history.append(current_x_pos)
                
                # Keep only last 15 entries to avoid memory issues
                if len(action_history) > 15:
                    action_history = action_history[-15:]
                    position_history = position_history[-15:]
                
                if show_controller:
                    controller_overlay.show(self.actions[action])
                
                next_state, raw_reward, done, info = self.env.step(action)

                # *CRITICAL*: Apply same reward shaping as training for consistent evaluation
                # Raw reward = base Mario environment reward (score, progress, completion)
                # Shaped reward = raw reward + custom modifications:
                #   - Progress bonus: +0.1 per pixel moved right
                #   - Time penalty: -0.1 per step (encourages speed)
                #   - Death penalty: -10 if Mario dies
                #   - Flag bonus: +100 for completing level
                shaped_reward = self.shape_reward(raw_reward, info, done)
                
                state         = next_state
                total_reward += shaped_reward
                steps        += 1
                
                if done:
                    break
            
            test_scores.append(total_reward)
            if model_path is not None:
                print(f'\nTest Episode {episode + 1}: Score = {total_reward}, Steps = {steps}')
                print(f"Final Position: {info.get('x_pos', 0)}, Lives: {info.get('life', 3)}")
                if stuck_counter > 0:
                    print(f'Environment unresponsive {stuck_counter} times...')
        
        avg_test_score = np.mean(test_scores)
        if model_path is not None:
            print(f'\nAverage Test Score: {avg_test_score:.2f}')
        
        cv2.destroyAllWindows()
        self.env.close()

        temp = False
        if info.get('flag_get', False):
            temp = True

        return temp
    
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
        plt.savefig(f"mario_training_progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.show()

        return
