import os
import sys
import torch
import random
import numpy as np
import torch.optim as optim
from collections import deque
import torch.nn.functional as F

base_dir = os.path.dirname(__file__)
temp = os.path.abspath(
    os.path.join(base_dir, '..', 'expert-SMB_DQN', 'super_dqn')
)
sys.path.append(temp)
from agent import MarioAgent # Import og agent

class DaggerMarioAgent(MarioAgent):
    '''
    DAGGER-specific Mario Agent που κληρονομεί την κλάση MarioAgent
    
    - Αλλαγές:
    1. Η μνήμη αποθηκεύει ζεύγη (state, expert_action) αντί για πλήρη RL tuples
    2. Η συνάρτηση replay() εκτελεί supervised learning αντί για Q-learning
    3. Ξεχωριστός optimizer για το supervised learning
    '''
    
    def __init__(self, *args, **kwargs):
        # Parent DQN agent αρχικοποίηση
        super().__init__(*args, **kwargs)
        
        # DAGGER μνήμη για ζεύγη (state, expert_action)
        self.dagger_memory = deque(maxlen = 50000)
        
        # Ξεχωριστός optimizer για supervised learning (x2 learning rate)
        self.dagger_optimizer = optim.Adam(
            self.q_network.parameters(), 
            lr = self.learning_rate * 2
        )
        
        return
        
    def remember(self, state: np.ndarray, expert_action: int, *args) -> None:
        '''
        Αποθήκευση ζευγών (state, expert_action) για εκπαίδευση DAGGER
        
        Args:
            state:         Current state
            expert_action: Action του expert για το συγκεκριμένο state
            *args:         Παράμετροι που αγνοούνται, αφού DAGGER!
        '''
        self.dagger_memory.append((state, expert_action))

        return
        
    def replay(self) -> float:
        '''
        Εκπαίδευση DAGGER - Supervised learning με expert demonstrations
        
        Returns:
            loss: Training loss, or None αν τα δεδομένα δεν επαρκούν
        '''
        if len(self.dagger_memory) < self.batch_size:
            return None
        
        # Sample batch από DAGGER memory (state, expert_action pairs)
        batch = random.sample(self.dagger_memory, self.batch_size)
        
        states         = np.array([transition[0] for transition in batch])
        expert_actions = np.array([transition[1] for transition in batch])
        
        # Μετατροπή σε tensors
        states         = torch.FloatTensor(states).to(self.device)
        expert_actions = torch.LongTensor(expert_actions).to(self.device)
        
        # Forward pass: Q-values -> action logits
        action_logits = self.q_network(states)
        
        # Supervised learning loss: μάθε να προβλέπεις τα actions του expert
        loss = F.cross_entropy(action_logits, expert_actions)
        
        # Backpropagation
        self.dagger_optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (inherited)
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.gradient_clip)
        
        self.dagger_optimizer.step()
        
        return loss.item()
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        '''
        Επιλογή action με βάση το τρέχον policy ή το greedy policy
        
        Κατά το training:   Χρήση του current policy
        Κατά το evaluation: Χρήση του greedy policy
        '''
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_logits = self.q_network(state_tensor)
            
            if training: # Θέλουμε exploration!
                action_probs = F.softmax(action_logits, dim=1)
                action       = torch.multinomial(action_probs, 1).item()
            else: # Greedy policy
                action = action_logits.argmax().item()
        
        return action
    
    def save_model(self, filepath: str) -> None:
        checkpoint = {
            'q_network_state_dict':        self.q_network.state_dict(),
            'target_network_state_dict':   self.target_network.state_dict(),
            'optimizer_state_dict':        self.optimizer.state_dict(),
            'dagger_optimizer_state_dict': self.dagger_optimizer.state_dict(),
            'epsilon':                     self.epsilon,
            'gradient_clip':               self.gradient_clip,
            'scores':                      self.scores,
            'avg_scores':                  self.avg_scores,
            'dagger_memory_size':          len(self.dagger_memory)
        }
        
        torch.save(checkpoint, filepath)
        print(f'DAGGER model saved: {filepath}')

        return
        
    def load_model(self, filepath: str) -> None:
        super().load_model(filepath)
        
        # Load DAGGER-specific components αν υπάρχουν!
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category = FutureWarning)
            checkpoint = torch.load(filepath, map_location=self.device)
            
        if 'dagger_optimizer_state_dict' in checkpoint:
            self.dagger_optimizer.load_state_dict(checkpoint['dagger_optimizer_state_dict'])
            print("DAGGER optimizer state loaded")
        
        return
