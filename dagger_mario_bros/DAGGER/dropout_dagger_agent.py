import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
from typing import Optional
import random
import os
import sys

# Add path to import from super_dqn
base_dir = os.path.dirname(__file__)
pkg_parent = os.path.abspath(os.path.join(base_dir, '..', 'expert-SMB_DQN'))
sys.path.insert(0, pkg_parent)
super_dqn_path = os.path.abspath(os.path.join(base_dir, '..', 'expert-SMB_DQN', 'super_dqn'))
sys.path.append(super_dqn_path)

from agent import MarioAgent
from dagger_agent import DaggerMarioAgent


class DropoutDQN(nn.Module):
    '''
    Deep Q-Network με dropout regularization για τον Mario
    Παρόμοιο με το κανονικό DQN αλλά με dropout layers για καλύτερη γενίκευση
    '''

    def __init__(self, input_shape: tuple, n_actions: int, dropout_rate: float = 0.5) -> None:
        super(DropoutDQN, self).__init__()
        
        self.dropout_rate = dropout_rate
        
        # Συνελικτικό κομμάτι (για εξαγωγή χαρακτηριστικών από εικόνα εισόδου)
        self.conv = nn.Sequential(
            # 1ο συνελικτικό επίπεδο
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Dropout2d(p=dropout_rate * 0.25),  # Lighter dropout για conv layers
            
            # 2ο συνελικτικό επίπεδο
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Dropout2d(p=dropout_rate * 0.25),
            
            # 3ο συνελικτικό επίπεδο
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Dropout2d(p=dropout_rate * 0.25)
        )
        
        # Υπολογισμός του μεγέθους εξόδου των conv layers
        conv_out_size = self._get_conv_out(input_shape)
        
        # Fully Connected layers με dropout
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),  # Κανονικό dropout για FC layer
            
            # Τελική έξοδος: Q-τιμή για κάθε action
            nn.Linear(512, n_actions)
        )

        return
    
    def _get_conv_out(self, shape: tuple) -> int:
        '''Υπολογίζει το μέγεθος εξόδου των συνελικτικών επιπέδων'''
        # Αναλυτικότερα, περνάει 1 μηδενικό tensor για
        # να δει τις διαστάσεις εξόδου των conv layers!
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Εκτέλεση forward pass με dropout:
        # CNN -> flatten -> Fully Connected -> Q-values
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)


class DropoutDaggerMarioAgent(DaggerMarioAgent):
    '''
    DAGGER-specific Mario Agent με dropout regularization
    
    Κληρονομεί από DaggerMarioAgent και προσθέτει dropout στο δίκτυο
    για καλύτερη γενίκευση και αποφυγή overfitting
    '''
    
    def __init__(self, state_shape: tuple, n_actions: int, dropout_rate: float = 0.5, *args, **kwargs):
        # Αποθήκευση dropout_rate πριν την parent αρχικοποίηση
        self.dropout_rate = dropout_rate
        
        # Κλήση του parent constructor (DaggerMarioAgent inherits from MarioAgent)
        super().__init__(state_shape, n_actions, *args, **kwargs)
        
        # Αντικατάσταση του κανονικού δικτύου με dropout δίκτυο
        self.q_network = DropoutDQN(state_shape, n_actions, dropout_rate).to(self.device)
        self.target_network = DropoutDQN(state_shape, n_actions, dropout_rate).to(self.device)
        
        # Αντικατάσταση των optimizers με τα νέα δίκτυα
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        # DAGGER μνήμη για ζεύγη (state, expert_action)
        self.dagger_memory = deque(maxlen=50000)
        
        # Ξεχωριστός optimizer για supervised learning
        self.dagger_optimizer = optim.Adam(
            self.q_network.parameters(), 
            lr=self.learning_rate * 1.5
        )
        
        # Synchronize target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        print(f'DropoutDaggerMarioAgent initialized with dropout_rate={dropout_rate}')
        
        return
    
    def act(self, state: np.ndarray, **kwargs) -> int:
        '''
        Επιλογή action με dropout δίκτυο
        Κατά το testing το δίκτυο μπαίνει σε eval mode για να απενεργοποιηθεί το dropout
        '''
        if np.random.random() <= self.epsilon:
            return np.random.choice(self.n_actions)
        
        # Θέτουμε το δίκτυο σε evaluation mode για inference
        self.q_network.eval()
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            action = q_values.argmax().item()
        
        return action
    
    def replay(self) -> Optional[float]:
        '''
        Εκπαίδευση DAGGER με dropout - Supervised learning με expert demonstrations
        Το dropout είναι ενεργό κατά την εκπαίδευση για regularization
        '''
        if len(self.dagger_memory) < self.batch_size:
            return None

        # Θέτουμε το δίκτυο σε training mode για να ενεργοποιηθεί το dropout
        self.q_network.train()

        # Sample batch από DAGGER memory (state, expert_action pairs)
        batch = random.sample(self.dagger_memory, self.batch_size)

        states = [transition[0] for transition in batch]
        expert_actions = [transition[1] for transition in batch]

        # Μετατροπή σε tensors
        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        expert_actions_tensor = torch.LongTensor(expert_actions).to(self.device)

        # Forward pass με dropout ενεργό
        action_logits = self.q_network(states_tensor)

        # Supervised loss
        loss = F.cross_entropy(action_logits, expert_actions_tensor)

        # Backpropagation
        self.dagger_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.q_network.parameters(), self.gradient_clip
        )
        self.dagger_optimizer.step()

        return loss.item()
    
    def save_model(self, filepath: str) -> None:
        '''Αποθήκευση μοντέλου με dropout_rate'''
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'dagger_optimizer_state_dict': self.dagger_optimizer.state_dict(),
            'epsilon': self.epsilon,
            'gradient_clip': self.gradient_clip,
            'scores': self.scores,
            'avg_scores': self.avg_scores,
            'dagger_memory_size': len(self.dagger_memory),
            'dropout_rate': self.dropout_rate  # Αποθήκευση dropout rate
        }
        
        torch.save(checkpoint, filepath)
        print(f'DropoutDAGGER model saved: {filepath}')
        
        return
    
    def load_model(self, filepath: str) -> None:
        '''Φόρτωση μοντέλου με dropout support'''
        # Φόρτωση κανονικών components από parent's parent (MarioAgent)
        super(DaggerMarioAgent, self).load_model(filepath)
        
        # Load DAGGER και dropout-specific components
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=FutureWarning)
            checkpoint = torch.load(filepath, map_location=self.device)
        
        if 'dagger_optimizer_state_dict' in checkpoint:
            self.dagger_optimizer.load_state_dict(checkpoint['dagger_optimizer_state_dict'])
            print("DropoutDAGGER optimizer state loaded")
        
        if 'dropout_rate' in checkpoint:
            loaded_dropout_rate = checkpoint['dropout_rate']
            if loaded_dropout_rate != self.dropout_rate:
                print(f"Warning: Model was trained with dropout_rate={loaded_dropout_rate}, "
                      f"but current agent has dropout_rate={self.dropout_rate}")
        
        return