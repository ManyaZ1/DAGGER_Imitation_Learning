import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import random
import numpy as np
from collections import deque

# ----- Ορισμός του νευρωνικού δικτύου DQN -----
class DQN(nn.Module):
    '''Deep Q-Network για τον Mario
    Input shape: (C, H, W) = (4, 84, 84)    # 4 stacked grayscale images

    output of a convolutional layer = floor((input - kernel_size) / stride) + 1
    '''

    def __init__(self, input_shape: tuple, n_actions: int) -> None:
        super(DQN, self).__init__()
        
        # Συνελικτικό κομμάτι (για εξαγωγή χαρακτηριστικών από εικόνα εισόδου)
        self.conv = nn.Sequential(

            # 1ο συνελικτικό επίπεδο
            nn.Conv2d(input_shape[0], 32, kernel_size = 8, stride = 4),  # 32 separate convolution filters, stride 4 to  downscale image,  input_shape[0]=4
            nn.ReLU(),

            # 2ο συνελικτικό επίπεδο
            nn.Conv2d(32, 64, kernel_size = 4, stride = 2),
            nn.ReLU(),

            # 3ο συνελικτικό επίπεδο
            nn.Conv2d(64, 64, kernel_size = 3, stride = 1),
            nn.ReLU()
        )
        
        # Υπολογισμός του μεγέθους εξόδου των conv για
        # σύνδεση με το πλήρως συνδεδεμένο δίκτυο!
        conv_out_size = self._get_conv_out(input_shape)
        
        # Fully Connected layers (πολιτική εξόδου Q-τιμών για κάθε ενέργεια)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),

            # Τελική έξοδος: Q-τιμή για κάθε ενέργεια
            nn.Linear(512, n_actions)
        )

        return
    
    def _get_conv_out(self, shape: tuple) -> int:
        '''Υπολογίζει το μέγεθος εξόδου των συνελικτικών επιπέδων'''
        # Αναλυτικότερα, περνάει 1 μηδενικό tensor για
        # να δει τις διαστάσεις εξόδου των conv layers!
        o = self.conv(torch.zeros(1, *shape))

        return int(np.prod(o.size())) # Συνολικός αριθμός εξόδων (flatten)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Εκτέλεση forward pass:
        # CNN -> flatten -> Fully Connected -> Q-values
        conv_out = self.conv(x).view(x.size()[0], -1)

        return self.fc(conv_out)



# ----- Ορισμός Agent που χρησιμοποιεί το DQN -----
class MarioAgent:
    '''DQN Agent για το Super Mario Bros'''

    def __init__(self,
                 state_shape:   tuple,
                 n_actions:     int,
                 lr:            float = 5e-5,
                 epsilon:       float = 1.,
                 epsilon_decay: float = 0.99995,
                 epsilon_min:   float = 0.01,
                 gradient_clip: float = 1.) -> None:
        self.state_shape   = state_shape
        self.n_actions     = n_actions
        self.epsilon       = epsilon       # Αρχική πιθανότητα τυχαίας εξερεύνησης
        self.epsilon_decay = epsilon_decay # Ρυθμός μείωσης της πιθανότητας
        self.epsilon_min   = epsilon_min   # Κατώτατο όριο ε
        self.learning_rate = lr
        self.gradient_clip = gradient_clip # Gradient clipping threshold
        
        # Επιλογή συσκευής (GPU αν υπάρχει κατά προτίμηση)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Δημιουργία του κύριου και του target δικτύου
        self.q_network      = DQN(state_shape, n_actions).to(self.device)
        self.target_network = DQN(state_shape, n_actions).to(self.device)

        self.optimizer      = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Replay buffer (για 'εμπειρίες')
        self.memory             = deque(maxlen = 100000)
        self.batch_size         = 32
        self.gamma              = 0.99 # Συντελεστής (για μελλοντικές ανταμοιβές)
        self.update_target_freq = 1000 # Κάθε πόσα βήματα ενημερώνεται το target δίκτυο
        self.step_count         = 0
        
        # Στατιστικά εκπαίδευσης
        self.scores = []
        self.avg_scores = []
        
        print(f'Γίνεται χρήση της συσκευής: {self.device}')
        print(f'Gradient clipping threshold: {self.gradient_clip}')
        print(f'Η αρχιτεκτονική του δικτύου είναι:\n{self.q_network}\n')

        return
    
    def remember(self,
                 state:      np.ndarray,
                 action:     int,
                 reward:     float,
                 next_state: np.ndarray,
                 done:       bool) -> None:
        '''Αποθήκευση εμπειρίας στο buffer'''
        self.memory.append((state, action, reward, next_state, done))

        return
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        '''Επιλογή ενέργειας με χρήση πολιτικής epsilon-greedy'''
        if training and random.random() <= self.epsilon:
            # Τυχαία ενέργεια (exploration)
            return random.choice(range(self.n_actions))
        
        # Αλλιώς, επιλέγει την καλύτερη ενέργεια βάσει Q-τιμών (exploitation)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)

        return q_values.cpu().argmax().item()
    
    def replay(self) -> None:
        '''Εκπαίδευση μοντέλου βάσει batch εμπειριών'''
        if len(self.memory) < self.batch_size:
            return # Αν δεν υπάρχουν αρκετές 'εμπειρίες'
        
        # Δειγματοληψία batch από το buffer
        batch = random.sample(self.memory, self.batch_size)
        
        # Αρχικός διαχωρισμός σε numpy arrays για απόδοση!
        states      = np.array([e[0] for e in batch])
        actions     = np.array([e[1] for e in batch])
        rewards     = np.array([e[2] for e in batch], dtype = np.float32)
        next_states = np.array([e[3] for e in batch])
        dones       = np.array([bool(e[4]) for e in batch], dtype = bool)
        
        # Μετατροπή σε PyTorch tensors
        states      = torch.FloatTensor(states).to(self.device)
        actions     = torch.LongTensor(actions).to(self.device)
        rewards     = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones       = torch.BoolTensor(dones).to(self.device)
        
        # Q-τιμές για τις επιλεγμένες ενέργειες στο τρέχον state
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Q-τιμές του target δικτύου για το επόμενο state
        next_q_values = self.target_network(next_states).max(1)[0].detach()

        # Bellman: R + γ * max(Q') (αν όχι done)
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Backpropagation με gradient clipping
        self.optimizer.zero_grad()
        loss.backward()

        # Εφαρμόζουμε gradient clipping για να αποφύγουμε exploding gradients!
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.gradient_clip)

        self.optimizer.step()
        
        # Ενημέρωση του target δικτύου κάθε N {= update_target_freq} βήματα
        self.step_count += 1
        if self.step_count % self.update_target_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Ενημέρωση της εξερεύνησης (ε)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return
    
    def save_model(self, filepath: str) -> None:
        '''Αποθήκευση του εκπαιδευμένου μοντέλου'''
        torch.save({
            'q_network_state_dict':      self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict':      self.optimizer.state_dict(),
            'epsilon':                   self.epsilon,
            'gradient_clip':             self.gradient_clip,
            'scores':                    self.scores,
            'avg_scores':                self.avg_scores
        }, filepath)
        print(f'Το μοντέλο αποθηκεύτηκε στο {filepath}')

        return
    
    def load_model(self, filepath: str) -> None:
        '''Φόρτωση εκπαιδευμένου μοντέλου'''
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon       = checkpoint['epsilon']
        self.gradient_clip = checkpoint.get('gradient_clip', 1.)
        self.scores        = checkpoint.get('scores', [])
        self.avg_scores    = checkpoint.get('avg_scores', [])
        print(f'Το μοντέλο φορτώθηκε από το {filepath}')

        return
