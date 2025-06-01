import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import random
import numpy as np
from collections import deque

# ----- Ορισμός του νευρωνικού δικτύου DQN -----

class EnhancedDQN(nn.Module):
    '''Enhanced DQN that incorporates both visual and structured information'''

    def __init__(self, input_shape: tuple, n_actions: int, structured_state_size: int = 8):
        super(EnhancedDQN, self).__init__()
        
        # Convolutional layers for visual processing
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Calculate conv output size
        conv_out_size = self._get_conv_out(input_shape)
        
        # Separate processing for structured state
        self.structured_fc = nn.Sequential(
            nn.Linear(structured_state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # Combined fully connected layers
        combined_size = conv_out_size + 64  # CNN features + structured features
        self.fc = nn.Sequential(
            nn.Linear(combined_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, visual_input, structured_input):
        # Process visual input through CNN
        conv_out = self.conv(visual_input).view(visual_input.size(0), -1)
        
        # Process structured input
        structured_out = self.structured_fc(structured_input)
        
        # Combine features
        combined = torch.cat([conv_out, structured_out], dim=1)
        
        return self.fc(combined)


class EnhancedMarioAgent:
    '''Enhanced Mario Agent that uses both visual and structured information'''

    def __init__(self, state_shape, n_actions, structured_state_size=8, lr=5e-5, **kwargs):
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.structured_state_size = structured_state_size
        
        # Initialize other parameters...
        self.epsilon = kwargs.get('epsilon', 1.0)
        self.epsilon_decay = kwargs.get('epsilon_decay', 0.99995)
        self.epsilon_min = kwargs.get('epsilon_min', 0.01)
        self.learning_rate = lr
        self.gradient_clip = kwargs.get('gradient_clip', 1.0)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Enhanced networks
        self.q_network = EnhancedDQN(state_shape, n_actions, structured_state_size).to(self.device)
        self.target_network = EnhancedDQN(state_shape, n_actions, structured_state_size).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Replay buffer now stores structured state too
        self.memory = deque(maxlen=100000)
        self.batch_size = 32
        self.gamma = 0.99
        self.update_target_freq = 1000
        self.step_count = 0
        # Training statistics
        self.scores = []
        self.avg_scores = []

    def remember(self, state, action, reward, next_state, done, structured_state, next_structured_state):
        '''Store experience including structured state information'''
        self.memory.append((state, action, reward, next_state, done, structured_state, next_structured_state))

    def act(self, state, structured_state, training=True):
        '''Choose action using both visual and structured information'''
        if training and np.random.random() <= self.epsilon:
            return np.random.choice(self.n_actions)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        structured_tensor = torch.FloatTensor(structured_state).unsqueeze(0).to(self.device)
        
        q_values = self.q_network(state_tensor, structured_tensor)
        return q_values.cpu().argmax().item()

    def replay(self):
        '''Train model using batch of experiences with structured information'''
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch], dtype=np.float32)
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch], dtype=bool)
        structured_states = np.array([e[5] for e in batch])
        next_structured_states = np.array([e[6] for e in batch])
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        structured_states = torch.FloatTensor(structured_states).to(self.device)
        next_structured_states = torch.FloatTensor(next_structured_states).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states, structured_states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        next_q_values = self.target_network(next_states, next_structured_states).max(1)[0].detach()
        
        # Target Q values
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Loss and backprop
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.gradient_clip)
        self.optimizer.step()
        
        # Update target network
        self.step_count += 1
        if self.step_count % self.update_target_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    def save_model(self, filepath: str) -> None:
        '''Save the trained model'''
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'gradient_clip': self.gradient_clip,
            'scores': self.scores,
            'avg_scores': self.avg_scores
        }, filepath)
        print(f'Model saved to {filepath}')

    def load_model(self, filepath: str) -> None:
        '''Load trained model'''
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.gradient_clip = checkpoint.get('gradient_clip', 1.0)
        self.scores = checkpoint.get('scores', [])
        self.avg_scores = checkpoint.get('avg_scores', [])
        print(f'Model loaded from {filepath}')
