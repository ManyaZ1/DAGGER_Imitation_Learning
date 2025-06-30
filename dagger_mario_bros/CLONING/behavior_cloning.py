import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import pickle
from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split

import os, sys
base_dir = os.path.dirname(__file__)
temp     = os.path.abspath(os.path.join(base_dir, '..'))
sys.path.append(temp)
from observation_wrapper import PartialObservationWrapper

class ExpertDemonstrationCollector:
    """Συλλέκτης expert demonstrations για behavior cloning"""
    
    def __init__(self, expert_agent, env):
        self.expert_agent = expert_agent
        self.env = env
        
    def collect_demonstrations(self, 
                             num_episodes: int = 50,
                             save_path: Optional[str] = None) -> List[Dict]:
        """Συλλογή expert demonstrations"""
        
        print(f"Συλλογή {num_episodes} expert demonstrations...")
        demonstrations = []
        episode_rewards = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_demos = []
            done = False
            steps = 0
            
            while not done and steps < 1000:  # Max steps to prevent infinite loops
                # Expert επιλέγει action (χωρίς exploration)
                action = self.expert_agent.act(state, training=False)
                
                # Αποθήκευση state-action pair
                episode_demos.append({
                    'state': state.copy(),
                    'action': action
                })
                
                # Εκτέλεση action
                next_state, reward, done, info = self.env.step(action)
                
                state = next_state
                episode_reward += reward
                steps += 1
            
            # Αποθήκευση demonstrations από αυτό το episode
            demonstrations.extend(episode_demos)
            episode_rewards.append(episode_reward)
            
            if episode % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                print(f"Episode {episode}/{num_episodes}, "
                      f"Steps: {steps}, "
                      f"Reward: {episode_reward:.2f}, "
                      f"Avg Reward: {avg_reward:.2f}")
        
        print(f"Συλλέχθηκαν {len(demonstrations)} state-action pairs")
        print(f"Μέσος όρος reward: {np.mean(episode_rewards):.2f}")
        
        # Αποθήκευση αν ζητήθηκε
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                pickle.dump(demonstrations, f)
            print(f"Demonstrations αποθηκεύτηκαν στο {save_path}")
        # if save_path:
        #     with open(save_path, 'wb') as f:
        #         pickle.dump(demonstrations, f)
        #     print(f"Demonstrations αποθηκεύτηκαν στο {save_path}")
        
        return demonstrations

class BehaviorCloningNetwork(nn.Module):
    """Νευρωνικό δίκτυο για behavior cloning"""
    
    def __init__(self, input_shape: Tuple[int, ...], n_actions: int):
        super(BehaviorCloningNetwork, self).__init__()
        
        self.input_shape = input_shape
        self.n_actions = n_actions
        
        # Adaptive architecture based on input shape
        if len(input_shape) == 3:  # Convolutional for image input
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
            
            self.classifier = nn.Sequential(
                nn.Linear(conv_out_size, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, n_actions)
            )
            
        else:  # Fully connected for flattened input
            input_size = np.prod(input_shape)
            self.classifier = nn.Sequential(
                nn.Linear(input_size, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, n_actions)
            )
            self.conv = None
    
    def _get_conv_out(self, shape: Tuple[int, ...]) -> int:
        """Υπολογισμός μεγέθους εξόδου conv layers"""
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.conv is not None:
            x = self.conv(x)
            x = x.view(x.size(0), -1)  # Flatten
        else:
            x = x.view(x.size(0), -1)  # Flatten
        
        return self.classifier(x)


class BehaviorCloningAgent:
    """Agent που εκπαιδεύεται με behavior cloning"""
    
    def __init__(self, 
                 state_shape: Tuple[int, ...],
                 n_actions: int,
                 lr: float = 1e-4):
        
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Δημιουργία δικτύου
        self.network = BehaviorCloningNetwork(state_shape, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.training_losses = []
        self.validation_losses = []
        self.validation_accuracies = []
        
        print(f"BC Agent initialized on {self.device}")
        print(f"Network architecture:\n{self.network}")
    
    def train(self, 
              demonstrations: List[Dict],
              observation_wrapper: Optional[PartialObservationWrapper] = None,
              epochs: int = 100,
              batch_size: int = 32,
              validation_split: float = 0.2) -> None:
        """Εκπαίδευση με behavior cloning"""
        
        print(f"\nΕκπαίδευση BC agent για {epochs} epochs...")
        
        # Προετοιμασία δεδομένων
        states, actions = self._prepare_data(demonstrations, observation_wrapper)
        
        # Train/validation split
        train_states, val_states, train_actions, val_actions = train_test_split(
            states, actions, test_size=validation_split, random_state=42
        )
        
        # Convert to tensors
        train_states = torch.FloatTensor(train_states).to(self.device)
        train_actions = torch.LongTensor(train_actions).to(self.device)
        val_states = torch.FloatTensor(val_states).to(self.device)
        val_actions = torch.LongTensor(val_actions).to(self.device)
        
        print(f"Training samples: {len(train_states)}")
        print(f"Validation samples: {len(val_states)}")
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            self.network.train()
            train_loss = 0.0
            
            # Shuffle training data
            indices = torch.randperm(len(train_states))
            
            for i in range(0, len(train_states), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_states = train_states[batch_indices]
                batch_actions = train_actions[batch_indices]
                
                # Forward pass
                predictions = self.network(batch_states)
                loss = self.criterion(predictions, batch_actions)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / (len(train_states) // batch_size + 1)
            self.training_losses.append(avg_train_loss)
            
            # Validation phase
            self.network.eval()
            with torch.no_grad():
                val_predictions = self.network(val_states) #  Για κάθε sample στο validation set, δίνει logits για κάθε action
                #πόσο "λάθος" είναι το δίκτυο στις προβλέψεις του σε σχέση με τον expert
                val_loss = self.criterion(val_predictions, val_actions)  #val_actions ground-truth actions του expert 
 
                # Accuracy
                predicted_actions = val_predictions.argmax(dim=1)
                accuracy = (predicted_actions == val_actions).float().mean().item()
                
                self.validation_losses.append(val_loss.item())
                self.validation_accuracies.append(accuracy)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs}")
                print(f"Train Loss: {avg_train_loss:.4f}")
                print(f"Val Loss: {val_loss.item():.4f}")
                print(f"Val Accuracy: {accuracy:.4f}")
                print("-" * 40)
        
        print("Εκπαίδευση ολοκληρώθηκε!")
    
    def _prepare_data(self, 
                     demonstrations: List[Dict],
                     observation_wrapper: Optional[PartialObservationWrapper] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Προετοιμασία δεδομένων για εκπαίδευση"""
        
        states = []
        actions = []
        
        for demo in demonstrations:
            state = demo['state']
            action = demo['action']
            
            # Εφαρμογή observation transformation αν υπάρχει
            if observation_wrapper:
                state = observation_wrapper.transform_observation(state)
            
            states.append(state)
            actions.append(action)
        
        return np.array(states), np.array(actions)
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Επιλογή action βάσει του εκπαιδευμένου δικτύου"""
        
        self.network.eval()
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            predictions = self.network(state_tensor)
            action = predictions.argmax().item()
        
        return action
    
    def plot_training_progress(self) -> None:
        """Γραφική απεικόνιση προόδου εκπαίδευσης"""
        
        if not self.training_losses:
            print("Δεν υπάρχουν δεδομένα εκπαίδευσης για γραφική απεικόνιση")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        epochs = range(1, len(self.training_losses) + 1)
        ax1.plot(epochs, self.training_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, self.validation_losses, 'r-', label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Progress - Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(epochs, self.validation_accuracies, 'g-', label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training Progress - Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(f'plots_bc/bc_training_progress_{timestamp}.png', dpi=300, bbox_inches='tight') #
        #plt.show()
    
    def save_model(self, filepath: str) -> None:
        """Αποθήκευση εκπαιδευμένου μοντέλου"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_losses': self.training_losses,
            'validation_losses': self.validation_losses,
            'validation_accuracies': self.validation_accuracies,
            'state_shape': self.state_shape,
            'n_actions': self.n_actions
        }, filepath)
        print(f'BC model αποθηκεύτηκε στο {filepath}')
    
    def load_model(self, filepath: str) -> None:
        """Φόρτωση εκπαιδευμένου μοντέλου"""
        print("[DEBUG] BEHAVIOR_CLONING.PY 325 calling load_model")
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_losses = checkpoint.get('training_losses', [])
        self.validation_losses = checkpoint.get('validation_losses', [])
        self.validation_accuracies = checkpoint.get('validation_accuracies', [])
        
        print(f'BC model φορτώθηκε από το {filepath}')


class BehaviorCloningEvaluator:
    """Αξιολογητής για behavior cloning agents"""
    
    def __init__(self, env):
        self.env = env
    
    def evaluate_agent(self, 
                      agent,
                      observation_wrapper: Optional[PartialObservationWrapper] = None,
                      num_episodes: int = 10) -> Dict:
        """Αξιολόγηση BC agent"""
        
        scores = []
        episode_lengths = []
        max_x_positions = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            steps = 0
            done = False
            max_x_pos = 0
            
            while not done and steps < 5000:
                # Transform observation if needed
                if observation_wrapper:
                    observed_state = observation_wrapper.transform_observation(state)
                else:
                    observed_state = state
                
                action = agent.act(observed_state, training=False)
                next_state, reward, done, info = self.env.step(action)
                
                state = next_state
                total_reward += reward
                steps += 1
                
                # Track progress
                x_pos = info.get('x_pos', 0)
                max_x_pos = max(max_x_pos, x_pos)
            print(f"[{episode+1:02d}] Score: {total_reward:.1f} | Steps: {steps} | Max X: {max_x_pos} | Flag: {info.get('flag_get', False)}")
            scores.append(total_reward)
            episode_lengths.append(steps)
            max_x_positions.append(max_x_pos)
        
        return {
            'scores': scores,
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'mean_episode_length': np.mean(episode_lengths),
            'mean_max_x_position': np.mean(max_x_positions)
        }
    
    def compare_agents(self, 
                      agents_dict: Dict,
                      observation_wrappers: Dict,
                      num_episodes: int = 10) -> Dict:
        """Σύγκριση πολλαπλών agents"""
        
        results = {}
        
        for agent_name, agent in agents_dict.items():
            wrapper = observation_wrappers.get(agent_name, None)
            print(f"\nΑξιολόγηση {agent_name}...")
            
            results[agent_name] = self.evaluate_agent(
                agent, wrapper, num_episodes
            )
            
            print(f"Μέσος όρος score: {results[agent_name]['mean_score']:.2f}")
            print(f"Μέσος όρος x position: {results[agent_name]['mean_max_x_position']:.2f}")
        
        return results