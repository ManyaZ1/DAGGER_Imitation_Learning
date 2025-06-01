import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import cv2

import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from super_dqn.agent import DQN
from super_dqn.env_wrappers import MarioPreprocessor 
# Assuming your existing imports and classes are available
# from your_module import DQN, MarioPreprocessor

class MarioModelVisualizer:
    """Comprehensive visualization suite for trained Mario DQN model"""
    
    def __init__(self, model_path, env_name='SuperMarioBros-1-1-v0'):
        self.model_path = model_path
        self.env_name = env_name
        self.activations = {}
        self.gradients = {}
        
        # Load model
        self.load_model()
        self.setup_environment()
        self.register_hooks()
        
    def load_model(self):
        """Load the trained model"""
        print(f"Loading model from: {self.model_path}")
        ckpt = torch.load(self.model_path, map_location='cpu')
        
        # Model architecture
        n_actions = len(SIMPLE_MOVEMENT)
        input_shape = (4, 84, 84)
        
        self.net = DQN(input_shape=input_shape, n_actions=n_actions)
        self.net.load_state_dict(ckpt['q_network_state_dict'])
        self.net.eval()
        
        # Store training stats if available
        self.scores = ckpt.get('scores', [])
        self.avg_scores = ckpt.get('avg_scores', [])
        
        print("Model loaded successfully!")
        
    def setup_environment(self):
        """Setup the Mario environment"""
        self.env = gym_super_mario_bros.make(self.env_name)
        self.env = JoypadSpace(self.env, SIMPLE_MOVEMENT)
        self.env = MarioPreprocessor(self.env)
        
    def register_hooks(self):
        """Register hooks to capture activations and gradients"""
        def capture_activation(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook
            
        def capture_gradient(name):
            def hook(module, grad_input, grad_output):
                self.gradients[name] = grad_output[0].detach()
            return hook
        
        # Register hooks on convolutional layers
        self.net.conv[0].register_forward_hook(capture_activation('conv1'))
        self.net.conv[2].register_forward_hook(capture_activation('conv2'))
        self.net.conv[4].register_forward_hook(capture_activation('conv3'))
        
        # Register backward hooks for gradients
        self.net.conv[0].register_backward_hook(capture_gradient('conv1_grad'))
        self.net.conv[2].register_backward_hook(capture_gradient('conv2_grad'))
        self.net.conv[4].register_backward_hook(capture_gradient('conv3_grad'))
    
    def visualize_training_progress(self):
        """Plot training progress if available"""
        if not self.scores:
            print("No training scores available in checkpoint")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Raw scores
        ax1.plot(self.scores, alpha=0.7, color='blue')
        ax1.set_title('Training Scores Over Episodes')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Score')
        ax1.grid(True, alpha=0.3)
        
        # Average scores
        if self.avg_scores:
            ax2.plot(self.avg_scores, color='red', linewidth=2)
            ax2.set_title('Average Scores (100-episode window)')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Average Score')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def visualize_conv_filters(self):
        """Visualize learned convolutional filters"""
        layers = [('conv1', self.net.conv[0]), ('conv2', self.net.conv[2]), ('conv3', self.net.conv[4])]
        
        for layer_name, layer in layers:
            weights = layer.weight.data.cpu().numpy()
            num_filters = weights.shape[0]
            num_channels = weights.shape[1]
            
            print(f"\n{layer_name} filters shape: {weights.shape}")
            
            # Show first 16 filters for each input channel
            fig, axes = plt.subplots(4, 4, figsize=(12, 12))
            fig.suptitle(f'{layer_name} Filters (First 16)', fontsize=16)
            
            for i in range(min(16, num_filters)):
                ax = axes[i//4, i%4]
                
                # Average across input channels or show first channel
                if num_channels > 1:
                    filter_img = weights[i].mean(axis=0)  # Average across channels
                else:
                    filter_img = weights[i, 0]
                
                im = ax.imshow(filter_img, cmap='RdBu_r', vmin=-filter_img.std(), vmax=filter_img.std())
                ax.set_title(f'Filter {i}')
                ax.axis('off')
                
            plt.tight_layout()
            plt.show()
    
    def visualize_feature_maps(self, state):
        """Visualize feature maps for a given state"""
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        
        # Forward pass to get activations
        with torch.no_grad():
            _ = self.net(state_tensor)
        
        # Visualize activations for each layer
        for layer_name in ['conv1', 'conv2', 'conv3']:
            if layer_name not in self.activations:
                continue
                
            activation = self.activations[layer_name].squeeze().cpu().numpy()
            num_channels = activation.shape[0]
            
            # Show first 16 feature maps
            fig, axes = plt.subplots(4, 4, figsize=(12, 12))
            fig.suptitle(f'{layer_name} Feature Maps', fontsize=16)
            
            for i in range(min(16, num_channels)):
                ax = axes[i//4, i%4]
                ax.imshow(activation[i], cmap='viridis')
                ax.set_title(f'Channel {i}')
                ax.axis('off')
                
            plt.tight_layout()
            plt.show()
    
    def visualize_q_values_heatmap(self, num_episodes=50):
        """Visualize Q-values across different game states"""
        action_names = ['NOOP', 'RIGHT', 'R+A', 'R+B', 'R+A+B', 'A', 'LEFT']
        q_value_history = []
        positions = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            step = 0
            
            while not done and step < 200:  # Limit steps per episode
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                
                with torch.no_grad():
                    q_values = self.net(state_tensor).squeeze().numpy()
                
                q_value_history.append(q_values)
                # You might want to track x_pos from info if available
                positions.append(step)  # Use step as proxy for position
                
                # Take action with highest Q-value
                action = np.argmax(q_values)
                state, _, done, info = self.env.step(action)
                step += 1
                
            if episode % 10 == 0:
                print(f"Episode {episode}/{num_episodes} completed")
        
        # Create heatmap
        q_matrix = np.array(q_value_history)
        
        plt.figure(figsize=(15, 8))
        sns.heatmap(q_matrix.T, 
                   yticklabels=action_names,
                   xticklabels=False,
                   cmap='RdYlBu_r',
                   center=0)
        plt.title('Q-Values Heatmap Across Game States')
        plt.xlabel('Game Steps')
        plt.ylabel('Actions')
        plt.show()
        
        # Plot Q-value statistics
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Mean Q-values per action
        mean_q_values = q_matrix.mean(axis=0)
        ax1.bar(action_names, mean_q_values)
        ax1.set_title('Average Q-Values per Action')
        ax1.set_ylabel('Q-Value')
        ax1.tick_params(axis='x', rotation=45)
        
        # Q-value variance per action
        var_q_values = q_matrix.var(axis=0)
        ax2.bar(action_names, var_q_values)
        ax2.set_title('Q-Value Variance per Action')
        ax2.set_ylabel('Variance')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def visualize_saliency_map(self, state):
        """Generate saliency map showing which pixels are most important"""
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        state_tensor.requires_grad_(True)
        
        # Forward pass
        q_values = self.net(state_tensor)
        
        # Get the action with highest Q-value
        best_action = q_values.argmax()
        
        # Backward pass to get gradients
        self.net.zero_grad()
        q_values[0, best_action].backward()
        
        # Get gradients with respect to input
        gradients = state_tensor.grad.data.abs()
        
        # Average across channels and batch
        saliency = gradients.squeeze().mean(dim=0).cpu().numpy()
        
        # Visualize
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original frames (last 3 frames)
        for i in range(3):
            axes[0, i].imshow(state[i+1], cmap='gray')
            axes[0, i].set_title(f'Frame {i+1}')
            axes[0, i].axis('off')
        
        # Saliency map
        axes[1, 1].imshow(saliency, cmap='hot')
        axes[1, 1].set_title('Saliency Map')
        axes[1, 1].axis('off')
        
        # Overlay saliency on last frame
        overlay = np.zeros((84, 84, 3))
        overlay[:, :, 0] = state[-1] / 255.0  # Original frame
        overlay[:, :, 1] = state[-1] / 255.0  # Original frame
        overlay[:, :, 2] = state[-1] / 255.0  # Original frame
        
        # Add saliency as red overlay
        saliency_norm = (saliency - saliency.min()) / (saliency.max() - saliency.min())
        overlay[:, :, 0] = np.maximum(overlay[:, :, 0], saliency_norm)
        
        axes[1, 2].imshow(overlay)
        axes[1, 2].set_title('Saliency Overlay')
        axes[1, 2].axis('off')
        
        # Remove empty subplots
        axes[1, 0].axis('off')
        
        plt.suptitle(f'Saliency Analysis (Best Action: {best_action})', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def analyze_decision_making(self, num_steps=100):
        """Analyze decision-making patterns"""
        state = self.env.reset()
        
        decisions = []
        q_value_spreads = []
        confidences = []
        
        for step in range(num_steps):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                q_values = self.net(state_tensor).squeeze().numpy()
            
            best_action = np.argmax(q_values)
            q_spread = q_values.max() - q_values.min()
            confidence = torch.softmax(torch.tensor(q_values), dim=0).max().item()
            
            decisions.append(best_action)
            q_value_spreads.append(q_spread)
            confidences.append(confidence)
            
            # Take the action
            state, _, done, _ = self.env.step(best_action)
            
            if done:
                state = self.env.reset()
        
        # Visualize decision patterns
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Action distribution
        action_counts = np.bincount(decisions, minlength=len(SIMPLE_MOVEMENT))
        action_names = ['NOOP', 'RIGHT', 'R+A', 'R+B', 'R+A+B', 'A', 'LEFT']
        
        axes[0, 0].bar(action_names, action_counts)
        axes[0, 0].set_title('Action Distribution')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Q-value spread over time
        axes[0, 1].plot(q_value_spreads)
        axes[0, 1].set_title('Q-Value Spread Over Time')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Max Q - Min Q')
        
        # Confidence over time
        axes[1, 0].plot(confidences)
        axes[1, 0].set_title('Decision Confidence Over Time')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Softmax Probability')
        
        # Confidence vs Q-spread scatter
        axes[1, 1].scatter(q_value_spreads, confidences, alpha=0.6)
        axes[1, 1].set_xlabel('Q-Value Spread')
        axes[1, 1].set_ylabel('Confidence')
        axes[1, 1].set_title('Confidence vs Q-Value Spread')
        
        plt.tight_layout()
        plt.show()
    
    def run_comprehensive_analysis(self):
        """Run all visualizations"""
        print("=== Mario DQN Model Analysis ===")
        
        # 1. Training progress
        print("\n1. Visualizing training progress...")
        self.visualize_training_progress()
        
        # 2. Convolutional filters
        print("\n2. Visualizing learned filters...")
        self.visualize_conv_filters()
        
        # 3. Get a sample state for further analysis
        sample_state = self.env.reset()
        
        # 4. Feature maps
        print("\n3. Visualizing feature maps...")
        self.visualize_feature_maps(sample_state)
        
        # 5. Saliency map
        print("\n4. Generating saliency map...")
        self.visualize_saliency_map(sample_state)
        
        # 6. Q-values heatmap
        print("\n5. Analyzing Q-values across states...")
        self.visualize_q_values_heatmap(num_episodes=20)
        
        # 7. Decision making analysis
        print("\n6. Analyzing decision-making patterns...")
        self.analyze_decision_making()
        
        print("\nAnalysis complete!")

# Usage example:
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, 'models', 'mario_model_best.pth')
    
    # Create visualizer
    visualizer = MarioModelVisualizer(MODEL_PATH)
    
    # Run comprehensive analysis
    visualizer.run_comprehensive_analysis()
    
    # Or run individual visualizations:
    # visualizer.visualize_training_progress()
    # visualizer.visualize_conv_filters()
    # sample_state = visualizer.env.reset()
    # visualizer.visualize_saliency_map(sample_state)