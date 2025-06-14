import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple
import seaborn as sns
from scipy import stats
import os

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class MarioDQNVisualizer:
    def __init__(self, model_path: str):
        """
        Initialize the visualizer with a saved model file
        
        Args:
            model_path: Path to the saved .pth model file
        """
        self.model_path = model_path
        self.data = None
        self.load_model_data()
    
    def load_model_data(self) -> None:
        """Load the saved model data"""
        try:
            self.data = torch.load(self.model_path, map_location='cpu')
            print(f"✅ Model data loaded successfully from {self.model_path}")
            print(f"📊 Episodes trained: {len(self.data.get('scores', []))}")
            print(f"🎯 Final epsilon: {self.data.get('epsilon', 'N/A')}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def plot_episode_scores(self, figsize: Tuple[int, int] = (12, 6)) -> None:
        """Plot episode scores over time"""
        scores = self.data.get('scores', [])
        if not scores:
            print("⚠️ No scores data found in model file")
            return
        
        plt.figure(figsize=figsize)
        episodes = range(1, len(scores) + 1)
        
        # Plot raw scores
        plt.subplot(1, 2, 1)
        plt.plot(episodes, scores, alpha=0.7, linewidth=0.8, color='steelblue')
        plt.title('Episode Scores Over Time', fontsize=14, fontweight='bold')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(episodes, scores, 1)
        p = np.poly1d(z)
        plt.plot(episodes, p(episodes), "--", color='red', alpha=0.8, 
                label=f'Trend (slope: {z[0]:.2f})')
        plt.legend()
        
        # Plot score distribution
        plt.subplot(1, 2, 2)
        plt.hist(scores, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        plt.title('Score Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        plt.axvline(np.mean(scores), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(scores):.1f}')
        plt.axvline(np.median(scores), color='orange', linestyle='--', 
                   label=f'Median: {np.median(scores):.1f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_moving_average(self, window_sizes: List[int] = [50, 100, 200], 
                           figsize: Tuple[int, int] = (14, 8)) -> None:
        """Plot moving averages with different window sizes"""
        scores = self.data.get('scores', [])
        avg_scores = self.data.get('avg_scores', [])
        
        if not scores:
            print("⚠️ No scores data found")
            return
        
        plt.figure(figsize=figsize)
        episodes = range(1, len(scores) + 1)
        
        # Plot raw scores (lighter)
        plt.plot(episodes, scores, alpha=0.3, linewidth=0.5, 
                color='gray', label='Raw Scores')
        
        # Plot saved average scores if available
        if avg_scores and len(avg_scores) == len(scores):
            plt.plot(episodes, avg_scores, linewidth=2, color='red', 
                    label='Saved Average Scores')
        
        # Calculate and plot different moving averages
        colors = ['blue', 'green', 'orange', 'purple']
        for i, window in enumerate(window_sizes):
            if len(scores) >= window:
                moving_avg = self.calculate_moving_average(scores, window)
                plt.plot(range(window, len(scores) + 1), moving_avg, 
                        linewidth=2, color=colors[i % len(colors)], 
                        label=f'Moving Avg ({window} episodes)')
        
        plt.title('Episode Scores with Moving Averages', fontsize=16, fontweight='bold')
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_epsilon_decay(self, initial_epsilon: float = 1.0, 
                          decay_rate: float = 0.995, min_epsilon: float = 0.01,
                          figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Plot epsilon decay curve (reconstructed from final epsilon and episode count)
        
        Args:
            initial_epsilon: Starting epsilon value
            decay_rate: Epsilon decay rate per episode
            min_epsilon: Minimum epsilon value
        """
        scores = self.data.get('scores', [])
        final_epsilon = self.data.get('epsilon', min_epsilon)
        
        if not scores:
            print("⚠️ No episode data found")
            return
        
        num_episodes = len(scores)
        
        # Reconstruct epsilon decay curve
        epsilons = []
        epsilon = initial_epsilon
        
        for episode in range(num_episodes):
            epsilons.append(epsilon)
            if epsilon > min_epsilon:
                epsilon = max(min_epsilon, epsilon * decay_rate)
        
        plt.figure(figsize=figsize)
        episodes = range(1, num_episodes + 1)
        
        plt.plot(episodes, epsilons, linewidth=2, color='darkgreen')
        plt.title('Epsilon Decay Curve', fontsize=16, fontweight='bold')
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Epsilon (Exploration Rate)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add annotations
        plt.axhline(y=min_epsilon, color='red', linestyle='--', alpha=0.7, 
                   label=f'Min Epsilon: {min_epsilon}')
        plt.axhline(y=final_epsilon, color='orange', linestyle='--', alpha=0.7, 
                   label=f'Final Epsilon: {final_epsilon:.4f}')
        
        # Find when epsilon reached minimum
        min_reached = next((i for i, e in enumerate(epsilons) if e <= min_epsilon), num_episodes)
        if min_reached < num_episodes:
            plt.axvline(x=min_reached, color='blue', linestyle=':', alpha=0.7, 
                       label=f'Min reached at episode {min_reached}')
        
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def plot_performance_metrics(self, figsize: Tuple[int, int] = (15, 10)) -> None:
        """Plot comprehensive performance metrics"""
        scores = self.data.get('scores', [])
        if not scores:
            print("⚠️ No scores data found")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Super Mario Bros DQN Performance Analysis', fontsize=16, fontweight='bold')
        
        episodes = range(1, len(scores) + 1)
        
        # 1. Cumulative average score
        cumulative_avg = np.cumsum(scores) / np.arange(1, len(scores) + 1)
        axes[0, 0].plot(episodes, cumulative_avg, linewidth=2, color='blue')
        axes[0, 0].set_title('Cumulative Average Score')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Cumulative Average')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Score improvement over time (rolling difference)
        if len(scores) > 1:
            rolling_diff = np.convolve(np.diff(scores), np.ones(min(50, len(scores)//4))/min(50, len(scores)//4), mode='valid')
            axes[0, 1].plot(range(51, len(scores) + 1), rolling_diff, color='green', linewidth=2)
            axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
            axes[0, 1].set_title('Score Improvement Trend (50-episode moving avg)')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Score Change')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Performance stability (rolling standard deviation)
        window = min(100, len(scores)//4)
        if window > 1:
            rolling_std = []
            for i in range(window, len(scores) + 1):
                rolling_std.append(np.std(scores[i-window:i]))
            
            axes[1, 0].plot(range(window, len(scores) + 1), rolling_std, 
                           color='orange', linewidth=2)
            axes[1, 0].set_title(f'Performance Stability ({window}-episode rolling std)')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Score Standard Deviation')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. High score achievements over time
        max_score_so_far = []
        current_max = float('-inf')
        for score in scores:
            if score > current_max:
                current_max = score
            max_score_so_far.append(current_max)
        
        axes[1, 1].plot(episodes, max_score_so_far, linewidth=2, color='red')
        axes[1, 1].set_title('Best Score Achieved Over Time')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Best Score So Far')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def calculate_moving_average(self, data: List[float], window: int) -> List[float]:
        """Calculate moving average"""
        return [np.mean(data[i:i+window]) for i in range(len(data) - window + 1)]
    
    def print_training_summary(self) -> None:
        """Print a summary of training statistics"""
        scores = self.data.get('scores', [])
        if not scores:
            print("⚠️ No scores data found")
            return
        
        print("\n" + "="*50)
        print("🎮 SUPER MARIO BROS DQN TRAINING SUMMARY")
        print("="*50)
        print(f"📊 Total Episodes: {len(scores)}")
        print(f"🏆 Best Score: {max(scores):.1f}")
        print(f"📉 Worst Score: {min(scores):.1f}")
        print(f"📈 Average Score: {np.mean(scores):.1f}")
        print(f"📊 Median Score: {np.median(scores):.1f}")
        print(f"📏 Standard Deviation: {np.std(scores):.1f}")
        print(f"🎯 Final Epsilon: {self.data.get('epsilon', 'N/A')}")
        
        # Performance in different phases
        if len(scores) >= 100:
            early_scores = scores[:100]
            late_scores = scores[-100:]
            print(f"\n🚀 Early Training (first 100 episodes): {np.mean(early_scores):.1f}")
            print(f"🎯 Late Training (last 100 episodes): {np.mean(late_scores):.1f}")
            print(f"📈 Improvement: {np.mean(late_scores) - np.mean(early_scores):.1f}")
        
        print("="*50)
    
    def create_all_visualizations(self, save_plots: bool = False, 
                                output_dir: str = "mario_plots") -> None:
        """Create all visualizations at once"""
        if save_plots:
            os.makedirs(output_dir, exist_ok=True)
            plt.ioff()  # Turn off interactive mode for saving
        
        print("🎨 Generating visualizations...")
        
        # Print summary
        self.print_training_summary()
        
        # Generate all plots
        self.plot_episode_scores()
        if save_plots:
            plt.savefig(f"{output_dir}/episode_scores.png", dpi=300, bbox_inches='tight')
        
        self.plot_moving_average()
        if save_plots:
            plt.savefig(f"{output_dir}/moving_averages.png", dpi=300, bbox_inches='tight')
        
        self.plot_epsilon_decay()
        if save_plots:
            plt.savefig(f"{output_dir}/epsilon_decay.png", dpi=300, bbox_inches='tight')
        
        self.plot_performance_metrics()
        if save_plots:
            plt.savefig(f"{output_dir}/performance_metrics.png", dpi=300, bbox_inches='tight')
        
        if save_plots:
            plt.ion()  # Turn interactive mode back on
            print(f"📁 All plots saved to {output_dir}/ directory")
        
        print("✅ All visualizations completed!")

# Example usage
if __name__ == "__main__":
    # Replace with your actual model path
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, 'models', 'ep30000_MARIO_EXPERT.pth')    
    try:
        # Create visualizer instance
        visualizer = MarioDQNVisualizer(MODEL_PATH)
        
        # Generate all visualizations
        visualizer.create_all_visualizations(save_plots=True)
        
        # Or create individual plots:
        # visualizer.plot_episode_scores()
        # visualizer.plot_moving_average(window_sizes=[50, 100])
        # visualizer.plot_epsilon_decay()
        # visualizer.plot_performance_metrics()
        
    except FileNotFoundError:
        print(f"❌ Model file not found: {MODEL_PATH}")
        print("Please update MODEL_PATH with the correct path to your .pth file")
    except Exception as e:
        print(f"❌ Error: {e}")