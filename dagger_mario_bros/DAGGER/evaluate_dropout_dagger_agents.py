import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Dynamically add the parent path that contains super_dqn/
base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(base_dir, '..', 'expert-SMB_DQN'))
sys.path.insert(0, project_root)
from super_dqn.trainer import MarioTrainer
from dropout_dagger_agent import DropoutDaggerMarioAgent
from observation_wrapper import PartialObservationWrapper

class DropoutDaggerEvaluator:
    """Evaluator for comparing different DropoutDAgger agents"""
    
    def __init__(self, env):
        self.env = env
        self.state_shape = env.observation_space.shape
        self.n_actions = env.action_space.n
    
    def evaluate_agent(self, agent, obs_wrapper, num_episodes=10, agent_name="Agent"):
        """Evaluate a single agent"""
        scores = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            done = False
            steps = 0
            
            while not done and steps < 5000:
                # Transform observation if needed
                if obs_wrapper:
                    observed_state = obs_wrapper.transform_observation(state)
                else:
                    observed_state = state
                
                action = agent.act(observed_state)
                next_state, reward, done, info = self.env.step(action)
                
                # Use same reward shaping as training
                # Assuming MarioTrainer has shape_reward method
                if hasattr(self.env, 'shape_reward'):
                    shaped_reward = self.env.shape_reward(reward, info, done)
                else:
                    shaped_reward = reward
                
                total_reward += shaped_reward
                state = next_state
                steps += 1
            
            scores.append(total_reward)
            print(f"{agent_name} Episode {episode+1}: Score = {total_reward:.2f}, "
                  f"Position = {info.get('x_pos', 0)}, Steps = {steps}")
        
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print(f"{agent_name} Average: {mean_score:.2f} ± {std_score:.2f}")
        
        return scores
    
    def compare_agents(self, agents_dict, wrappers_dict, num_episodes=10):
        """Compare multiple DropoutDAgger agents"""
        
        print("="*60)
        print("DROPOUT DAGGER AGENTS EVALUATION")
        print("="*60)
        
        all_results = {}
        
        for agent_name, agent in agents_dict.items():
            wrapper = wrappers_dict.get(agent_name, None)
            
            print(f"\n--- Evaluating {agent_name} ---")
            scores = self.evaluate_agent(agent, wrapper, num_episodes, agent_name)
            
            all_results[agent_name] = {
                'scores': scores,
                'mean': np.mean(scores),
                'std': np.std(scores)
            }
        
        # Generate comparison plots
        self._plot_comparison_results(all_results)
        
        # Print summary
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        for agent_name, result in all_results.items():
            print(f"{agent_name}: {result['mean']:.2f} ± {result['std']:.2f}")
        
        return all_results
    
    def _plot_comparison_results(self, results):
        """Plot comparison results"""
        agents = list(results.keys())
        means = [results[agent]['mean'] for agent in agents]
        stds = [results[agent]['std'] for agent in agents]
        
        plt.figure(figsize=(12, 8))
        
        # Bar plot με error bars
        x_pos = np.arange(len(agents))
        bars = plt.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)
        
        # Color coding for different observation types
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i, bar in enumerate(bars):
            bar.set_color(colors[i % len(colors)])
        
        plt.xlabel('Agent Type')
        plt.ylabel('Average Score')
        plt.title('DropoutDAgger Agents Performance Comparison')
        plt.xticks(x_pos, agents, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(f'dropout_dagger_evaluation_results_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """Main evaluation script"""
    
    # Setup Environment
    trainer = MarioTrainer(world='1', stage='1', action_type='simple')
    env = trainer.env
    state_shape = env.observation_space.shape
    n_actions = env.action_space.n
    
    # Define all agents you want to test
    # NOTE: Update these paths to your actual trained models
    agent_files = {
        "full_state": "dropout_dagger_full_state.pth",
        "partial_obs": "dropout_dagger_partial_obs.pth", 
        "noisy_obs_01": "dropout_dagger_noisy_obs_σ=0.1.pth",
        "noisy_obs_02": "dropout_dagger_noisy_obs_σ=0.2.pth",
        "downsampled": "dropout_dagger_downsampled.pth"
    }
    
    wrappers = {
        "full_state": None,
        "partial_obs": PartialObservationWrapper('partial'),
        "noisy_obs_01": PartialObservationWrapper('noisy', noise_level=0.1),
        "noisy_obs_02": PartialObservationWrapper('noisy', noise_level=0.2),
        "downsampled": PartialObservationWrapper('downsampled')
    }
    
    # Load models
    agents = {}
    for name, filename in agent_files.items():
        results_dir = os.path.join(os.path.dirname(__file__), 'dropout_dagger_results')
        
        # Try to find model in results directory
        model_found = False
        for root, dirs, files in os.walk(results_dir):
            for file in files:
                if name in file and file.endswith('.pth'):
                    path = os.path.join(root, file)
                    model_found = True
                    break
            if model_found:
                break
        
        if not model_found:
            path = os.path.join(results_dir, filename)
        
        if not os.path.exists(path):
            print(f"⚠️  Model not found: {path}")
            print(f"    Skipping {name}")
            continue
        
        print(f"Loading: {path}")
        
        # Use the right input shape
        sample_state = env.reset()
        if wrappers[name]:
            sample_state = wrappers[name].transform_observation(sample_state)
        
        input_shape = sample_state.shape
        
        # Create agent with appropriate dropout rate (assuming 0.5 default)
        agent = DropoutDaggerMarioAgent(input_shape, n_actions, dropout_rate=0.5)
        
        try:
            agent.load_model(path)
            agents[name] = agent
            print(f"✅ Successfully loaded {name}")
        except Exception as e:
            print(f"❌ Failed to load {name}: {e}")
            continue
    
    if not agents:
        print("❌ No agents could be loaded. Make sure you have trained DropoutDAgger models.")
        print("    Run dropout_dagger_main.py first to train the models.")
        return
    
    # Evaluate all loaded agents
    evaluator = DropoutDaggerEvaluator(env)
    results = evaluator.compare_agents(agents, wrappers, num_episodes=10)
    
    # Save detailed results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f'dropout_dagger_detailed_results_{timestamp}.txt'
    
    with open(results_file, 'w') as f:
        f.write("DropoutDAgger Evaluation Results\n")
        f.write("="*50 + "\n\n")
        
        for agent_name, result in results.items():
            f.write(f"Agent: {agent_name}\n")
            f.write(f"Mean Score: {result['mean']:.2f}\n")
            f.write(f"Std Score: {result['std']:.2f}\n")
            f.write(f"Individual Scores: {result['scores']}\n")
            f.write("-"*30 + "\n")
    
    print(f"\n📊 Detailed results saved to: {results_file}")

if __name__ == "__main__":
    main()