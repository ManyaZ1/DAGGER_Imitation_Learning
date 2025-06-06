import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import torch

# Assuming your existing files are importable
base_dir        = os.path.dirname(__file__)              # …/CLONING
pkg_parent = os.path.abspath(os.path.join(base_dir, '..', 'expert-SMB_DQN'))
sys.path.insert(0, pkg_parent)   
super_dqn_path  = os.path.abspath(os.path.join(base_dir, '..',
                                               'expert-SMB_DQN',
                                               'super_dqn'))              # …/expert-SMB_DQN/super_dqn
sys.path.append(super_dqn_path)                                           # add to PYTHONPATH
# ───────────────────────────────────────────────────────────────────

from super_dqn.agent   import MarioAgent
from super_dqn.trainer import MarioTrainer

from behavior_cloning import (
    ExpertDemonstrationCollector, 
    BehaviorCloningAgent, 
    PartialObservationWrapper
)

class MarioBehaviorCloningExperiment:
    """Πλήρες experiment για Behavior Cloning στο Mario"""
    
    def __init__(self, world='1', stage='1', action_type='simple'):
        self.world = world
        self.stage = stage
        self.action_type = action_type
        
        # Setup trainer (που έχει το environment)
        self.trainer = MarioTrainer(world, stage, action_type)
        self.env = self.trainer.env
        self.n_actions = self.env.action_space.n
        self.state_shape = self.env.observation_space.shape
        
        print(f"Mario BC Experiment Setup:")
        print(f"World: {world}-{stage}")
        print(f"State shape: {self.state_shape}")
        print(f"Actions: {self.n_actions}")
    
    def step1_collect_expert_demonstrations(self, 
                                          expert_model_path,
                                          num_episodes=50,
                                          save_path=None):
        """Βήμα 1: Συλλογή expert demonstrations"""
        
        print("\n" + "="*60)
        print("ΒΗΜΑ 1: ΣΥΛΛΟΓΗ EXPERT DEMONSTRATIONS")
        print("="*60)
        
        # Load expert agent
        expert_agent = MarioAgent(self.state_shape, self.n_actions)
        expert_agent.load_model(expert_model_path)
        
        # Collect demonstrations
        collector = ExpertDemonstrationCollector(expert_agent, self.env)
        demonstrations = collector.collect_demonstrations(
            num_episodes=num_episodes,
            save_path=save_path
        )
        
        # Analyze demonstrations
        self._analyze_demonstrations(demonstrations)
        
        return demonstrations
    
    def step2_train_behavior_cloning_agents(self, 
                                          demonstrations_path,
                                          epochs=100):
        """Βήμα 2: Εκπαίδευση BC agents με διαφορετικά observation scenarios"""
        
        print("\n" + "="*60)
        print("ΒΗΜΑ 2: ΕΚΠΑΙΔΕΥΣΗ BEHAVIOR CLONING AGENTS")
        print("="*60)
        
        # Load demonstrations
        with open(demonstrations_path, 'rb') as f:
            demonstrations = pickle.load(f)
        
        # Define scenarios
        scenarios = {
            'full_state': None,
            'partial_obs (2/4 channels)': PartialObservationWrapper('partial'),
            'noisy_obs (σ=0.1)': PartialObservationWrapper('noisy', noise_level=0.1),
            'noisy_obs (σ=0.2)': PartialObservationWrapper('noisy', noise_level=0.2),
            'downsampled': PartialObservationWrapper('downsampled')
        }
        
        # Train BC agents για κάθε scenario
        bc_agents = {}
        training_results = {}
        
        for scenario_name, obs_wrapper in scenarios.items():
            print(f"\n--- Training BC Agent: {scenario_name} ---")
            
            # Create BC agent
            #bc_agent = BehaviorCloningAgent(self.state_shape, self.n_actions, lr=1e-4)
            # NEW (auto-detect true shape)
            # Use one transformed demo to determine the true input shape
            sample_state = demonstrations[0]['state']
            if obs_wrapper:
                sample_state = obs_wrapper.transform_observation(sample_state)

            input_shape = sample_state.shape
            bc_agent = BehaviorCloningAgent(input_shape, self.n_actions)
                        # Train
            bc_agent.train(
                demonstrations=demonstrations,
                observation_wrapper=obs_wrapper,
                epochs=epochs,
                batch_size=32,
                validation_split=0.2
            )
            
            # Save agent
            safe_name = scenario_name.lower()
            safe_name = safe_name.replace(" ", "_").replace("(", "").replace(")", "")
            safe_name = safe_name.replace("/", "_").replace("\\", "_")
            agent_path = f'bc_agent_{safe_name}.pth'
            #agent_path = f'bc_agent_{scenario_name.replace(" ", "_").replace("(", "").replace(")", "")}.pth'
            torch.save(bc_agent.network.state_dict(), agent_path)
            
            bc_agents[scenario_name] = bc_agent
            training_results[scenario_name] = {
                'final_loss': bc_agent.training_losses[-1],
                'final_val_acc': bc_agent.validation_accuracies[-1],
                'best_val_acc': max(bc_agent.validation_accuracies)
            }
            
            # Plot training progress
            bc_agent.plot_training_progress()
        
        return bc_agents, training_results
    
    def step3_evaluate_bc_agents(self, 
                                bc_agents, 
                                expert_model_path,
                                test_episodes=10):
        """Βήμα 3: Αξιολόγηση BC agents vs Expert"""
        
        print("\n" + "="*60)
        print("ΒΗΜΑ 3: ΑΞΙΟΛΟΓΗΣΗ BC AGENTS")
        print("="*60)
        
        # Load expert για comparison
        expert_agent = MarioAgent(self.state_shape, self.n_actions)
        expert_agent.load_model(expert_model_path)
        
        evaluation_results = {}
        
        # Test expert performance
        print("\n--- Testing Expert Agent ---")
        expert_scores = self._test_agent(expert_agent, test_episodes, "Expert")
        evaluation_results['Expert'] = {
            'scores': expert_scores,
            'mean': np.mean(expert_scores),
            'std': np.std(expert_scores)
        }
        
        # Test BC agents
        scenarios = {
            'full_state': None,
            'partial_obs': PartialObservationWrapper('partial'),
            'noisy_obs_01': PartialObservationWrapper('noisy', noise_level=0.1),
            'noisy_obs_02': PartialObservationWrapper('noisy', noise_level=0.2),
            'downsampled': PartialObservationWrapper('downsampled')
        }
        
        for scenario_name, obs_wrapper in scenarios.items():
            if scenario_name in bc_agents:
                print(f"\n--- Testing BC Agent: {scenario_name} ---")
                bc_scores = self._test_bc_agent(
                    bc_agents[scenario_name], 
                    obs_wrapper, 
                    test_episodes, 
                    scenario_name
                )
                
                evaluation_results[f'BC_{scenario_name}'] = {
                    'scores': bc_scores,
                    'mean': np.mean(bc_scores),
                    'std': np.std(bc_scores)
                }
        
        # Generate comparison plots
        self._plot_evaluation_results(evaluation_results)
        
        return evaluation_results
    
    def _test_agent(self, agent, episodes, agent_name):
        """Test standard RL agent"""
        scores = []
        
        for episode in range(episodes):
            #self.trainer.prev_x_pos = 40 # Reset position for each episode
            self.trainer.prev_x_pos = 40  # Reset position for each episode
            state = self.env.reset()
            total_reward = 0
            done = False
            steps = 0
            
            while not done and steps < 5000:  # Max steps limit
                action = agent.act(state, training=False)
                next_state, reward, done, info = self.env.step(action)
                
                # Apply same reward shaping as training
                shaped_reward = self.trainer.shape_reward(reward, info, done)
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
    
    def _test_bc_agent(self, bc_agent, obs_wrapper, episodes, agent_name):
        """Test behavior cloning agent"""
        scores = []
        
        for episode in range(episodes):
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
                
                action = bc_agent.act(observed_state, training=False)
                next_state, reward, done, info = self.env.step(action)
                
                # Apply same reward shaping
                shaped_reward = self.trainer.shape_reward(reward, info, done)
                total_reward += shaped_reward
                
                state = next_state
                steps += 1
            
            scores.append(total_reward)
            print(f"BC_{agent_name} Episode {episode+1}: Score = {total_reward:.2f}, "
                  f"Position = {info.get('x_pos', 0)}, Steps = {steps}")
        
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print(f"BC_{agent_name} Average: {mean_score:.2f} ± {std_score:.2f}")
        
        return scores
    
    def _analyze_demonstrations(self, demonstrations):
        """Analyze expert demonstrations"""
        actions = [demo['action'] for demo in demonstrations]
        action_counts = np.bincount(actions, minlength=self.n_actions)
        
        print(f"\nDemonstration Analysis:")
        print(f"Total state-action pairs: {len(demonstrations)}")
        print(f"Action distribution:")
        for i, count in enumerate(action_counts):
            percentage = (count / len(demonstrations)) * 100
            print(f"  Action {i}: {count} ({percentage:.1f}%)")
    
    def _plot_evaluation_results(self, results):
        """Plot comparison results"""
        agents = list(results.keys())
        means = [results[agent]['mean'] for agent in agents]
        stds = [results[agent]['std'] for agent in agents]
        
        plt.figure(figsize=(12, 6))
        
        # Bar plot με error bars
        x_pos = np.arange(len(agents))
        bars = plt.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)
        
        # Color expert differently
        for i, agent in enumerate(agents):
            if 'Expert' in agent:
                bars[i].set_color('red')
                bars[i].set_alpha(1.0)
        
        plt.xlabel('Agent Type')
        plt.ylabel('Average Score')
        plt.title('Behavior Cloning vs Expert Performance')
        plt.xticks(x_pos, agents, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(f'bc_evaluation_results_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_full_experiment(self, expert_model_path):
        """Εκτέλεση πλήρους experiment"""
        
        print("🎮 MARIO BEHAVIOR CLONING EXPERIMENT 🎮")
        print("="*60)
        
        # Paths
        demonstrations_path = f'mario_demonstrations_{self.world}_{self.stage}.pkl'
        
        # Step 1: Collect demonstrations
        demonstrations = self.step1_collect_expert_demonstrations(
            expert_model_path=expert_model_path,
            num_episodes=10,  # Increase for better performance 100
            save_path=demonstrations_path
        )
        
        # Step 2: Train BC agents
        bc_agents, training_results = self.step2_train_behavior_cloning_agents(
            demonstrations_path=demonstrations_path,
            epochs=100  # Increase for better convergence 150
        )
        
        # Step 3: Evaluate
        evaluation_results = self.step3_evaluate_bc_agents(
            bc_agents=bc_agents,
            expert_model_path=expert_model_path,
            test_episodes=20  # More episodes for statistical significance
        )
        
        print("\n" + "="*60)
        print("EXPERIMENT COMPLETED!")
        print("="*60)
        
        # Summary
        print("\nFINAL RESULTS SUMMARY:")
        for agent_name, result in evaluation_results.items():
            print(f"{agent_name}: {result['mean']:.2f} ± {result['std']:.2f}")
        
        return {
            'demonstrations': demonstrations,
            'bc_agents': bc_agents,
            'training_results': training_results,
            'evaluation_results': evaluation_results
        }

# Example usage
if __name__ == "__main__":
    # Initialize experiment
    experiment = MarioBehaviorCloningExperiment(world='1', stage='1', action_type='simple')
    
    # Run experiment (you need to provide path to your trained expert model)
    expert_model_path = 'ep30000_MARIO_EXPERT.pth'  # Adjust path
    
    try:
        results = experiment.run_full_experiment(expert_model_path)
        print("Experiment completed successfully!")
    except FileNotFoundError:
        print(f"Expert model not found at {expert_model_path}")
        print("Please train an expert model first using your existing trainer!")