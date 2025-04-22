import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# --- Environment Setup ---
env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# --- Expert Policy ---
class Expert:
    def predict(self, state):
        # Access state as a numpy array, not a tuple
        return 0 if state[2] < 0 else 1  # Now works!

# --- BC/DAGGER Learner (Neural Network) ---
class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, x):
        return self.net(x)

# --- Train with Behavior Cloning ---
def train_bc(expert, num_episodes=10):
    states, actions = [], []
    for _ in range(num_episodes):
        obs, _ = env.reset()  # Unpack tuple (obs, info)
        done = False
        while not done:
            action = expert.predict(obs)  # Pass obs (numpy array)
            states.append(obs)
            actions.append(action)
            next_obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated  # New done logic
            obs = next_obs
    return states, actions

# --- Train with DAGGER ---
def train_dagger(expert, policy, num_iterations=5):
    optimizer = optim.Adam(policy.parameters())
    states, actions = [], []
    rewards = []

    for _ in range(num_iterations):
        # --- Rollout ---
        trajectory = []
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            trajectory.append(obs)
            with torch.no_grad():
                action = torch.argmax(policy(torch.FloatTensor(obs))).item()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            obs = next_obs
        
        rewards.append(total_reward)

        # --- Expert labeling + aggregation ---
        expert_actions = [expert.predict(s) for s in trajectory]
        states.extend(trajectory)
        actions.extend(expert_actions)

        # --- Train policy ---
        states_np = np.array(states, dtype=np.float32)
        actions_np = np.array(actions, dtype=np.int64)
        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(states_np),
            torch.from_numpy(actions_np)
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        
        for epoch in range(3):
            for batch_states, batch_actions in loader:
                logits = policy(batch_states)
                loss = nn.CrossEntropyLoss()(logits, batch_actions)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
    return (policy, rewards)

# --- Evaluate Policy (Fixed) ---
def evaluate(policy, num_episodes=10):
    total_reward = 0
    for _ in range(num_episodes):
        obs, _ = env.reset()  # Unpack tuple
        done = False
        while not done:
            action = torch.argmax(policy(torch.FloatTensor(obs))).item()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            obs = next_obs
    return total_reward / num_episodes

# --- Run Experiment ---
if __name__ == "__main__":
    expert = Expert()
    
    # Train BC
    (bc_states, bc_actions) = train_bc(expert)
    bc_policy = Policy()
    bc_optimizer = optim.Adam(bc_policy.parameters())
    
    bc_states_np = np.array(bc_states, dtype=np.float32)
    bc_actions_np = np.array(bc_actions, dtype=np.int64)

    bc_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(bc_states_np),
        torch.from_numpy(bc_actions_np)
    )
    bc_loader = torch.utils.data.DataLoader(bc_dataset, batch_size=32, shuffle=True)
    for epoch in range(5):
        for states, actions in bc_loader:
            logits = bc_policy(states)
            loss = nn.CrossEntropyLoss()(logits, actions)
            bc_optimizer.zero_grad()
            loss.backward()
            bc_optimizer.step()
    
    # Train DAGGER
    dagger_policy = Policy()
    dagger_policy, dagger_rewards = train_dagger(expert, dagger_policy)

    # --- Compare ---
    print(f"BC Average Reward: {evaluate(bc_policy, 30)}")
    print(f"DAGGER Average Reward: {evaluate(dagger_policy, 30)}")

    # --- Plot reward over iterations ---
    plt.plot(dagger_rewards, marker='o')
    plt.xlabel("DAGGER Iteration")
    plt.ylabel("Reward")
    plt.title("DAGGER Training Progress")
    plt.grid(True)
    plt.show()
