import os,sys
import torch

# Dynamically add the parent path that contains super_dqn/
base_dir     = os.path.dirname(os.path.abspath(__file__))               # path to this file
project_root = os.path.abspath(os.path.join(base_dir, '..', 'expert-SMB_DQN'))
sys.path.insert(0, project_root)  
from super_dqn.trainer import MarioTrainer
from behavior_cloning import BehaviorCloningAgent, PartialObservationWrapper, BehaviorCloningEvaluator

# -- Setup Environment --
trainer = MarioTrainer(world='1', stage='1', action_type='simple')
env = trainer.env
state_shape = env.observation_space.shape
n_actions   = env.action_space.n

# -- Define all agents you want to test --
agent_files = {
    "full_state":           "bc_agent_full_state.pth",
    "partial_obs":          "bc_agent_partial_obs_2_4_channels.pth",
    "noisy_obs_01":         "bc_agent_noisy_obs_σ=0.1.pth",
    "noisy_obs_02":         "bc_agent_noisy_obs_σ=0.2.pth",
    "downsampled":          "bc_agent_downsampled.pth"
}

wrappers = {
    "full_state":    None,
    "partial_obs":   PartialObservationWrapper('partial'),
    "noisy_obs_01":  PartialObservationWrapper('noisy', noise_level=0.1),
    "noisy_obs_02":  PartialObservationWrapper('noisy', noise_level=0.2),
    "downsampled":   PartialObservationWrapper('downsampled')
}

# -- Load models --
agents = {}
for name, filename in agent_files.items():
    #path = os.path.join("models", filename)  # adjust path if different
    path = filename  # Models are in the current folder

    print(f"Loading: {path}")

    # Use the right input shape
    sample_state = env.reset()
    if wrappers[name]:
        sample_state = wrappers[name].transform_observation(sample_state)
    input_shape = sample_state.shape

    agent = BehaviorCloningAgent(input_shape, n_actions)
    #agent.load_model(path)
    try:
        agent.load_model(path)  # works if it was saved with save_model()
    except KeyError:
        print(f"Fallback: loading weights only from {path}")
        state_dict = torch.load(path, map_location=agent.device)
        agent.network.load_state_dict(state_dict)
        agents[name] = agent

# -- Evaluate all loaded agents --
evaluator = BehaviorCloningEvaluator(env)
evaluator.compare_agents(agents, wrappers, num_episodes=10)
