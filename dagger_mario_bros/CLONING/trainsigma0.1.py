from behavior_cloning import BehaviorCloningAgent, PartialObservationWrapper
import pickle
import torch
import os
# Load demonstrations
base_dir        = os.path.dirname(__file__)  
fp=os.path.join(base_dir,"models","mario_demonstrations_1_1.pkl")
with open(fp, 'rb') as f:
    demonstrations = pickle.load(f)

# Apply the noisy observation wrapper (σ = 0.1)
wrapper = PartialObservationWrapper('noisy', noise_level=0.1)

# Get transformed input shape
sample_state = wrapper.transform_observation(demonstrations[0]['state'])
input_shape = sample_state.shape
n_actions = 7  # or get from env: experiment.env.action_space.n

# Train the agent
bc_agent = BehaviorCloningAgent(input_shape, n_actions)
bc_agent.train(
    demonstrations=demonstrations,
    observation_wrapper=wrapper,
    epochs=200,  # ← your custom number of epochs here
    batch_size=32,
    validation_split=0.2
)

# Save the model
new_dir = os.path.join(base_dir, "model_sigma0.1_200epochs")
os.makedirs(new_dir, exist_ok=True)

model_path = os.path.join(new_dir, "bc_agent.pth")
torch.save(bc_agent.network.state_dict(), model_path)
