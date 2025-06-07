import os
import torch
import sys

# Add path to import from super_dqn and use BC classes
base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(base_dir, '..', 'expert-SMB_DQN'))
sys.path.insert(0, project_root)

from super_dqn.trainer import MarioTrainer
from behavior_cloning import BehaviorCloningAgent, PartialObservationWrapper
from super_dqn.visual_utils import MarioRenderer

# All saved agents (add/remove as needed)
agent_choices = {
    "1": ("full_state",        "bc_agent_full_state.pth",         None),
    "2": ("partial_obs",       "bc_agent_partial_obs_2_4_channels.pth", PartialObservationWrapper("partial")),
    "3": ("noisy_obs_σ=0.1",   "bc_agent_noisy_obs_σ=0.1.pth",    PartialObservationWrapper("noisy", noise_level=0.1)),
    "4": ("noisy_obs_σ=0.2",   "bc_agent_noisy_obs_σ=0.2.pth",    PartialObservationWrapper("noisy", noise_level=0.2)),
    "5": ("downsampled",       "bc_agent_downsampled.pth",        PartialObservationWrapper("downsampled")),
}

def main():
    print("🎮 Επιλέξτε έναν BC Agent για δοκιμή:")
    for key, (name, _, _) in agent_choices.items():
        print(f"[{key}] {name}")

    choice = input("Επιλογή agent: ").strip()
    if choice not in agent_choices:
        print("❌ Άκυρη επιλογή.")
        return

    name, filename, wrapper = agent_choices[choice]
    print(f"\n🔁 Φόρτωση agent: {name} ({filename})")

    # Initialize environment
    trainer = MarioTrainer(world='1', stage='1', action_type='simple')
    env = trainer.env
    state = env.reset()
    if wrapper:
        sample_state = wrapper.transform_observation(state)
    else:
        sample_state = state
    input_shape = sample_state.shape
    n_actions = env.action_space.n

    # Load agent
    agent = BehaviorCloningAgent(input_shape, n_actions)
    try:
        agent.load_model(filename)
    except KeyError:
        print("⚠️  Loading weights only (raw state_dict).")
        state_dict = torch.load(filename, map_location=agent.device)
        agent.network.load_state_dict(state_dict)

    # Render
    print("🎬 Παρακολούθηση του agent...")
    renderer = MarioRenderer(env, scale=3.0)
    total_reward = 0
    done = False
    steps = 0

    while not done and steps < 5000:
        env.render()
        renderer.render()

        if wrapper:
            obs = wrapper.transform_observation(state)
        else:
            obs = state

        action = agent.act(obs, training=False)
        state, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1

    print(f"\n✅ Ολοκληρώθηκε! Score: {total_reward:.1f}, X: {info.get('x_pos', 0)}, Flag: {info.get('flag_get', False)}")

if __name__ == "__main__":
    main()
