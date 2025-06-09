import os
import sys

# Γιατί μας τα ζάλιζε ένα gym...
import warnings
warnings.filterwarnings('ignore', category = UserWarning, module = 'gym')

# Setup paths
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_dir, '..', 'expert-SMB_DQN', 'super_dqn'))

# Imports
from dagger_agent import DaggerMarioAgent
from trainer import MarioTrainer

def main():
    trainer = MarioTrainer(world = '1', stage = '1', action_type = 'simple')

    # 'mario_FLAG_iter474_ep5_3394_20250609_053224.pth' καθεται διπλα στη σημαια και αραζει
    model_path = os.path.join( 
        base_dir, 'models_dagger',
        'dagger_mario_iter25_20250609_130732.pth'
    )

    trainer.test(
        model_path, episodes = 1, render = True, show_controller = True,
        env_unresponsive = True
    )
    
    return

if __name__ == '__main__':
    main()
