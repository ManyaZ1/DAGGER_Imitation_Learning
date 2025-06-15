import os
import sys

base_dir   = os.path.dirname(__file__)              
pkg_parent = os.path.abspath(os.path.join(base_dir, '..', 'expert-SMB_DQN'))
sys.path.insert(0, pkg_parent)   
super_dqn_path  = os.path.abspath(
    os.path.join(base_dir, '..', 'expert-SMB_DQN', 'super_dqn')
) # …/expert-SMB_DQN/super_dqn
sys.path.append(super_dqn_path) # add to PYTHONPATH

import warnings
warnings.filterwarnings('ignore', category = UserWarning, module = 'gym')

# Setup paths
sys.path.append(os.path.join(base_dir, '..', 'expert-SMB_DQN', 'super_dqn'))

# Imports
from trainer import MarioTrainer

def main():
    trainer = MarioTrainer(world = '1', stage = '1', action_type = 'simple')

    model_path = os.path.join( 
        base_dir, 'SUCCESS',
        # 'dagger_mario_iter27_20250609_131303.pth'
        'mario_FLAG_iter574_ep19_3425_20250609_072905.pth'
    )

    trainer.test(
        model_path, episodes = 1, render = True, show_controller = True,
        env_unresponsive = True
    )
    
    return

if __name__ == '__main__':
    main()
