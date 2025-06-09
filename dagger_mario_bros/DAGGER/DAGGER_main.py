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
    #'mario_FLAG_iter192_ep12_3416_20250609_182914.pth'σκαλα
    model_path = os.path.join( 
        base_dir, 'models_dagger', 'mario_FLAG_iter248_ep12_3419_20250609_194001.pth'#'mario_FLAG_iter574_ep19_3425_20250609_072905-success.pth'
        #'mario_FLAG_iter105_ep12_3346_20250608_154904.pth'
    )

    trainer.test(
        model_path, episodes = 1, render = True, show_controller = True,
        env_unresponsive = True
    )
    
    return

if __name__ == '__main__':
    main()
