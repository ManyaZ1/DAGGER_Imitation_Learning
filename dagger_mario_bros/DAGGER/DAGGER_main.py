import os
import sys

base_dir   = os.path.dirname(__file__)              

import warnings
warnings.filterwarnings('ignore', category = UserWarning, module = 'gym')

# Imports
from dagger_trainer import DaggerTrainer, DaggerConfig

def main():
    config = DaggerConfig(
        observation_type          = 'partial',
        iterations                = 1,
        episodes_per_iter         = 1,
        training_batches_per_iter = 1,
        expert_model_path         = ' ',
        render                    = True,
        only_for_testing          = True
    )
    trainerd = DaggerTrainer(config)

    model_path = os.path.join( 
        base_dir, 'SUCCESS',
        # 'dagger_mario_iter27_20250609_131303.pth'
        'mario_FLAG_iter4_ep2_3437_BC_warmup20250616_225237_partial.pth'
    )

    trainerd.test(
        model_path, episodes = 1, render = True, show_controller = True,
        env_unresponsive = True
    )
    
    return

if __name__ == '__main__':
    main()
