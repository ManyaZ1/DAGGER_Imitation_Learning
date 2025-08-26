import os

base_dir = os.path.dirname(__file__)              

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='gym')

# Imports
from dropout_dagger_trainer import DropoutDaggerTrainer, DropoutDaggerConfig

def main():
    config = DropoutDaggerConfig(
        observation_type='partial',
        dropout_rate=0.5,
        iterations=1,
        episodes_per_iter=1,
        training_batches_per_iter=1,
        expert_model_path=' ',
        render=True,
        only_for_testing=True
    )
    trainer = DropoutDaggerTrainer(config)

    # This should be updated with actual trained model path
    model_path = os.path.join( 
        base_dir, 'dropout_dagger_results',
        # Update this path when you have trained models
        'mario_FLAG_iter1_ep1_XXXX_YYYYMMDD_HHMMSS_partial_dropout0.5.pth'
    )

    # Test the model (similar to the original DAGGER test)
    trainer.test(
        model_path, episodes=1, render=True, show_controller=True,
        observation_wrapper=trainer.observation_wrapper,
        env_unresponsive=True
    )
    
    return

if __name__ == '__main__':
    print("DropoutDAgger Testing Script")
    print("Update the model_path in the script to point to your trained model.")
    main()