import os
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='gym')

# Imports
from dropout_dagger_trainer import DropoutDaggerTrainer, DropoutDaggerConfig

def train_dropout_dagger_full_state():
    """Train DropoutDAgger agent with full state observations"""
    
    base_dir = os.path.dirname(__file__)
    expert_model_path = os.path.join(
        base_dir, '..', 'expert-SMB_DQN', 'models', 'ep30000_MARIO_EXPERT.pth'
    )
    
    config = DropoutDaggerConfig(
        observation_type=None,  # Full state
        dropout_rate=0.5,
        iterations=3,
        episodes_per_iter=10,
        training_batches_per_iter=50,
        expert_model_path=expert_model_path,
        render=False,
        world='1',
        stage='1',
        save_frequency=1
    )
    
    print("="*60)
    print("TRAINING DROPOUT DAGGER - FULL STATE")
    print("="*60)
    print(f"Dropout rate: {config.dropout_rate}")
    print(f"Observation type: Full State")
    print(f"Iterations: {config.iterations}")
    print(f"Episodes per iteration: {config.episodes_per_iter}")
    print("="*60)
    
    trainer = DropoutDaggerTrainer(config)
    results = trainer.train()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED!")
    print("="*60)
    print(f"Best model: {results['best_model_path']}")
    print(f"Best score: {results['best_score']:.2f}")
    print(f"Total episodes: {results['total_episodes']}")
    
    return results

def train_dropout_dagger_partial_obs():
    """Train DropoutDAgger agent with partial observations (2/4 channels)"""
    
    base_dir = os.path.dirname(__file__)
    expert_model_path = os.path.join(
        base_dir, '..', 'expert-SMB_DQN', 'models', 'ep30000_MARIO_EXPERT.pth'
    )
    
    config = DropoutDaggerConfig(
        observation_type='partial',
        dropout_rate=0.5,
        iterations=3,
        episodes_per_iter=10,
        training_batches_per_iter=50,
        expert_model_path=expert_model_path,
        render=False,
        world='1',
        stage='1',
        save_frequency=1
    )
    
    print("="*60)
    print("TRAINING DROPOUT DAGGER - PARTIAL OBSERVATIONS")
    print("="*60)
    print(f"Dropout rate: {config.dropout_rate}")
    print(f"Observation type: Partial (2/4 channels)")
    print(f"Iterations: {config.iterations}")
    print(f"Episodes per iteration: {config.episodes_per_iter}")
    print("="*60)
    
    trainer = DropoutDaggerTrainer(config)
    results = trainer.train()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED!")
    print("="*60)
    print(f"Best model: {results['best_model_path']}")
    print(f"Best score: {results['best_score']:.2f}")
    print(f"Total episodes: {results['total_episodes']}")
    
    return results

def train_dropout_dagger_noisy_01():
    """Train DropoutDAgger agent with noisy observations (σ=0.1)"""
    
    base_dir = os.path.dirname(__file__)
    expert_model_path = os.path.join(
        base_dir, '..', 'expert-SMB_DQN', 'models', 'ep30000_MARIO_EXPERT.pth'
    )
    
    config = DropoutDaggerConfig(
        observation_type='noisy',
        noise_level=0.1,
        dropout_rate=0.5,
        iterations=3,
        episodes_per_iter=10,
        training_batches_per_iter=50,
        expert_model_path=expert_model_path,
        render=False,
        world='1',
        stage='1',
        save_frequency=1
    )
    
    print("="*60)
    print("TRAINING DROPOUT DAGGER - NOISY OBSERVATIONS (σ=0.1)")
    print("="*60)
    print(f"Dropout rate: {config.dropout_rate}")
    print(f"Observation type: Noisy (σ=0.1)")
    print(f"Iterations: {config.iterations}")
    print(f"Episodes per iteration: {config.episodes_per_iter}")
    print("="*60)
    
    trainer = DropoutDaggerTrainer(config)
    results = trainer.train()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED!")
    print("="*60)
    print(f"Best model: {results['best_model_path']}")
    print(f"Best score: {results['best_score']:.2f}")
    print(f"Total episodes: {results['total_episodes']}")
    
    return results

def train_dropout_dagger_noisy_02():
    """Train DropoutDAgger agent with noisy observations (σ=0.2)"""
    
    base_dir = os.path.dirname(__file__)
    expert_model_path = os.path.join(
        base_dir, '..', 'expert-SMB_DQN', 'models', 'ep30000_MARIO_EXPERT.pth'
    )
    
    config = DropoutDaggerConfig(
        observation_type='noisy',
        noise_level=0.2,
        dropout_rate=0.5,
        iterations=3,
        episodes_per_iter=10,
        training_batches_per_iter=50,
        expert_model_path=expert_model_path,
        render=False,
        world='1',
        stage='1',
        save_frequency=1
    )
    
    print("="*60)
    print("TRAINING DROPOUT DAGGER - NOISY OBSERVATIONS (σ=0.2)")
    print("="*60)
    print(f"Dropout rate: {config.dropout_rate}")
    print(f"Observation type: Noisy (σ=0.2)")
    print(f"Iterations: {config.iterations}")
    print(f"Episodes per iteration: {config.episodes_per_iter}")
    print("="*60)
    
    trainer = DropoutDaggerTrainer(config)
    results = trainer.train()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED!")
    print("="*60)
    print(f"Best model: {results['best_model_path']}")
    print(f"Best score: {results['best_score']:.2f}")
    print(f"Total episodes: {results['total_episodes']}")
    
    return results

def train_dropout_dagger_downsampled():
    """Train DropoutDAgger agent with downsampled observations"""
    
    base_dir = os.path.dirname(__file__)
    expert_model_path = os.path.join(
        base_dir, '..', 'expert-SMB_DQN', 'models', 'ep30000_MARIO_EXPERT.pth'
    )
    
    config = DropoutDaggerConfig(
        observation_type='downsampled',
        dropout_rate=0.5,
        iterations=3,
        episodes_per_iter=10,
        training_batches_per_iter=50,
        expert_model_path=expert_model_path,
        render=False,
        world='1',
        stage='1',
        save_frequency=1
    )
    
    print("="*60)
    print("TRAINING DROPOUT DAGGER - DOWNSAMPLED OBSERVATIONS")
    print("="*60)
    print(f"Dropout rate: {config.dropout_rate}")
    print(f"Observation type: Downsampled")
    print(f"Iterations: {config.iterations}")
    print(f"Episodes per iteration: {config.episodes_per_iter}")
    print("="*60)
    
    trainer = DropoutDaggerTrainer(config)
    results = trainer.train()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED!")
    print("="*60)
    print(f"Best model: {results['best_model_path']}")
    print(f"Best score: {results['best_score']:.2f}")
    print(f"Total episodes: {results['total_episodes']}")
    
    return results

def main():
    """Main function to train all DropoutDAgger variants"""
    
    print("🎮 DROPOUT DAGGER TRAINING SUITE 🎮")
    print("This will train DropoutDAgger agents with all observation modes")
    print("\nAvailable training modes:")
    print("1. Full State")
    print("2. Partial Observations (2/4 channels)")
    print("3. Noisy Observations (σ=0.1)")
    print("4. Noisy Observations (σ=0.2)")
    print("5. Downsampled Images")
    print("6. Train All")
    
    choice = input("\nSelect training mode (1-6): ").strip()
    
    try:
        if choice == '1':
            train_dropout_dagger_full_state()
        elif choice == '2':
            train_dropout_dagger_partial_obs()
        elif choice == '3':
            train_dropout_dagger_noisy_01()
        elif choice == '4':
            train_dropout_dagger_noisy_02()
        elif choice == '5':
            train_dropout_dagger_downsampled()
        elif choice == '6':
            print("\n🚀 Training all DropoutDAgger variants...")
            results = {}
            
            print("\n1/5: Training Full State...")
            results['full_state'] = train_dropout_dagger_full_state()
            
            print("\n2/5: Training Partial Observations...")
            results['partial_obs'] = train_dropout_dagger_partial_obs()
            
            print("\n3/5: Training Noisy σ=0.1...")
            results['noisy_01'] = train_dropout_dagger_noisy_01()
            
            print("\n4/5: Training Noisy σ=0.2...")
            results['noisy_02'] = train_dropout_dagger_noisy_02()
            
            print("\n5/5: Training Downsampled...")
            results['downsampled'] = train_dropout_dagger_downsampled()
            
            print("\n🎉 ALL TRAINING COMPLETED!")
            print("\n=== SUMMARY ===")
            for mode, result in results.items():
                print(f"{mode}: Best Score = {result['best_score']:.2f}")
                
        else:
            print("Invalid choice. Please select 1-6.")
            return
            
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()