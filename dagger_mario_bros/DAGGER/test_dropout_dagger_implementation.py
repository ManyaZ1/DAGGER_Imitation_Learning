import os
import sys
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='gym')

# Imports
from dropout_dagger_trainer import DropoutDaggerTrainer, DropoutDaggerConfig
from dropout_dagger_agent import DropoutDaggerMarioAgent

def test_dropout_dagger_implementation():
    """Test DropoutDAgger implementation with minimal training"""
    
    print("🧪 Testing DropoutDAgger Implementation")
    print("="*50)
    
    base_dir = os.path.dirname(__file__)
    expert_model_path = os.path.join(
        base_dir, '..', 'expert-SMB_DQN', 'models', 'ep30000_MARIO_EXPERT.pth'
    )
    
    # Test different observation modes
    test_configs = [
        {'name': 'Full State', 'obs_type': None, 'noise': 0.0},
        {'name': 'Partial Obs', 'obs_type': 'partial', 'noise': 0.0},
        {'name': 'Noisy Obs', 'obs_type': 'noisy', 'noise': 0.1},
        {'name': 'Downsampled', 'obs_type': 'downsampled', 'noise': 0.0},
    ]
    
    for test_config in test_configs:
        print(f"\n📋 Testing: {test_config['name']}")
        print("-" * 30)
        
        try:
            config = DropoutDaggerConfig(
                observation_type=test_config['obs_type'],
                noise_level=test_config['noise'],
                dropout_rate=0.5,
                iterations=1,
                episodes_per_iter=2,  # Very short test
                training_batches_per_iter=5,
                expert_model_path=expert_model_path,
                render=False,
                world='1',
                stage='1',
                save_frequency=1
            )
            
            # Test trainer creation
            trainer = DropoutDaggerTrainer(config)
            print(f"✅ Trainer created successfully")
            print(f"   State shape: {trainer.state_shape}")
            print(f"   N actions: {trainer.n_actions}")
            print(f"   Dropout rate: {config.dropout_rate}")
            
            # Test learner agent
            learner = trainer.learner
            print(f"✅ Learner agent created")
            print(f"   Agent type: {type(learner).__name__}")
            print(f"   Dropout rate: {learner.dropout_rate}")
            
            # Test observation wrapper
            if trainer.observation_wrapper:
                print(f"✅ Observation wrapper: {trainer.observation_wrapper.obs_type}")
            else:
                print(f"✅ No observation wrapper (full state)")
            
            # Test a single episode run (without rendering)
            print(f"🎮 Running test episode...")
            test_episode_data = trainer._run_episode(0, 0)
            print(f"✅ Episode completed:")
            print(f"   Reward: {test_episode_data['reward']:.2f}")
            print(f"   Steps: {test_episode_data['steps']}")
            print(f"   Expert agreement: {test_episode_data['expert_agreement']:.3f}")
            print(f"   Memory size: {len(learner.dagger_memory)}")
            
            # Test training
            if len(learner.dagger_memory) >= learner.batch_size:
                print(f"🏋️ Testing training...")
                loss = trainer._train_learner_immediate(num_batches=2)
                print(f"✅ Training successful, loss: {loss:.6f}")
            else:
                print(f"⚠️  Not enough data for training (need {learner.batch_size}, have {len(learner.dagger_memory)})")
            
            # Test model saving/loading
            print(f"💾 Testing model save/load...")
            test_model_path = f'/tmp/test_dropout_dagger_{test_config["name"].replace(" ", "_").lower()}.pth'
            learner.save_model(test_model_path)
            
            # Create new agent and load
            test_agent = DropoutDaggerMarioAgent(trainer.state_shape, trainer.n_actions, config.dropout_rate)
            test_agent.load_model(test_model_path)
            print(f"✅ Model save/load successful")
            
            # Clean up
            os.remove(test_model_path)
            
        except Exception as e:
            print(f"❌ Error in {test_config['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*50)
    print("🎉 DropoutDAgger Implementation Test Complete!")

def test_network_dropout():
    """Test the dropout network functionality"""
    
    print("\n🧪 Testing Dropout Network Functionality")
    print("="*50)
    
    import torch
    import numpy as np
    from dropout_dagger_agent import DropoutDQN
    
    # Test different input shapes
    test_shapes = [
        (4, 84, 84),  # Full state
        (2, 84, 84),  # Partial state
        (4, 42, 42),  # Downsampled
    ]
    
    for shape in test_shapes:
        print(f"\n📐 Testing shape: {shape}")
        
        try:
            # Create network
            network = DropoutDQN(shape, n_actions=7, dropout_rate=0.5)
            print(f"✅ Network created")
            
            # Test forward pass in training mode
            network.train()
            dummy_input = torch.randn(4, *shape)  # Batch of 4
            output_train = network(dummy_input)
            print(f"✅ Training forward pass: {output_train.shape}")
            
            # Test forward pass in eval mode
            network.eval()
            output_eval = network(dummy_input)
            print(f"✅ Eval forward pass: {output_eval.shape}")
            
            # Test that dropout makes a difference
            network.train()
            output_train_1 = network(dummy_input)
            output_train_2 = network(dummy_input)
            
            # Outputs should be different due to dropout (with high probability)
            difference = torch.abs(output_train_1 - output_train_2).mean().item()
            if difference > 0.001:
                print(f"✅ Dropout working (difference: {difference:.6f})")
            else:
                print(f"⚠️  Dropout might not be working (difference: {difference:.6f})")
            
        except Exception as e:
            print(f"❌ Error with shape {shape}: {e}")

def main():
    """Main test function"""
    
    print("🧪 DROPOUT DAGGER IMPLEMENTATION TESTER 🧪")
    print("This script tests the DropoutDAgger implementation")
    print("without running full training.\n")
    
    # Test network functionality
    test_network_dropout()
    
    # Test implementation
    test_dropout_dagger_implementation()
    
    print("\n🎯 To run full training, use: python dropout_dagger_main.py")
    print("🎯 To test trained models, use: python dropout_dagger_test_interactive.py")

if __name__ == '__main__':
    main()