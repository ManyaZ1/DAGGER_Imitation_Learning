from super_dqn import MarioTrainer

# Γιατί μας τα ζάλιζε ένα gym...
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='gym')

# Example usage
if __name__ == "__main__":
    # Create trainer
    trainer = MarioTrainer(world='1', stage='1', action_type='simple')
    
    # # Train the agent
    # trainer.train(episodes=500, save_freq=50, render=False)
    
    # Test the trained model (uncomment after training)
    # temp = "models/mario_model_ep400_20250531_014110.pth"
    temp = "models/mario_model_ep250_20250531_013000.pth"
    trainer.test(temp, episodes=3, render=True)
