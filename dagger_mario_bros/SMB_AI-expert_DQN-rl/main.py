from super_dqn.trainer import MarioTrainer

# Γιατί μας τα ζάλιζε ένα gym...
import warnings
warnings.filterwarnings('ignore', category = UserWarning, module = 'gym')

def main():
    trainer = MarioTrainer(world = '1', stage = '1', action_type = 'simple')
    
    # trainer.train(episodes = 10000, save_freq = 1000, render = False)
    
    # temp = 'models/mario_model_ep400_20250531_014110.pth'
    # temp = 'models/mario_model_ep250_20250531_013000.pth'
    temp = 'models/mario_model_ep4000_20250531_101627.pth'
    trainer.test(temp, episodes = 3, render = True)

    return

if __name__ == '__main__':
    main()
