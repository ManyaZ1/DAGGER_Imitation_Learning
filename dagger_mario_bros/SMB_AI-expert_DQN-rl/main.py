from super_dqn.trainer import MarioTrainer
import glob
import os

# Γιατί μας τα ζάλιζε ένα gym...
import warnings
warnings.filterwarnings('ignore', category = UserWarning, module = 'gym')

def find_latest_model(models_dir = 'models', prefix = 'mario_model_'):
    '''Επιστρέφει το πιο πρόσφατο αρχείο μοντέλου με βάση το timestamp στο όνομα.'''
    model_paths = glob.glob(os.path.join(models_dir, f'{prefix}*.pth'))
    if not model_paths:
        raise FileNotFoundError(f'No model files found in {models_dir}')
    
    return max(model_paths, key = os.path.getmtime)

def main():
    user_input = input('Εκπαίδευση ή Δοκιμή (train/test): ').strip().lower()
    if user_input not in ['train', 'test']:
        print("Παρακαλώ εισάγετε 'train' ή 'test'!")
        return
    
    trainer = MarioTrainer(world = '1', stage = '1', action_type = 'simple')

    if user_input == 'train':
        trainer.train(episodes = 40000, save_freq = 5000, render = False)
    elif user_input == 'test':
        model_path = find_latest_model()
        trainer.test(model_path, episodes = 3, render = True)

    return

if __name__ == '__main__':
    main()
