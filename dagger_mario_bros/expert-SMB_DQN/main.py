from super_dqn.trainer import MarioTrainer
import glob
import os

# Γιατί μας τα ζάλιζε ένα gym...
import warnings
warnings.filterwarnings('ignore', category = UserWarning, module = 'gym')

def find_latest_model(models_dir = 'models', prefix = 'mario_model_'):
    '''Επιστρέφει το πιο πρόσφατο αρχείο μοντέλου με βάση το timestamp στο όνομα.'''
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_paths = glob.glob(os.path.join(base_dir, models_dir, f'{prefix}*.pth'))
    if not model_paths:
        raise FileNotFoundError(f'No model files found in {models_dir}')
    
    return max(model_paths, key = os.path.getmtime)

def main():
    choices = ['train', 'test', '1st_expert', '2nd_expert', 'dagger']
    user_input = input(f'Εκπαίδευση ή Δοκιμή ({"/".join(choices)}): ').strip().lower()
    if user_input not in choices:
        print(f'\nΠαρακαλώ εισάγετε: {" ή ".join(choices)}')
        print('Επιλογή ΜΗ έγκυρη. Τερματισμός προγράμματος...') 
        return
    
    trainer = MarioTrainer(world = '1', stage = '1', action_type = 'simple')
    parrent_dir = os.path.dirname(os.path.abspath(__file__))

    if user_input == 'train':
        trainer.train(episodes = 30000, save_freq = 5000, render = False)
    elif user_input == 'test':
        try:
            model_path = find_latest_model()
            # 1/0 # Sorry, but the mario_model_best, is truly
                # the best model (at least for now...)!!!!!
        except:
            model_path = os.path.join(parrent_dir, 'models', 'mario_model_best.pth')
        trainer.test(
            model_path, episodes = 1, render = True, show_controller = True
        )
    elif user_input == '1st_expert':
        model_path = os.path.join(parrent_dir, 'models', 'WORKING_MARIO_AGENT.pth')
        trainer.test(
            model_path, episodes = 1, render = True, show_controller = True
        )
    elif user_input == '2nd_expert':
        model_path = os.path.join(parrent_dir, 'models', 'ep30000_MARIO_EXPERT.pth')
        trainer.test(
            model_path, episodes = 1, render = True, show_controller = True
        )
    elif user_input == 'dagger':
        model_path = os.path.join(
            parrent_dir, '..', 'DAGGER', 'models_dagger',
            'mario_FLAG_iter77_ep9_3301_20250608_153057.pth'
        )
        trainer.test(
            model_path, episodes = 1, render = True, show_controller = True
        )
    
    return

if __name__ == '__main__':
    main()
