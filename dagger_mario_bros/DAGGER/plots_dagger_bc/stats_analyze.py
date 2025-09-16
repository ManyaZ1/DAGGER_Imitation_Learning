import numpy as np
import os
import re

def get_all_metrics(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    best_reward      = float(re.search(r'Best Episode Reward:\s*([\d.]+)', content).group(1))
    final_reward     = float(re.search(r'Final Iteration Reward:\s*([\d.]+)', content).group(1))
    avg_reward       = float(re.search(r'Average Episode Reward:\s*([\d.]+)', content).group(1))
    expert_agreement = float(re.search(r'Final Expert Agreement:\s*([\d.]+)', content).group(1))
    final_loss       = float(re.search(r'Final Training Loss:\s*([\d.]+)', content).group(1))
    
    return best_reward, final_reward, avg_reward, expert_agreement, final_loss

def calc_stats(values):
    return {
        'median': np.median(values),
        'q25':    np.percentile(values, 25),
        'q75':    np.percentile(values, 75),
        'q10':    np.percentile(values, 10),
        'q90':    np.percentile(values, 90),
        'std':    np.std(values),
        'min':    np.min(values),
        'max':    np.max(values)
    }

def analyze_experiment(exp_dir):
    files = [f for f in os.listdir(exp_dir) if f.startswith('training_summary_')]
    best_rewards, final_rewards, avg_rewards, agreements, losses = [], [], [], [], []
    completed_runs = 0
    
    for file in files:
        best, final, avg, agreement, loss = get_all_metrics(os.path.join(exp_dir, file))
        best_rewards.append(best)
        final_rewards.append(final)
        avg_rewards.append(avg)
        agreements.append(agreement)
        losses.append(loss)
        if best >= 3336:
            # Βρέθηκε πειραματικά ότι είναι το
            # κατώφλι για την ολοκλήρωση της πίστας!
            completed_runs += 1
    
    return {
        'best':            calc_stats(best_rewards),
        'final':           calc_stats(final_rewards),
        'avg':             calc_stats(avg_rewards),
        'agreement':       calc_stats(agreements),
        'loss':            calc_stats(losses),
        'completion_rate': completed_runs / len(files) * 100,
        'total_runs':      len(files)
    }

def print_metric_comparison(name, plain_stats):
    print(f"\n* {name.upper()}:")
    print("-" * 50)
    print(f"  Median:  {plain_stats['median']:.2f}")
    print(f"  25%-75%: {plain_stats['q25']:.2f} - {plain_stats['q75']:.2f}")
    print(f"  10%-90%: {plain_stats['q10']:.2f} - {plain_stats['q90']:.2f}")
    print(f"  Min-Max: {plain_stats['min']:.2f} - {plain_stats['max']:.2f}")

def main():
    base_dir = r"c:\Users\nick1\Documents\GitHub\DAGGER_Imitation_Learning\dagger_mario_bros\DAGGER\plots_dagger_bc"
    
    plain = analyze_experiment(base_dir)
    
    print("="*55)
    print("DAGGER with BC warmup (100 iters - 10 episodes) RESULTS")
    print("="*55)

    print_metric_comparison("Best Reward", plain['best'])
    print_metric_comparison("Final Reward", plain['final'])
    print_metric_comparison("Average Reward", plain['avg'])
    print_metric_comparison("Expert Agreement", plain['agreement'])
    print_metric_comparison("Final Loss", plain['loss'])

    print(f"\nSTAGE COMPLETION:")
    print("-" * 50)
    print(f"DAGGER with BC warmup (100 iters - 10 episodes): {plain['completion_rate']:.1f}% ({plain['total_runs']} runs)")

if __name__ == "__main__":
    main()
