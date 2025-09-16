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

def print_metric_comparison(name, plain_stats, bc_stats):
    print(f"\n* {name.upper()}:")
    print("-" * 50)
    print(f"Plain DAGGER:")
    print(f"  Median:  {plain_stats['median']:.2f}")
    print(f"  25%-75%: {plain_stats['q25']:.2f} - {plain_stats['q75']:.2f}")
    print(f"  10%-90%: {plain_stats['q10']:.2f} - {plain_stats['q90']:.2f}")
    print(f"  Min-Max: {plain_stats['min']:.2f} - {plain_stats['max']:.2f}")
    
    print(f"\nDAGGER + BC Warmup:")
    print(f"  Median:  {bc_stats['median']:.2f}")
    print(f"  25%-75%: {bc_stats['q25']:.2f} - {bc_stats['q75']:.2f}")
    print(f"  10%-90%: {bc_stats['q10']:.2f} - {bc_stats['q90']:.2f}")
    print(f"  Min-Max: {bc_stats['min']:.2f} - {bc_stats['max']:.2f}")
    
    improvement = ((bc_stats['median'] - plain_stats['median']) / plain_stats['median']) * 100
    print(f"\n-> Improvement: {improvement:+.1f}%")

def main():
    base_dir = r"c:\Users\nick1\Documents\GitHub\DAGGER_Imitation_Learning\dagger_mario_bros\DAGGER\statistics"
    
    plain = analyze_experiment(os.path.join(base_dir, "data-plain_dagger"))
    bc_warmup = analyze_experiment(os.path.join(base_dir, "data-bc_warmup_dagger"))
    
    print("="*40)
    print("DAGGER Vs. BC-WARMUP DAGGER - COMPARISON")
    print("="*40)
    
    print_metric_comparison("Best Reward", plain['best'], bc_warmup['best'])
    print_metric_comparison("Final Reward", plain['final'], bc_warmup['final'])
    print_metric_comparison("Average Reward", plain['avg'], bc_warmup['avg'])
    print_metric_comparison("Expert Agreement", plain['agreement'], bc_warmup['agreement'])
    print_metric_comparison("Final Loss", plain['loss'], bc_warmup['loss'])
    
    print(f"\nSTAGE COMPLETION:")
    print("-" * 50)
    print(f"Plain DAGGER:      {plain['completion_rate']:.1f}% ({plain['total_runs']} runs)")
    print(f"BC Warmup DAGGER: {bc_warmup['completion_rate']:.1f}% ({bc_warmup['total_runs']} runs)")
    print(f"Improvement:      {(bc_warmup['completion_rate'] - plain['completion_rate']):+.1f}%")

if __name__ == "__main__":
    main()
