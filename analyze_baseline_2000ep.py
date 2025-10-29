"""
Baseline Ablation Study 결과 분석 스크립트
"""

import pandas as pd
import numpy as np
import glob
from pathlib import Path

# Results directory
results_dir = Path("ablation_results/baseline_20251029_165119")
seeds = [42, 123, 456, 789, 1024]

print("="*80)
print("BASELINE ABLATION STUDY RESULTS")
print("="*80)
print()

# Collect A3C results
a3c_results = []
for seed in seeds:
    seed_dir = results_dir / f"seed_{seed}" / "a3c"
    summary_file = seed_dir / "summary_global.csv"

    if summary_file.exists():
        df = pd.read_csv(summary_file)
        final_row = df.iloc[-1]

        a3c_results.append({
            'seed': seed,
            'reward': final_row['reward'],
            'policy_loss': final_row['policy_loss'],
            'value_loss': final_row['value_loss'],
            'episode': final_row['episode']
        })

# Collect Individual results
ind_results = []
for seed in seeds:
    seed_dir = results_dir / f"seed_{seed}" / "individual"

    seed_workers = []
    for worker_id in range(5):
        summary_file = seed_dir / f"summary_Individual_{worker_id}.csv"

        if summary_file.exists():
            df = pd.read_csv(summary_file)
            final_row = df.iloc[-1]

            seed_workers.append({
                'seed': seed,
                'worker_id': worker_id,
                'reward': final_row['reward'],
                'policy_loss': final_row['policy_loss'],
                'value_loss': final_row['value_loss'],
            })

    # Average across workers for this seed
    if seed_workers:
        avg_reward = np.mean([w['reward'] for w in seed_workers])
        avg_value_loss = np.mean([w['value_loss'] for w in seed_workers])
        avg_policy_loss = np.mean([w['policy_loss'] for w in seed_workers])
        std_reward = np.std([w['reward'] for w in seed_workers])

        ind_results.append({
            'seed': seed,
            'avg_reward': avg_reward,
            'std_reward': std_reward,
            'avg_policy_loss': avg_policy_loss,
            'avg_value_loss': avg_value_loss,
            'workers': seed_workers
        })

# Convert to DataFrames
df_a3c = pd.DataFrame(a3c_results)
df_ind = pd.DataFrame(ind_results)

# Print A3C Results
print("[1] A3C Global Model Results")
print("-" * 80)
for _, row in df_a3c.iterrows():
    print(f"  Seed {int(row['seed']):4d}: Reward = {row['reward']:6.2f}, "
          f"Value Loss = {row['value_loss']:8.2f}, "
          f"Policy Loss = {row['policy_loss']:8.4f}")

print()
print(f"  Mean Reward    : {df_a3c['reward'].mean():.2f} ± {df_a3c['reward'].std():.2f}")
print(f"  Mean Value Loss: {df_a3c['value_loss'].mean():.2f} ± {df_a3c['value_loss'].std():.2f}")
print(f"  Variance       : {df_a3c['reward'].var():.4f}")
print()

# Print Individual Results
print("[2] Individual Workers Results (Averaged per seed)")
print("-" * 80)
for _, row in df_ind.iterrows():
    print(f"  Seed {int(row['seed']):4d}: Avg Reward = {row['avg_reward']:6.2f} ± {row['std_reward']:5.2f}, "
          f"Avg Value Loss = {row['avg_value_loss']:8.2f}")

print()
print(f"  Mean Reward (across seeds): {df_ind['avg_reward'].mean():.2f} ± {df_ind['avg_reward'].std():.2f}")
print(f"  Mean Std (within seed)    : {df_ind['std_reward'].mean():.2f}")
print(f"  Variance (across seeds)   : {df_ind['avg_reward'].var():.4f}")
print()

# Comparison
print("[3] A3C vs Individual Comparison")
print("-" * 80)
print(f"  A3C Mean Reward        : {df_a3c['reward'].mean():.2f}")
print(f"  Individual Mean Reward : {df_ind['avg_reward'].mean():.2f}")
print(f"  Difference             : {df_a3c['reward'].mean() - df_ind['avg_reward'].mean():+.2f}")
print(f"  Percentage Difference  : {100 * (df_a3c['reward'].mean() - df_ind['avg_reward'].mean()) / df_ind['avg_reward'].mean():+.2f}%")
print()
print(f"  A3C Variance           : {df_a3c['reward'].var():.4f}")
print(f"  Individual Variance    : {df_ind['avg_reward'].var():.4f}")
print(f"  Stability Ratio        : {df_ind['avg_reward'].var() / df_a3c['reward'].var():.2f}x")
print()

# Worker-level variance analysis
print("[4] Individual Worker Variance Analysis")
print("-" * 80)
all_worker_rewards = []
for result in ind_results:
    for worker in result['workers']:
        all_worker_rewards.append(worker['reward'])

print(f"  Total Individual Experiments: {len(all_worker_rewards)} (5 seeds × 5 workers)")
print(f"  Mean Reward                 : {np.mean(all_worker_rewards):.2f}")
print(f"  Std Deviation               : {np.std(all_worker_rewards):.2f}")
print(f"  Min Reward                  : {np.min(all_worker_rewards):.2f}")
print(f"  Max Reward                  : {np.max(all_worker_rewards):.2f}")
print(f"  Range                       : {np.max(all_worker_rewards) - np.min(all_worker_rewards):.2f}")
print()

# Per-seed breakdown
print("[5] Per-Seed Breakdown (Individual Workers)")
print("-" * 80)
for result in ind_results:
    print(f"  Seed {int(result['seed']):4d}:")
    worker_rewards = [w['reward'] for w in result['workers']]
    for i, wr in enumerate(worker_rewards):
        print(f"    Worker {i}: {wr:6.2f}")
    print(f"    Mean: {np.mean(worker_rewards):6.2f}, Std: {np.std(worker_rewards):5.2f}")
    print()

# Statistical test
from scipy import stats
if len(df_a3c) > 1 and len(df_ind) > 1:
    t_stat, p_value = stats.ttest_ind(df_a3c['reward'], df_ind['avg_reward'])
    print("[6] Statistical Significance Test")
    print("-" * 80)
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value    : {p_value:.4f}")
    if p_value < 0.05:
        winner = 'A3C' if t_stat > 0 else 'Individual'
        print(f"  Result     : {winner} is significantly better (p < 0.05)")
    else:
        print(f"  Result     : No significant difference (p >= 0.05)")
    print()

# Save summary
summary_data = {
    'model': ['A3C', 'Individual'],
    'mean_reward': [df_a3c['reward'].mean(), df_ind['avg_reward'].mean()],
    'std_reward': [df_a3c['reward'].std(), df_ind['avg_reward'].std()],
    'variance': [df_a3c['reward'].var(), df_ind['avg_reward'].var()],
    'min_reward': [df_a3c['reward'].min(), df_ind['avg_reward'].min()],
    'max_reward': [df_a3c['reward'].max(), df_ind['avg_reward'].max()],
}

df_summary = pd.DataFrame(summary_data)
summary_file = results_dir / "summary_analysis.csv"
df_summary.to_csv(summary_file, index=False)

print(f"[Summary saved to: {summary_file}]")
print()
print("="*80)
