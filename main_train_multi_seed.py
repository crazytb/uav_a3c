"""
Multi-seed training script for robust evaluation
Runs A3C and Individual training with multiple random seeds
"""

import os
import subprocess
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import params to get n_workers
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from drl_framework import params

# Configuration
N_SEEDS = 10  # Number of random seeds to run
SEEDS = [42, 123, 456, 789, 1024, 2048, 4096, 8192, 16384, 32768]  # Fixed seeds for reproducibility
N_WORKERS = params.n_workers  # Get from params.py

print("="*80)
print(f"Multi-Seed Training Configuration")
print("="*80)
print(f"  Seeds: {N_SEEDS} experiments")
print(f"  Workers: {N_WORKERS}")
print(f"  Episodes per worker: {params.target_episode_count}")
print("="*80)
print()

all_results = []

for seed_idx, seed in enumerate(SEEDS):
    print(f"\n{'='*80}")
    print(f"[{seed_idx+1}/{N_SEEDS}] Running with seed={seed}")
    print(f"{'='*80}\n")

    # Set environment variable for seed
    env = os.environ.copy()
    env['RANDOM_SEED'] = str(seed)

    # Run training
    try:
        result = subprocess.run(
            ['conda', 'run', '-n', 'torch-cert', 'python', 'main_train.py'],
            env=env,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )

        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        # Find the latest run directories
        import glob
        a3c_runs = sorted(glob.glob('runs/a3c_*'))
        ind_runs = sorted(glob.glob('runs/individual_*'))

        if a3c_runs and ind_runs:
            latest_a3c = a3c_runs[-1]
            latest_ind = ind_runs[-1]

            # Extract timestamp from path
            timestamp = os.path.basename(latest_a3c).replace('a3c_', '')

            # Read final rewards from summary CSVs
            a3c_summary = pd.read_csv(os.path.join(latest_a3c, 'summary_global.csv'))
            a3c_final_reward = a3c_summary['reward'].iloc[-1]

            # Individual: average of all workers
            ind_rewards = []
            for worker_id in range(N_WORKERS):
                ind_summary_path = os.path.join(latest_ind, f'summary_Individual_{worker_id}.csv')
                if os.path.exists(ind_summary_path):
                    ind_summary = pd.read_csv(ind_summary_path)
                    ind_rewards.append(ind_summary['reward'].iloc[-1])

            ind_final_reward = np.mean(ind_rewards) if ind_rewards else 0.0

            all_results.append({
                'seed': seed,
                'timestamp': timestamp,
                'a3c_final_reward': a3c_final_reward,
                'individual_final_reward': ind_final_reward,
                'difference': a3c_final_reward - ind_final_reward
            })

            print(f"\n[Seed {seed}] Results:")
            print(f"  A3C Final Reward       : {a3c_final_reward:.3f}")
            print(f"  Individual Final Reward: {ind_final_reward:.3f}")
            print(f"  Difference (A3C - Ind) : {a3c_final_reward - ind_final_reward:+.3f}")

    except subprocess.TimeoutExpired:
        print(f"[WARNING] Seed {seed} timed out after 1 hour")
    except Exception as e:
        print(f"[ERROR] Seed {seed} failed: {e}")

# ========================================
# Aggregate Results
# ========================================

if all_results:
    df_results = pd.DataFrame(all_results)

    print("\n" + "="*80)
    print("=== Multi-Seed Results Summary ===")
    print("="*80)
    print()

    # Statistics
    print("[1] A3C Global")
    print(f"  Mean  : {df_results['a3c_final_reward'].mean():.3f}")
    print(f"  Std   : {df_results['a3c_final_reward'].std():.3f}")
    print(f"  Min   : {df_results['a3c_final_reward'].min():.3f}")
    print(f"  Max   : {df_results['a3c_final_reward'].max():.3f}")
    print()

    print("[2] Individual Workers (Average)")
    print(f"  Mean  : {df_results['individual_final_reward'].mean():.3f}")
    print(f"  Std   : {df_results['individual_final_reward'].std():.3f}")
    print(f"  Min   : {df_results['individual_final_reward'].min():.3f}")
    print(f"  Max   : {df_results['individual_final_reward'].max():.3f}")
    print()

    print("[3] Difference (A3C - Individual)")
    print(f"  Mean  : {df_results['difference'].mean():+.3f}")
    print(f"  Std   : {df_results['difference'].std():.3f}")
    print()

    # Statistical significance (paired t-test)
    from scipy import stats
    t_stat, p_value = stats.ttest_rel(
        df_results['a3c_final_reward'],
        df_results['individual_final_reward']
    )

    print("[4] Statistical Significance (Paired t-test)")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value    : {p_value:.4f}")
    if p_value < 0.05:
        winner = 'A3C' if t_stat > 0 else 'Individual'
        print(f"  Result     : {winner} is significantly better (p < 0.05)")
    else:
        print(f"  Result     : No significant difference (p >= 0.05)")
    print()

    # Save results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_output = f"multi_seed_results_{timestamp}.csv"
    df_results.to_csv(csv_output, index=False)
    print(f"[Results saved] {csv_output}")
    print()

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (1) Box plot comparison
    ax1 = axes[0]
    data_to_plot = [df_results['a3c_final_reward'], df_results['individual_final_reward']]
    bp = ax1.boxplot(data_to_plot, labels=['A3C Global', 'Individual'], patch_artist=True)
    bp['boxes'][0].set_facecolor('#3498db')
    bp['boxes'][1].set_facecolor('#e74c3c')
    ax1.set_ylabel('Final Reward', fontsize=12)
    ax1.set_title('Final Reward Distribution', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # (2) Seed-by-seed comparison
    ax2 = axes[1]
    x = np.arange(len(df_results))
    width = 0.35
    ax2.bar(x - width/2, df_results['a3c_final_reward'], width, label='A3C Global', color='#3498db')
    ax2.bar(x + width/2, df_results['individual_final_reward'], width, label='Individual', color='#e74c3c')
    ax2.set_xlabel('Seed Index', fontsize=12)
    ax2.set_ylabel('Final Reward', fontsize=12)
    ax2.set_title('Seed-by-Seed Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"S{i+1}" for i in range(len(df_results))])
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    # (3) Difference plot
    ax3 = axes[2]
    colors = ['#2ecc71' if d > 0 else '#e74c3c' for d in df_results['difference']]
    ax3.bar(x, df_results['difference'], color=colors, alpha=0.7)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax3.axhline(y=df_results['difference'].mean(), color='blue', linestyle='--',
                linewidth=2, label=f'Mean: {df_results["difference"].mean():+.3f}')
    ax3.set_xlabel('Seed Index', fontsize=12)
    ax3.set_ylabel('Difference (A3C - Individual)', fontsize=12)
    ax3.set_title('Performance Difference', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"S{i+1}" for i in range(len(df_results))])
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plot_output = f"multi_seed_comparison_{timestamp}.png"
    plt.savefig(plot_output, dpi=180, bbox_inches='tight')
    print(f"[Visualization saved] {plot_output}")
    print()

    print("="*80)
    print("Multi-Seed Training Complete!")
    print("="*80)

else:
    print("\n[ERROR] No results collected. Please check the training script.")
