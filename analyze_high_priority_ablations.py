#!/usr/bin/env python3
"""
Analyze High Priority Ablation Results
Compares 4 high-priority ablations against baseline
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import json

def load_training_results(ablation_dir, ablation_name):
    """Load training results from all seeds for one ablation"""
    results = {
        'a3c': [],
        'individual': []
    }

    ablation_path = Path(ablation_dir) / ablation_name
    if not ablation_path.exists():
        print(f"Warning: {ablation_path} does not exist")
        return None

    # Load results from each seed
    for seed_dir in sorted(ablation_path.glob("seed_*")):
        seed = seed_dir.name.split('_')[1]

        # Load A3C results
        a3c_log = seed_dir / "a3c" / "training_log.csv"
        if a3c_log.exists():
            df = pd.read_csv(a3c_log)
            if len(df) > 0:
                final_reward = df['reward'].iloc[-100:].mean()  # Last 100 episodes
                results['a3c'].append({
                    'seed': int(seed),
                    'final_reward': final_reward,
                    'all_rewards': df['reward'].values
                })

        # Load Individual results
        individual_dir = seed_dir / "individual"
        if individual_dir.exists():
            individual_rewards = []
            for worker_csv in sorted(individual_dir.glob("worker_*.csv")):
                df = pd.read_csv(worker_csv)
                if len(df) > 0:
                    final_reward = df['reward'].iloc[-100:].mean()
                    individual_rewards.append(final_reward)

            if individual_rewards:
                results['individual'].append({
                    'seed': int(seed),
                    'final_reward': np.mean(individual_rewards),
                    'all_rewards': individual_rewards
                })

    return results


def compute_statistics(results_list):
    """Compute mean, std, and other statistics from results"""
    if not results_list:
        return None

    rewards = [r['final_reward'] for r in results_list]

    return {
        'mean': np.mean(rewards),
        'std': np.std(rewards, ddof=1),
        'median': np.median(rewards),
        'min': np.min(rewards),
        'max': np.max(rewards),
        'n': len(rewards)
    }


def compare_with_baseline(ablation_stats, baseline_stats):
    """Compare ablation with baseline using t-test"""
    if ablation_stats is None or baseline_stats is None:
        return None

    # Compute difference
    diff = ablation_stats['mean'] - baseline_stats['mean']
    pct_diff = (diff / baseline_stats['mean']) * 100

    return {
        'diff': diff,
        'pct_diff': pct_diff,
        'ablation_mean': ablation_stats['mean'],
        'ablation_std': ablation_stats['std'],
        'baseline_mean': baseline_stats['mean'],
        'baseline_std': baseline_stats['std'],
    }


def main():
    # Configuration
    ablation_dir = "ablation_results/high_priority"
    output_dir = "ablation_results/analysis"
    os.makedirs(output_dir, exist_ok=True)

    # Baseline reference values (from CLAUDE.md)
    baseline_ref = {
        'a3c': {'mean': 60.31, 'std': 6.41, 'n': 5},
        'individual': {'mean': 57.57, 'std': 4.84, 'n': 5}
    }

    # High priority ablations
    ablations = [
        "ablation_1_no_rnn",
        "ablation_2_no_layer_norm",
        "ablation_15_few_workers",
        "ablation_16_many_workers"
    ]

    ablation_descriptions = {
        "ablation_1_no_rnn": "No RNN (Feedforward)",
        "ablation_2_no_layer_norm": "No Layer Normalization",
        "ablation_15_few_workers": "Fewer Workers (n=3)",
        "ablation_16_many_workers": "More Workers (n=10)"
    }

    print("="*80)
    print("High Priority Ablation Analysis")
    print("="*80)
    print()

    # Collect all results
    all_results = {}
    for ablation in ablations:
        print(f"Loading results: {ablation}")
        results = load_training_results(ablation_dir, ablation)
        if results:
            all_results[ablation] = results
            print(f"  A3C seeds: {len(results['a3c'])}")
            print(f"  Individual seeds: {len(results['individual'])}")
        else:
            print(f"  No results found")
        print()

    # Compute statistics for each ablation
    print("="*80)
    print("Training Performance Summary")
    print("="*80)
    print()

    summary_rows = []

    # Baseline row
    summary_rows.append({
        'Ablation': 'Baseline',
        'Description': 'Full A3C (5 workers, RNN, LayerNorm)',
        'A3C_Mean': baseline_ref['a3c']['mean'],
        'A3C_Std': baseline_ref['a3c']['std'],
        'Individual_Mean': baseline_ref['individual']['mean'],
        'Individual_Std': baseline_ref['individual']['std'],
        'Gap': baseline_ref['a3c']['mean'] - baseline_ref['individual']['mean'],
        'Gap_Pct': ((baseline_ref['a3c']['mean'] - baseline_ref['individual']['mean']) /
                    baseline_ref['individual']['mean'] * 100)
    })

    for ablation in ablations:
        if ablation not in all_results:
            continue

        results = all_results[ablation]

        # Compute statistics
        a3c_stats = compute_statistics(results['a3c'])
        individual_stats = compute_statistics(results['individual'])

        if a3c_stats and individual_stats:
            gap = a3c_stats['mean'] - individual_stats['mean']
            gap_pct = (gap / individual_stats['mean']) * 100

            summary_rows.append({
                'Ablation': ablation,
                'Description': ablation_descriptions.get(ablation, ''),
                'A3C_Mean': a3c_stats['mean'],
                'A3C_Std': a3c_stats['std'],
                'Individual_Mean': individual_stats['mean'],
                'Individual_Std': individual_stats['std'],
                'Gap': gap,
                'Gap_Pct': gap_pct
            })

            # Compare with baseline
            a3c_comparison = compare_with_baseline(a3c_stats, baseline_ref['a3c'])
            individual_comparison = compare_with_baseline(individual_stats, baseline_ref['individual'])

            print(f"{ablation_descriptions.get(ablation, ablation)}")
            print(f"  A3C: {a3c_stats['mean']:.2f} ± {a3c_stats['std']:.2f}")
            print(f"    vs Baseline: {a3c_comparison['diff']:+.2f} ({a3c_comparison['pct_diff']:+.2f}%)")
            print(f"  Individual: {individual_stats['mean']:.2f} ± {individual_stats['std']:.2f}")
            print(f"    vs Baseline: {individual_comparison['diff']:+.2f} ({individual_comparison['pct_diff']:+.2f}%)")
            print(f"  Gap: {gap:.2f} ({gap_pct:+.2f}%)")
            print()

    # Save summary table
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = Path(output_dir) / "high_priority_training_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved summary table: {summary_csv}")
    print()

    # Print formatted table
    print("="*80)
    print("Training Performance Table (Paper-Ready)")
    print("="*80)
    print()
    print(f"{'Ablation':<35} {'A3C':<20} {'Individual':<20} {'Gap':>15}")
    print("-"*80)

    for _, row in summary_df.iterrows():
        ablation_name = row['Description'] if row['Description'] else row['Ablation']
        a3c_str = f"{row['A3C_Mean']:.2f} ± {row['A3C_Std']:.2f}"
        individual_str = f"{row['Individual_Mean']:.2f} ± {row['Individual_Std']:.2f}"
        gap_str = f"{row['Gap']:+.2f} ({row['Gap_Pct']:+.1f}%)"

        print(f"{ablation_name:<35} {a3c_str:<20} {individual_str:<20} {gap_str:>15}")

    print()

    # Key findings
    print("="*80)
    print("Key Findings")
    print("="*80)
    print()

    if len(summary_df) > 1:
        baseline_gap = summary_df.iloc[0]['Gap']

        for idx, row in summary_df.iloc[1:].iterrows():
            ablation_gap = row['Gap']
            gap_reduction = baseline_gap - ablation_gap
            gap_reduction_pct = (gap_reduction / baseline_gap) * 100

            print(f"{row['Description']}")
            print(f"  A3C advantage reduced by: {gap_reduction:.2f} ({gap_reduction_pct:.1f}%)")

            if gap_reduction > baseline_gap * 0.5:
                print(f"  → CRITICAL component for A3C's advantage")
            elif gap_reduction > baseline_gap * 0.25:
                print(f"  → IMPORTANT component for A3C's advantage")
            else:
                print(f"  → Minor impact on A3C's advantage")
            print()

    print("="*80)
    print("Analysis complete!")
    print(f"Results saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
