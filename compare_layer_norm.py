"""
Compare training results with and without Layer Normalization.

This script compares:
1. Training curves (reward, loss, entropy)
2. Final performance
3. Generalization performance
4. Training stability (variance)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def load_training_log(run_dir):
    """Load training log CSV"""
    log_path = os.path.join(run_dir, "training_log.csv")
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Training log not found: {log_path}")
    return pd.read_csv(log_path)

def analyze_training_stability(df, label):
    """Analyze training stability metrics"""
    print(f"\n{'=' * 60}")
    print(f"{label} - Training Stability Analysis")
    print(f"{'=' * 60}")

    # Group by episode for averaging across workers
    epi = df.groupby("episode").agg(
        reward_mean=("reward", "mean"),
        reward_std=("reward", "std"),
        value_loss_mean=("value_loss", "mean"),
        value_loss_max=("value_loss", "max"),
        policy_loss_mean=("policy_loss", "mean"),
        entropy_mean=("entropy", "mean")
    ).reset_index()

    # Calculate stability metrics
    total_episodes = len(epi)

    # Reward stability
    reward_overall_mean = epi["reward_mean"].mean()
    reward_overall_std = epi["reward_mean"].std()
    reward_range = epi["reward_mean"].max() - epi["reward_mean"].min()

    print(f"\n📊 Reward Statistics:")
    print(f"  Mean: {reward_overall_mean:.2f}")
    print(f"  Std:  {reward_overall_std:.2f}")
    print(f"  Range: {reward_range:.2f}")
    print(f"  CV (coefficient of variation): {reward_overall_std/reward_overall_mean:.2%}")

    # Value Loss explosions
    value_loss_threshold = 100
    value_loss_explosions = (epi["value_loss_max"] > value_loss_threshold).sum()
    value_loss_max = epi["value_loss_max"].max()
    value_loss_mean = epi["value_loss_mean"].mean()

    print(f"\n📉 Value Loss Statistics:")
    print(f"  Mean: {value_loss_mean:.2f}")
    print(f"  Max:  {value_loss_max:.2f}")
    print(f"  Explosions (>100): {value_loss_explosions} / {total_episodes} ({value_loss_explosions/total_episodes:.1%})")

    # Policy loss
    policy_loss_mean = epi["policy_loss_mean"].mean()
    policy_loss_std = epi["policy_loss_mean"].std()

    print(f"\n🎯 Policy Loss Statistics:")
    print(f"  Mean: {policy_loss_mean:.4f}")
    print(f"  Std:  {policy_loss_std:.4f}")

    # Entropy (exploration)
    entropy_mean = epi["entropy_mean"].mean()
    entropy_std = epi["entropy_mean"].std()

    print(f"\n🎲 Entropy Statistics:")
    print(f"  Mean: {entropy_mean:.4f}")
    print(f"  Std:  {entropy_std:.4f}")

    return {
        "reward_mean": reward_overall_mean,
        "reward_std": reward_overall_std,
        "reward_cv": reward_overall_std/reward_overall_mean,
        "value_loss_mean": value_loss_mean,
        "value_loss_max": value_loss_max,
        "value_loss_explosions": value_loss_explosions,
        "policy_loss_mean": policy_loss_mean,
        "entropy_mean": entropy_mean
    }

def plot_comparison(df_with_ln, df_without_ln, output_prefix="comparison"):
    """Plot side-by-side comparison of training curves"""

    # Prepare data
    epi_with = df_with_ln.groupby("episode").agg(
        reward=("reward", "mean"),
        value_loss=("value_loss", "mean"),
        policy_loss=("policy_loss", "mean"),
        entropy=("entropy", "mean")
    ).reset_index()

    epi_without = df_without_ln.groupby("episode").agg(
        reward=("reward", "mean"),
        value_loss=("value_loss", "mean"),
        policy_loss=("policy_loss", "mean"),
        entropy=("entropy", "mean")
    ).reset_index()

    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training Comparison: With vs Without Layer Normalization", fontsize=16, fontweight='bold')

    # Reward
    axes[0, 0].plot(epi_with["episode"], epi_with["reward"], label="With LayerNorm", alpha=0.7, linewidth=1.5)
    axes[0, 0].plot(epi_without["episode"], epi_without["reward"], label="Without LayerNorm", alpha=0.7, linewidth=1.5)
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Avg Reward")
    axes[0, 0].set_title("Reward")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Value Loss
    axes[0, 1].plot(epi_with["episode"], epi_with["value_loss"], label="With LayerNorm", alpha=0.7, linewidth=1.5)
    axes[0, 1].plot(epi_without["episode"], epi_without["value_loss"], label="Without LayerNorm", alpha=0.7, linewidth=1.5)
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Value Loss")
    axes[0, 1].set_title("Value Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    # Set y-axis limit to see explosions better
    axes[0, 1].set_ylim([0, min(200, max(epi_with["value_loss"].max(), epi_without["value_loss"].max()) * 1.1)])

    # Policy Loss
    axes[1, 0].plot(epi_with["episode"], epi_with["policy_loss"], label="With LayerNorm", alpha=0.7, linewidth=1.5)
    axes[1, 0].plot(epi_without["episode"], epi_without["policy_loss"], label="Without LayerNorm", alpha=0.7, linewidth=1.5)
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Policy Loss")
    axes[1, 0].set_title("Policy Loss")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Entropy
    axes[1, 1].plot(epi_with["episode"], epi_with["entropy"], label="With LayerNorm", alpha=0.7, linewidth=1.5)
    axes[1, 1].plot(epi_without["episode"], epi_without["entropy"], label="Without LayerNorm", alpha=0.7, linewidth=1.5)
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Entropy")
    axes[1, 1].set_title("Entropy")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_prefix}.png", dpi=200, bbox_inches='tight')
    print(f"\n[Saved] {output_prefix}.png")
    plt.close()

def main():
    if len(sys.argv) < 3:
        print("Usage: python compare_layer_norm.py <timestamp_with_ln> <timestamp_without_ln>")
        print("Example: python compare_layer_norm.py 20251027_141324 20251027_160000")
        sys.exit(1)

    ts_with = sys.argv[1]
    ts_without = sys.argv[2]

    print("=" * 80)
    print("Layer Normalization Comparison")
    print("=" * 80)
    print(f"\nComparing:")
    print(f"  WITH LayerNorm:    {ts_with}")
    print(f"  WITHOUT LayerNorm: {ts_without}")

    # Load training logs (A3C only for now)
    run_with = f"runs/a3c_{ts_with}"
    run_without = f"runs/a3c_{ts_without}"

    print(f"\nLoading training logs...")
    df_with = load_training_log(run_with)
    df_without = load_training_log(run_without)

    print(f"  WITH LayerNorm:    {len(df_with)} records")
    print(f"  WITHOUT LayerNorm: {len(df_without)} records")

    # Analyze stability
    stats_with = analyze_training_stability(df_with, "WITH LayerNorm")
    stats_without = analyze_training_stability(df_without, "WITHOUT LayerNorm")

    # Compare
    print(f"\n{'=' * 60}")
    print("Comparison Summary")
    print(f"{'=' * 60}")

    print(f"\n📊 Reward:")
    print(f"  With LN:    {stats_with['reward_mean']:.2f} ± {stats_with['reward_std']:.2f} (CV: {stats_with['reward_cv']:.2%})")
    print(f"  Without LN: {stats_without['reward_mean']:.2f} ± {stats_without['reward_std']:.2f} (CV: {stats_without['reward_cv']:.2%})")
    reward_improvement = (stats_with['reward_mean'] - stats_without['reward_mean']) / stats_without['reward_mean']
    stability_improvement = (stats_without['reward_cv'] - stats_with['reward_cv']) / stats_without['reward_cv']
    print(f"  → Mean improvement: {reward_improvement:+.1%}")
    print(f"  → Stability improvement (CV reduction): {stability_improvement:+.1%}")

    print(f"\n📉 Value Loss:")
    print(f"  With LN:    Max={stats_with['value_loss_max']:.1f}, Explosions={stats_with['value_loss_explosions']}")
    print(f"  Without LN: Max={stats_without['value_loss_max']:.1f}, Explosions={stats_without['value_loss_explosions']}")
    explosion_reduction = (stats_without['value_loss_explosions'] - stats_with['value_loss_explosions']) / max(stats_without['value_loss_explosions'], 1)
    print(f"  → Explosion reduction: {explosion_reduction:+.1%}")

    # Plot comparison
    plot_comparison(df_with, df_without, f"layer_norm_comparison_{ts_with}_vs_{ts_without}")

    print(f"\n{'=' * 80}")
    print("Analysis Complete!")
    print(f"{'=' * 80}")

if __name__ == "__main__":
    main()
