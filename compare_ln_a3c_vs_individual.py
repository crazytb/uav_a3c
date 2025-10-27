"""
Compare Layer Normalization effect on A3C vs Individual models separately.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_and_analyze(run_type, timestamp, label):
    """Load training log and compute statistics"""
    log_path = f"runs/{run_type}_{timestamp}/training_log.csv"
    df = pd.read_csv(log_path)

    # Group by episode
    epi = df.groupby("episode").agg(
        reward_mean=("reward", "mean"),
        reward_std=("reward", "std"),
        value_loss_mean=("value_loss", "mean"),
        value_loss_max=("value_loss", "max"),
        policy_loss_mean=("policy_loss", "mean"),
        entropy_mean=("entropy", "mean")
    ).reset_index()

    # Calculate statistics
    stats = {
        "reward_mean": epi["reward_mean"].mean(),
        "reward_std": epi["reward_mean"].std(),
        "reward_cv": epi["reward_mean"].std() / epi["reward_mean"].mean(),
        "value_loss_mean": epi["value_loss_mean"].mean(),
        "value_loss_max": epi["value_loss_max"].max(),
        "value_loss_explosions": (epi["value_loss_max"] > 100).sum(),
        "total_episodes": len(epi),
        "policy_loss_mean": epi["policy_loss_mean"].mean(),
        "entropy_mean": epi["entropy_mean"].mean()
    }

    return epi, stats

print("=" * 80)
print("Layer Normalization Effect: A3C vs Individual Comparison")
print("=" * 80)

# Load all data
print("\nLoading data...")
a3c_with_ln, stats_a3c_with = load_and_analyze("a3c", "20251027_141324", "A3C with LN")
a3c_without_ln, stats_a3c_without = load_and_analyze("a3c", "20251027_143604", "A3C without LN")
ind_with_ln, stats_ind_with = load_and_analyze("individual", "20251027_141324", "Individual with LN")
ind_without_ln, stats_ind_without = load_and_analyze("individual", "20251027_143604", "Individual without LN")

print("✓ Data loaded successfully")

# ===== A3C Analysis =====
print("\n" + "=" * 80)
print("A3C: Layer Normalization Effect")
print("=" * 80)

print("\n📊 Reward:")
print(f"  With LN:    {stats_a3c_with['reward_mean']:.2f} ± {stats_a3c_with['reward_std']:.2f} (CV: {stats_a3c_with['reward_cv']:.2%})")
print(f"  Without LN: {stats_a3c_without['reward_mean']:.2f} ± {stats_a3c_without['reward_std']:.2f} (CV: {stats_a3c_without['reward_cv']:.2%})")
reward_improv_a3c = (stats_a3c_with['reward_mean'] - stats_a3c_without['reward_mean']) / stats_a3c_without['reward_mean']
stability_improv_a3c = (stats_a3c_without['reward_cv'] - stats_a3c_with['reward_cv']) / stats_a3c_without['reward_cv']
print(f"  → Mean improvement: {reward_improv_a3c:+.1%}")
print(f"  → Stability improvement: {stability_improv_a3c:+.1%}")

print("\n📉 Value Loss:")
print(f"  With LN:    Mean={stats_a3c_with['value_loss_mean']:.1f}, Max={stats_a3c_with['value_loss_max']:.1f}, Explosions={stats_a3c_with['value_loss_explosions']}/{stats_a3c_with['total_episodes']}")
print(f"  Without LN: Mean={stats_a3c_without['value_loss_mean']:.1f}, Max={stats_a3c_without['value_loss_max']:.1f}, Explosions={stats_a3c_without['value_loss_explosions']}/{stats_a3c_without['total_episodes']}")
vloss_mean_reduction_a3c = (stats_a3c_without['value_loss_mean'] - stats_a3c_with['value_loss_mean']) / stats_a3c_without['value_loss_mean']
explosion_reduction_a3c = (stats_a3c_without['value_loss_explosions'] - stats_a3c_with['value_loss_explosions']) / stats_a3c_without['value_loss_explosions']
print(f"  → Mean value loss reduction: {vloss_mean_reduction_a3c:+.1%}")
print(f"  → Explosion reduction: {explosion_reduction_a3c:+.1%}")

# ===== Individual Analysis =====
print("\n" + "=" * 80)
print("Individual: Layer Normalization Effect")
print("=" * 80)

print("\n📊 Reward:")
print(f"  With LN:    {stats_ind_with['reward_mean']:.2f} ± {stats_ind_with['reward_std']:.2f} (CV: {stats_ind_with['reward_cv']:.2%})")
print(f"  Without LN: {stats_ind_without['reward_mean']:.2f} ± {stats_ind_without['reward_std']:.2f} (CV: {stats_ind_without['reward_cv']:.2%})")
reward_improv_ind = (stats_ind_with['reward_mean'] - stats_ind_without['reward_mean']) / stats_ind_without['reward_mean']
stability_improv_ind = (stats_ind_without['reward_cv'] - stats_ind_with['reward_cv']) / stats_ind_without['reward_cv']
print(f"  → Mean improvement: {reward_improv_ind:+.1%}")
print(f"  → Stability improvement: {stability_improv_ind:+.1%}")

print("\n📉 Value Loss:")
print(f"  With LN:    Mean={stats_ind_with['value_loss_mean']:.1f}, Max={stats_ind_with['value_loss_max']:.1f}, Explosions={stats_ind_with['value_loss_explosions']}/{stats_ind_with['total_episodes']}")
print(f"  Without LN: Mean={stats_ind_without['value_loss_mean']:.1f}, Max={stats_ind_without['value_loss_max']:.1f}, Explosions={stats_ind_without['value_loss_explosions']}/{stats_ind_without['total_episodes']}")
vloss_mean_reduction_ind = (stats_ind_without['value_loss_mean'] - stats_ind_with['value_loss_mean']) / stats_ind_without['value_loss_mean']
explosion_reduction_ind = (stats_ind_without['value_loss_explosions'] - stats_ind_with['value_loss_explosions']) / stats_ind_without['value_loss_explosions']
print(f"  → Mean value loss reduction: {vloss_mean_reduction_ind:+.1%}")
print(f"  → Explosion reduction: {explosion_reduction_ind:+.1%}")

# ===== Comparison Summary =====
print("\n" + "=" * 80)
print("Comparative Summary: Which benefits more from Layer Normalization?")
print("=" * 80)

print("\n🏆 Reward Improvement:")
print(f"  A3C:        {reward_improv_a3c:+.1%}")
print(f"  Individual: {reward_improv_ind:+.1%}")
if abs(reward_improv_a3c) > abs(reward_improv_ind):
    print(f"  → A3C benefits MORE from Layer Normalization (+{abs(reward_improv_a3c - reward_improv_ind):.1%}p difference)")
else:
    print(f"  → Individual benefits MORE from Layer Normalization (+{abs(reward_improv_ind - reward_improv_a3c):.1%}p difference)")

print("\n🛡️ Value Loss Reduction:")
print(f"  A3C:        {vloss_mean_reduction_a3c:+.1%}")
print(f"  Individual: {vloss_mean_reduction_ind:+.1%}")
if vloss_mean_reduction_a3c > vloss_mean_reduction_ind:
    print(f"  → A3C benefits MORE from Layer Normalization (+{vloss_mean_reduction_a3c - vloss_mean_reduction_ind:.1%}p difference)")
else:
    print(f"  → Individual benefits MORE from Layer Normalization (+{vloss_mean_reduction_ind - vloss_mean_reduction_a3c:.1%}p difference)")

print("\n💥 Explosion Reduction:")
print(f"  A3C:        {explosion_reduction_a3c:+.1%} (from {stats_a3c_without['value_loss_explosions']} to {stats_a3c_with['value_loss_explosions']})")
print(f"  Individual: {explosion_reduction_ind:+.1%} (from {stats_ind_without['value_loss_explosions']} to {stats_ind_with['value_loss_explosions']})")

# ===== Visualization =====
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Layer Normalization Effect: A3C vs Individual", fontsize=16, fontweight='bold')

# Row 1: A3C
# Reward
axes[0, 0].plot(a3c_with_ln["episode"], a3c_with_ln["reward"], label="With LN", alpha=0.7, linewidth=1.5, color='blue')
axes[0, 0].plot(a3c_without_ln["episode"], a3c_without_ln["reward"], label="Without LN", alpha=0.7, linewidth=1.5, color='orange')
axes[0, 0].set_title("A3C: Reward")
axes[0, 0].set_xlabel("Episode")
axes[0, 0].set_ylabel("Avg Reward")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Value Loss
axes[0, 1].plot(a3c_with_ln["episode"], a3c_with_ln["value_loss_mean"], label="With LN", alpha=0.7, linewidth=1.5, color='blue')
axes[0, 1].plot(a3c_without_ln["episode"], a3c_without_ln["value_loss_mean"], label="Without LN", alpha=0.7, linewidth=1.5, color='orange')
axes[0, 1].set_title("A3C: Value Loss")
axes[0, 1].set_xlabel("Episode")
axes[0, 1].set_ylabel("Value Loss")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_ylim([0, 200])

# Entropy
axes[0, 2].plot(a3c_with_ln["episode"], a3c_with_ln["entropy_mean"], label="With LN", alpha=0.7, linewidth=1.5, color='blue')
axes[0, 2].plot(a3c_without_ln["episode"], a3c_without_ln["entropy_mean"], label="Without LN", alpha=0.7, linewidth=1.5, color='orange')
axes[0, 2].set_title("A3C: Entropy")
axes[0, 2].set_xlabel("Episode")
axes[0, 2].set_ylabel("Entropy")
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# Row 2: Individual
# Reward
axes[1, 0].plot(ind_with_ln["episode"], ind_with_ln["reward"], label="With LN", alpha=0.7, linewidth=1.5, color='blue')
axes[1, 0].plot(ind_without_ln["episode"], ind_without_ln["reward"], label="Without LN", alpha=0.7, linewidth=1.5, color='orange')
axes[1, 0].set_title("Individual: Reward")
axes[1, 0].set_xlabel("Episode")
axes[1, 0].set_ylabel("Avg Reward")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Value Loss
axes[1, 1].plot(ind_with_ln["episode"], ind_with_ln["value_loss_mean"], label="With LN", alpha=0.7, linewidth=1.5, color='blue')
axes[1, 1].plot(ind_without_ln["episode"], ind_without_ln["value_loss_mean"], label="Without LN", alpha=0.7, linewidth=1.5, color='orange')
axes[1, 1].set_title("Individual: Value Loss")
axes[1, 1].set_xlabel("Episode")
axes[1, 1].set_ylabel("Value Loss")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_ylim([0, 200])

# Entropy
axes[1, 2].plot(ind_with_ln["episode"], ind_with_ln["entropy_mean"], label="With LN", alpha=0.7, linewidth=1.5, color='blue')
axes[1, 2].plot(ind_without_ln["episode"], ind_without_ln["entropy_mean"], label="Without LN", alpha=0.7, linewidth=1.5, color='orange')
axes[1, 2].set_title("Individual: Entropy")
axes[1, 2].set_xlabel("Episode")
axes[1, 2].set_ylabel("Entropy")
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("ln_effect_a3c_vs_individual.png", dpi=200, bbox_inches='tight')
print(f"\n[Saved] ln_effect_a3c_vs_individual.png")
plt.close()

# Create comparison bar chart
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Layer Normalization Benefits: A3C vs Individual", fontsize=14, fontweight='bold')

metrics = ['A3C', 'Individual']

# Reward improvement
reward_improvements = [reward_improv_a3c * 100, reward_improv_ind * 100]
axes[0].bar(metrics, reward_improvements, color=['#3498db', '#e74c3c'])
axes[0].set_ylabel("Reward Improvement (%)")
axes[0].set_title("Reward Gain from Layer Norm")
axes[0].axhline(y=0, color='black', linestyle='--', linewidth=0.8)
axes[0].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(reward_improvements):
    axes[0].text(i, v + 0.5, f"{v:.1f}%", ha='center', fontweight='bold')

# Value loss reduction
vloss_reductions = [vloss_mean_reduction_a3c * 100, vloss_mean_reduction_ind * 100]
axes[1].bar(metrics, vloss_reductions, color=['#3498db', '#e74c3c'])
axes[1].set_ylabel("Value Loss Reduction (%)")
axes[1].set_title("Value Loss Decrease from Layer Norm")
axes[1].axhline(y=0, color='black', linestyle='--', linewidth=0.8)
axes[1].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(vloss_reductions):
    axes[1].text(i, v + 2, f"{v:.1f}%", ha='center', fontweight='bold')

# Explosion reduction
explosion_reductions = [explosion_reduction_a3c * 100, explosion_reduction_ind * 100]
axes[2].bar(metrics, explosion_reductions, color=['#3498db', '#e74c3c'])
axes[2].set_ylabel("Explosion Reduction (%)")
axes[2].set_title("Value Loss Explosions Decrease")
axes[2].axhline(y=0, color='black', linestyle='--', linewidth=0.8)
axes[2].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(explosion_reductions):
    axes[2].text(i, v + 1, f"{v:.1f}%", ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig("ln_benefits_comparison.png", dpi=200, bbox_inches='tight')
print(f"[Saved] ln_benefits_comparison.png")
plt.close()

print("\n" + "=" * 80)
print("Analysis Complete!")
print("=" * 80)
