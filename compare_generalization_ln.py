"""
Compare generalization performance: With vs Without Layer Normalization

This script compares the generalization test results from:
- 20251027_141324 (WITH Layer Normalization)
- 20251027_143604 (WITHOUT Layer Normalization)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print("=" * 80)
print("Generalization Performance Comparison: With vs Without Layer Normalization")
print("=" * 80)

# Load CSV results
csv_with_ln = "generalization_results_v2_runs_20251027_141324.csv"
csv_without_ln = "generalization_results_v2_runs_20251027_143604.csv"

print("\nLoading results...")
df_with = pd.read_csv(csv_with_ln)
df_without = pd.read_csv(csv_without_ln)

print(f"  WITH LN:    {len(df_with)} test scenarios")
print(f"  WITHOUT LN: {len(df_without)} test scenarios")

# ===== Analysis by Environment Type =====
print("\n" + "=" * 80)
print("Performance by Environment Type")
print("=" * 80)

for env_type in ["Seen", "Intra", "Extra"]:
    print(f"\n{env_type}:")

    # A3C performance
    a3c_with = df_with[(df_with["model"] == "A3C_Global") & (df_with["env_type"] == env_type)]["mean_reward"].mean()
    a3c_without = df_without[(df_without["model"] == "A3C_Global") & (df_without["env_type"] == env_type)]["mean_reward"].mean()

    # Individual performance
    ind_with = df_with[(df_with["model"].str.startswith("Individual")) & (df_with["env_type"] == env_type)]["mean_reward"].mean()
    ind_without = df_without[(df_without["model"].str.startswith("Individual")) & (df_without["env_type"] == env_type)]["mean_reward"].mean()

    print(f"  A3C Global:")
    print(f"    With LN:    {a3c_with:.2f}")
    print(f"    Without LN: {a3c_without:.2f}")
    a3c_improvement = ((a3c_with - a3c_without) / a3c_without) * 100
    print(f"    → Improvement: {a3c_improvement:+.1f}%")

    print(f"  Individual (avg across workers):")
    print(f"    With LN:    {ind_with:.2f}")
    print(f"    Without LN: {ind_without:.2f}")
    ind_improvement = ((ind_with - ind_without) / ind_without) * 100
    print(f"    → Improvement: {ind_improvement:+.1f}%")

# ===== Individual Worker Analysis =====
print("\n" + "=" * 80)
print("Individual Worker Generalization Gap (Intra→Extra degradation)")
print("=" * 80)

workers = range(5)
for w in workers:
    model_name = f"Individual_W{w}"

    # WITH LN
    seen_with = df_with[(df_with["model"] == model_name) & (df_with["env_type"] == "Seen")]["mean_reward"].mean()
    intra_with = df_with[(df_with["model"] == model_name) & (df_with["env_type"] == "Intra")]["mean_reward"].mean()
    extra_with = df_with[(df_with["model"] == model_name) & (df_with["env_type"] == "Extra")]["mean_reward"].mean()

    # WITHOUT LN
    seen_without = df_without[(df_without["model"] == model_name) & (df_without["env_type"] == "Seen")]["mean_reward"].mean()
    intra_without = df_without[(df_without["model"] == model_name) & (df_without["env_type"] == "Intra")]["mean_reward"].mean()
    extra_without = df_without[(df_without["model"] == model_name) & (df_without["env_type"] == "Extra")]["mean_reward"].mean()

    # Training environment
    train_envs = ["comp=200,vel=5", "comp=200,vel=10", "comp=200,vel=15", "comp=200,vel=20", "comp=200,vel=25"]

    print(f"\nWorker {w} (trained on {train_envs[w]}):")
    print(f"  WITH LN:    Seen={seen_with:.1f}, Intra={intra_with:.1f} ({intra_with-seen_with:+.1f}), Extra={extra_with:.1f} ({extra_with-seen_with:+.1f})")
    print(f"  WITHOUT LN: Seen={seen_without:.1f}, Intra={intra_without:.1f} ({intra_without-seen_without:+.1f}), Extra={extra_without:.1f} ({extra_without-seen_without:+.1f})")

    # Generalization gap (Extra performance relative to Seen)
    gap_with = extra_with - seen_with
    gap_without = extra_without - seen_without
    print(f"  Generalization Gap (Extra - Seen):")
    print(f"    With LN:    {gap_with:+.1f}")
    print(f"    Without LN: {gap_without:+.1f}")
    print(f"    → LN improves gap by {gap_with - gap_without:+.1f}")

# ===== Variance Analysis =====
print("\n" + "=" * 80)
print("Performance Variance (Stability across environments)")
print("=" * 80)

# A3C variance
a3c_with_var = df_with[df_with["model"] == "A3C_Global"]["mean_reward"].std()
a3c_without_var = df_without[df_without["model"] == "A3C_Global"]["mean_reward"].std()

# Individual variance
ind_with_var = df_with[df_with["model"].str.startswith("Individual")]["mean_reward"].std()
ind_without_var = df_without[df_without["model"].str.startswith("Individual")]["mean_reward"].std()

print(f"\nA3C Global:")
print(f"  With LN:    σ={a3c_with_var:.2f}")
print(f"  Without LN: σ={a3c_without_var:.2f}")
print(f"  → LN reduces variance by {((a3c_without_var - a3c_with_var) / a3c_without_var) * 100:.1f}%")

print(f"\nIndividual (across all workers and environments):")
print(f"  With LN:    σ={ind_with_var:.2f}")
print(f"  Without LN: σ={ind_without_var:.2f}")
print(f"  → LN changes variance by {((ind_with_var - ind_without_var) / ind_without_var) * 100:+.1f}%")

# ===== Visualization =====
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Generalization Performance: With vs Without Layer Normalization", fontsize=16, fontweight='bold')

# Prepare data for A3C
env_types = ["Seen", "Intra", "Extra"]
a3c_with_means = [df_with[(df_with["model"] == "A3C_Global") & (df_with["env_type"] == t)]["mean_reward"].mean() for t in env_types]
a3c_without_means = [df_without[(df_without["model"] == "A3C_Global") & (df_without["env_type"] == t)]["mean_reward"].mean() for t in env_types]

# A3C: Environment Type Performance
axes[0, 0].bar(np.arange(3) - 0.15, a3c_with_means, width=0.3, label="With LN", color='#3498db', alpha=0.8)
axes[0, 0].bar(np.arange(3) + 0.15, a3c_without_means, width=0.3, label="Without LN", color='#e74c3c', alpha=0.8)
axes[0, 0].set_xticks(range(3))
axes[0, 0].set_xticklabels(env_types)
axes[0, 0].set_ylabel("Avg Reward")
axes[0, 0].set_title("A3C: Generalization by Environment Type")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Individual: Environment Type Performance
ind_with_means = [df_with[(df_with["model"].str.startswith("Individual")) & (df_with["env_type"] == t)]["mean_reward"].mean() for t in env_types]
ind_without_means = [df_without[(df_without["model"].str.startswith("Individual")) & (df_without["env_type"] == t)]["mean_reward"].mean() for t in env_types]

axes[0, 1].bar(np.arange(3) - 0.15, ind_with_means, width=0.3, label="With LN", color='#3498db', alpha=0.8)
axes[0, 1].bar(np.arange(3) + 0.15, ind_without_means, width=0.3, label="Without LN", color='#e74c3c', alpha=0.8)
axes[0, 1].set_xticks(range(3))
axes[0, 1].set_xticklabels(env_types)
axes[0, 1].set_ylabel("Avg Reward")
axes[0, 1].set_title("Individual: Generalization by Environment Type")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Improvement percentage
a3c_improvements = [((a3c_with_means[i] - a3c_without_means[i]) / a3c_without_means[i]) * 100 for i in range(3)]
ind_improvements = [((ind_with_means[i] - ind_without_means[i]) / ind_without_means[i]) * 100 for i in range(3)]

axes[0, 2].bar(np.arange(3) - 0.15, a3c_improvements, width=0.3, label="A3C", color='#2ecc71', alpha=0.8)
axes[0, 2].bar(np.arange(3) + 0.15, ind_improvements, width=0.3, label="Individual", color='#f39c12', alpha=0.8)
axes[0, 2].set_xticks(range(3))
axes[0, 2].set_xticklabels(env_types)
axes[0, 2].set_ylabel("Improvement (%)")
axes[0, 2].set_title("LayerNorm Benefit by Environment Type")
axes[0, 2].axhline(y=0, color='black', linestyle='--', linewidth=0.8)
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3, axis='y')

# Individual Worker Generalization Gap
worker_gaps_with = []
worker_gaps_without = []
for w in workers:
    model_name = f"Individual_W{w}"
    seen_with = df_with[(df_with["model"] == model_name) & (df_with["env_type"] == "Seen")]["mean_reward"].mean()
    extra_with = df_with[(df_with["model"] == model_name) & (df_with["env_type"] == "Extra")]["mean_reward"].mean()
    seen_without = df_without[(df_without["model"] == model_name) & (df_without["env_type"] == "Seen")]["mean_reward"].mean()
    extra_without = df_without[(df_without["model"] == model_name) & (df_without["env_type"] == "Extra")]["mean_reward"].mean()

    worker_gaps_with.append(extra_with - seen_with)
    worker_gaps_without.append(extra_without - seen_without)

axes[1, 0].bar(np.arange(5) - 0.15, worker_gaps_with, width=0.3, label="With LN", color='#3498db', alpha=0.8)
axes[1, 0].bar(np.arange(5) + 0.15, worker_gaps_without, width=0.3, label="Without LN", color='#e74c3c', alpha=0.8)
axes[1, 0].set_xticks(range(5))
axes[1, 0].set_xticklabels([f"W{i}" for i in range(5)])
axes[1, 0].set_ylabel("Extra - Seen Reward")
axes[1, 0].set_title("Individual: Generalization Gap (Extra - Seen)")
axes[1, 0].axhline(y=0, color='black', linestyle='--', linewidth=0.8)
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Variance comparison
variance_data = [
    ("A3C\nWith LN", a3c_with_var),
    ("A3C\nWithout LN", a3c_without_var),
    ("Individual\nWith LN", ind_with_var),
    ("Individual\nWithout LN", ind_without_var)
]
colors = ['#3498db', '#e74c3c', '#3498db', '#e74c3c']

axes[1, 1].bar(range(4), [v[1] for v in variance_data], color=colors, alpha=0.8)
axes[1, 1].set_xticks(range(4))
axes[1, 1].set_xticklabels([v[0] for v in variance_data])
axes[1, 1].set_ylabel("Reward Std Dev (σ)")
axes[1, 1].set_title("Performance Variance Across Environments")
axes[1, 1].grid(True, alpha=0.3, axis='y')

# Overall performance heatmap (summary)
summary_data = np.array([
    [a3c_with_means[0], a3c_with_means[1], a3c_with_means[2]],
    [a3c_without_means[0], a3c_without_means[1], a3c_without_means[2]],
    [ind_with_means[0], ind_with_means[1], ind_with_means[2]],
    [ind_without_means[0], ind_without_means[1], ind_without_means[2]]
])

im = axes[1, 2].imshow(summary_data, cmap='RdYlGn', aspect='auto', vmin=15, vmax=50)
axes[1, 2].set_xticks(range(3))
axes[1, 2].set_xticklabels(env_types)
axes[1, 2].set_yticks(range(4))
axes[1, 2].set_yticklabels(["A3C\nWith LN", "A3C\nWithout LN", "Individual\nWith LN", "Individual\nWithout LN"])
axes[1, 2].set_title("Performance Heatmap")

# Add text annotations
for i in range(4):
    for j in range(3):
        text = axes[1, 2].text(j, i, f"{summary_data[i, j]:.1f}",
                               ha="center", va="center", color="black", fontweight='bold')

plt.colorbar(im, ax=axes[1, 2])

plt.tight_layout()
plt.savefig("generalization_comparison_ln.png", dpi=200, bbox_inches='tight')
print(f"\n[Saved] generalization_comparison_ln.png")
plt.close()

# ===== Summary =====
print("\n" + "=" * 80)
print("Summary: Layer Normalization Effect on Generalization")
print("=" * 80)

print("\n🎯 Key Findings:")

# 1. Performance improvement
print("\n1. Overall Performance Improvement:")
a3c_overall_with = df_with[df_with["model"] == "A3C_Global"]["mean_reward"].mean()
a3c_overall_without = df_without[df_without["model"] == "A3C_Global"]["mean_reward"].mean()
ind_overall_with = df_with[df_with["model"].str.startswith("Individual")]["mean_reward"].mean()
ind_overall_without = df_without[df_without["model"].str.startswith("Individual")]["mean_reward"].mean()

print(f"   A3C:        {((a3c_overall_with - a3c_overall_without) / a3c_overall_without) * 100:+.1f}%")
print(f"   Individual: {((ind_overall_with - ind_overall_without) / ind_overall_without) * 100:+.1f}%")

# 2. Generalization capability
print("\n2. Generalization Capability (Extra environment performance):")
a3c_extra_with = df_with[(df_with["model"] == "A3C_Global") & (df_with["env_type"] == "Extra")]["mean_reward"].mean()
a3c_extra_without = df_without[(df_without["model"] == "A3C_Global") & (df_without["env_type"] == "Extra")]["mean_reward"].mean()
ind_extra_with = df_with[(df_with["model"].str.startswith("Individual")) & (df_with["env_type"] == "Extra")]["mean_reward"].mean()
ind_extra_without = df_without[(df_without["model"].str.startswith("Individual")) & (df_without["env_type"] == "Extra")]["mean_reward"].mean()

print(f"   A3C Extra:        {a3c_extra_with:.2f} vs {a3c_extra_without:.2f} ({((a3c_extra_with - a3c_extra_without) / a3c_extra_without) * 100:+.1f}%)")
print(f"   Individual Extra: {ind_extra_with:.2f} vs {ind_extra_without:.2f} ({((ind_extra_with - ind_extra_without) / ind_extra_without) * 100:+.1f}%)")

# 3. Stability
print("\n3. Performance Stability (Variance reduction):")
print(f"   A3C:        {((a3c_without_var - a3c_with_var) / a3c_without_var) * 100:+.1f}%")
print(f"   Individual: {((ind_without_var - ind_with_var) / ind_without_var) * 100:+.1f}%")

# 4. Worker-specific behavior
print("\n4. Individual Worker Behavior:")
print(f"   Workers with GOOD generalization (gap ≥ 0):")
good_with = sum(1 for g in worker_gaps_with if g >= 0)
good_without = sum(1 for g in worker_gaps_without if g >= 0)
print(f"     With LN:    {good_with}/5 workers")
print(f"     Without LN: {good_without}/5 workers")

print("\n" + "=" * 80)
print("Analysis Complete!")
print("=" * 80)
