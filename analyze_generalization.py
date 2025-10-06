"""
A3C vs Individual 일반화 성능 비교 분석
Seen, Interpolation, Extrapolation 환경별 상세 분석
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

# 최신 generalization 결과 파일 찾기
result_files = sorted(glob.glob('generalization_results_v2_*.csv'))

if not result_files:
    print("Error: No generalization results found!")
    print("Please run test_generalization_v2.py first")
    exit(1)

latest_file = result_files[-1]
print(f"Loading results from: {latest_file}")
print()

df = pd.read_csv(latest_file)

print("="*80)
print("=== A3C vs Individual: Generalization Performance Analysis ===")
print("="*80)
print()

# 1. Seen vs Interpolation vs Extrapolation 성능 비교
print("[1] Performance by Environment Type")
print("-"*80)

for env_type in ['Seen', 'Interpolation', 'Extrapolation']:
    a3c_data = df[(df['model'] == 'A3C_Global') & (df['env_type'] == env_type)]
    ind_data = df[(df['model'] != 'A3C_Global') & (df['env_type'] == env_type)]

    a3c_mean = a3c_data['mean_reward'].mean()
    a3c_std = a3c_data['mean_reward'].std()

    ind_mean = ind_data['mean_reward'].mean()
    ind_std = ind_data['mean_reward'].std()

    diff = a3c_mean - ind_mean

    print(f"\n{env_type}:")
    print(f"  A3C Global   : {a3c_mean:.4f} ± {a3c_std:.4f}")
    print(f"  Individual   : {ind_mean:.4f} ± {ind_std:.4f}")
    print(f"  Difference   : {diff:+.4f} ({'A3C better' if diff > 0 else 'Individual better'})")
    print(f"  Relative (%) : {(diff/ind_mean)*100:+.2f}%")

print()

# 2. A3C의 환경별 안정성
print("[2] A3C Global - Stability Across Environment Types")
print("-"*80)

a3c_seen = df[(df['model'] == 'A3C_Global') & (df['env_type'] == 'Seen')]['mean_reward']
a3c_interp = df[(df['model'] == 'A3C_Global') & (df['env_type'] == 'Interpolation')]['mean_reward']
a3c_extrap = df[(df['model'] == 'A3C_Global') & (df['env_type'] == 'Extrapolation')]['mean_reward']

print(f"Seen          : mean={a3c_seen.mean():.4f}, std={a3c_seen.std():.4f}, CV={a3c_seen.std()/a3c_seen.mean():.4f}")
print(f"Interpolation : mean={a3c_interp.mean():.4f}, std={a3c_interp.std():.4f}, CV={a3c_interp.std()/a3c_interp.mean():.4f}")
print(f"Extrapolation : mean={a3c_extrap.mean():.4f}, std={a3c_extrap.std():.4f}, CV={a3c_extrap.std()/a3c_extrap.mean():.4f}")

print()

# 3. Individual의 환경별 안정성
print("[3] Individual Workers - Stability Across Environment Types")
print("-"*80)

ind_seen = df[(df['model'] != 'A3C_Global') & (df['env_type'] == 'Seen')]['mean_reward']
ind_interp = df[(df['model'] != 'A3C_Global') & (df['env_type'] == 'Interpolation')]['mean_reward']
ind_extrap = df[(df['model'] != 'A3C_Global') & (df['env_type'] == 'Extrapolation')]['mean_reward']

print(f"Seen          : mean={ind_seen.mean():.4f}, std={ind_seen.std():.4f}, CV={ind_seen.std()/ind_seen.mean():.4f}")
print(f"Interpolation : mean={ind_interp.mean():.4f}, std={ind_interp.std():.4f}, CV={ind_interp.std()/ind_interp.mean():.4f}")
print(f"Extrapolation : mean={ind_extrap.mean():.4f}, std={ind_extrap.std():.4f}, CV={ind_extrap.std()/ind_extrap.mean():.4f}")

print()

# 4. Worker별 일반화 능력 분석
print("[4] Individual Workers - Generalization Ability")
print("-"*80)

# 각 worker의 seen 환경 찾기
seen_envs = [
    (80, 20), (110, 40), (140, 60), (170, 80), (200, 100)
]

for worker_id in range(5):
    worker_data = df[df['worker_id'] == worker_id]

    if worker_data.empty:
        continue

    trained_comp, trained_vel = seen_envs[worker_id]

    # Seen 성능 (학습한 정확한 환경)
    seen_perf = worker_data[
        (worker_data['test_comp'] == trained_comp) &
        (worker_data['test_vel'] == trained_vel)
    ]['mean_reward'].values

    seen_perf = seen_perf[0] if len(seen_perf) > 0 else 0

    # Interpolation 평균
    interp_perf = worker_data[worker_data['env_type'] == 'Interpolation']['mean_reward'].mean()

    # Extrapolation 평균
    extrap_perf = worker_data[worker_data['env_type'] == 'Extrapolation']['mean_reward'].mean()

    # 일반화 갭
    interp_gap = seen_perf - interp_perf
    extrap_gap = seen_perf - extrap_perf

    print(f"\nWorker {worker_id} (trained on comp={trained_comp}, vel={trained_vel}):")
    print(f"  Seen          : {seen_perf:.4f}")
    print(f"  Interpolation : {interp_perf:.4f} (gap: {interp_gap:+.4f})")
    print(f"  Extrapolation : {extrap_perf:.4f} (gap: {extrap_gap:+.4f})")

print()

# 5. 최악/최고 시나리오 분석
print("[5] Best/Worst Case Analysis by Environment Type")
print("-"*80)

for env_type in ['Seen', 'Interpolation', 'Extrapolation']:
    print(f"\n{env_type}:")

    # A3C
    a3c_data = df[(df['model'] == 'A3C_Global') & (df['env_type'] == env_type)]
    if not a3c_data.empty:
        worst_a3c = a3c_data.loc[a3c_data['mean_reward'].idxmin()]
        best_a3c = a3c_data.loc[a3c_data['mean_reward'].idxmax()]

        print(f"  A3C Global:")
        print(f"    Best : comp={best_a3c['test_comp']}, vel={best_a3c['test_vel']}, reward={best_a3c['mean_reward']:.4f}")
        print(f"    Worst: comp={worst_a3c['test_comp']}, vel={worst_a3c['test_vel']}, reward={worst_a3c['mean_reward']:.4f}")

    # Individual
    ind_data = df[(df['model'] != 'A3C_Global') & (df['env_type'] == env_type)]
    if not ind_data.empty:
        worst_ind = ind_data.loc[ind_data['mean_reward'].idxmin()]
        best_ind = ind_data.loc[ind_data['mean_reward'].idxmax()]

        print(f"  Individual:")
        print(f"    Best : worker={int(best_ind['worker_id'])}, comp={best_ind['test_comp']}, vel={best_ind['test_vel']}, reward={best_ind['mean_reward']:.4f}")
        print(f"    Worst: worker={int(worst_ind['worker_id'])}, comp={worst_ind['test_comp']}, vel={worst_ind['test_vel']}, reward={worst_ind['mean_reward']:.4f}")

print()

# ========================================
# 시각화
# ========================================

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# (1) Environment Type별 성능 비교
ax1 = axes[0, 0]
env_comparison = []
for env_type in ['Seen', 'Interpolation', 'Extrapolation']:
    a3c_mean = df[(df['model'] == 'A3C_Global') & (df['env_type'] == env_type)]['mean_reward'].mean()
    ind_mean = df[(df['model'] != 'A3C_Global') & (df['env_type'] == env_type)]['mean_reward'].mean()
    env_comparison.append({'Type': env_type, 'A3C': a3c_mean, 'Individual': ind_mean})

env_df = pd.DataFrame(env_comparison)
x = np.arange(len(env_df))
width = 0.35
ax1.bar(x - width/2, env_df['A3C'], width, label='A3C Global', color='#3498db')
ax1.bar(x + width/2, env_df['Individual'], width, label='Individual', color='#e74c3c')
ax1.set_xlabel('Environment Type', fontsize=12)
ax1.set_ylabel('Mean Reward', fontsize=12)
ax1.set_title('Performance by Environment Type', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(env_df['Type'])
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# (2) Violin plot - A3C
ax2 = axes[0, 1]
a3c_plot_data = []
for env_type in ['Seen', 'Interpolation', 'Extrapolation']:
    data = df[(df['model'] == 'A3C_Global') & (df['env_type'] == env_type)]['mean_reward'].values
    a3c_plot_data.extend([{'Type': env_type, 'Reward': r} for r in data])

a3c_plot_df = pd.DataFrame(a3c_plot_data)
sns.violinplot(data=a3c_plot_df, x='Type', y='Reward', ax=ax2, color='#3498db')
ax2.set_title('A3C Global - Reward Distribution', fontsize=14, fontweight='bold')
ax2.set_xlabel('Environment Type', fontsize=12)
ax2.set_ylabel('Reward', fontsize=12)
ax2.grid(axis='y', alpha=0.3)

# (3) Violin plot - Individual
ax3 = axes[0, 2]
ind_plot_data = []
for env_type in ['Seen', 'Interpolation', 'Extrapolation']:
    data = df[(df['model'] != 'A3C_Global') & (df['env_type'] == env_type)]['mean_reward'].values
    ind_plot_data.extend([{'Type': env_type, 'Reward': r} for r in data])

ind_plot_df = pd.DataFrame(ind_plot_data)
sns.violinplot(data=ind_plot_df, x='Type', y='Reward', ax=ax3, color='#e74c3c')
ax3.set_title('Individual - Reward Distribution', fontsize=14, fontweight='bold')
ax3.set_xlabel('Environment Type', fontsize=12)
ax3.set_ylabel('Reward', fontsize=12)
ax3.grid(axis='y', alpha=0.3)

# (4) Generalization Gap - A3C
ax4 = axes[1, 0]
a3c_seen_mean = df[(df['model'] == 'A3C_Global') & (df['env_type'] == 'Seen')]['mean_reward'].mean()
a3c_gaps = {
    'Interpolation': a3c_seen_mean - df[(df['model'] == 'A3C_Global') & (df['env_type'] == 'Interpolation')]['mean_reward'].mean(),
    'Extrapolation': a3c_seen_mean - df[(df['model'] == 'A3C_Global') & (df['env_type'] == 'Extrapolation')]['mean_reward'].mean()
}
colors = ['#f39c12' if gap > 0 else '#2ecc71' for gap in a3c_gaps.values()]
ax4.bar(a3c_gaps.keys(), a3c_gaps.values(), color=colors, alpha=0.7)
ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax4.set_title('A3C Global - Generalization Gap', fontsize=14, fontweight='bold')
ax4.set_ylabel('Performance Gap (Seen - Unseen)', fontsize=12)
ax4.grid(axis='y', alpha=0.3)

# (5) Generalization Gap - Individual
ax5 = axes[1, 1]
ind_seen_mean = df[(df['model'] != 'A3C_Global') & (df['env_type'] == 'Seen')]['mean_reward'].mean()
ind_gaps = {
    'Interpolation': ind_seen_mean - df[(df['model'] != 'A3C_Global') & (df['env_type'] == 'Interpolation')]['mean_reward'].mean(),
    'Extrapolation': ind_seen_mean - df[(df['model'] != 'A3C_Global') & (df['env_type'] == 'Extrapolation')]['mean_reward'].mean()
}
colors = ['#f39c12' if gap > 0 else '#2ecc71' for gap in ind_gaps.values()]
ax5.bar(ind_gaps.keys(), ind_gaps.values(), color=colors, alpha=0.7)
ax5.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax5.set_title('Individual - Generalization Gap', fontsize=14, fontweight='bold')
ax5.set_ylabel('Performance Gap (Seen - Unseen)', fontsize=12)
ax5.grid(axis='y', alpha=0.3)

# (6) Coefficient of Variation (안정성)
ax6 = axes[1, 2]
cv_data = []
for env_type in ['Seen', 'Interpolation', 'Extrapolation']:
    a3c_data = df[(df['model'] == 'A3C_Global') & (df['env_type'] == env_type)]['mean_reward']
    ind_data = df[(df['model'] != 'A3C_Global') & (df['env_type'] == env_type)]['mean_reward']

    a3c_cv = a3c_data.std() / a3c_data.mean() if a3c_data.mean() > 0 else 0
    ind_cv = ind_data.std() / ind_data.mean() if ind_data.mean() > 0 else 0

    cv_data.append({'Type': env_type, 'A3C': a3c_cv, 'Individual': ind_cv})

cv_df = pd.DataFrame(cv_data)
x = np.arange(len(cv_df))
ax6.bar(x - width/2, cv_df['A3C'], width, label='A3C Global', color='#3498db')
ax6.bar(x + width/2, cv_df['Individual'], width, label='Individual', color='#e74c3c')
ax6.set_xlabel('Environment Type', fontsize=12)
ax6.set_ylabel('Coefficient of Variation', fontsize=12)
ax6.set_title('Stability Comparison (Lower is Better)', fontsize=14, fontweight='bold')
ax6.set_xticks(x)
ax6.set_xticklabels(cv_df['Type'])
ax6.legend()
ax6.grid(axis='y', alpha=0.3)

plt.tight_layout()
output_file = 'generalization_analysis_comparison.png'
plt.savefig(output_file, dpi=180, bbox_inches='tight')
print(f"[Visualization saved] {output_file}")
print()

print("="*80)
print("Analysis Complete!")
print("="*80)
