"""
Step-level 로깅 데이터 분석
각 워커의 매 스텝마다 로컬/클라우드 comp_units와 선택한 action 분석
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

# 최신 학습 결과 찾기
a3c_runs = sorted(glob.glob('runs/a3c_*/A3C_worker_*_step_log.csv'))

if not a3c_runs:
    print("Error: No step log files found!")
    print("Please run main_train.py first")
    exit(1)

# 가장 최신 run의 디렉토리 찾기
latest_run_dir = os.path.dirname(a3c_runs[-1])
print(f"Analyzing step logs from: {latest_run_dir}")
print()

# 모든 워커의 step log 로드
step_logs = []
for worker_file in glob.glob(os.path.join(latest_run_dir, 'A3C_worker_*_step_log.csv')):
    df = pd.read_csv(worker_file)
    step_logs.append(df)

if not step_logs:
    print("Error: No step log data found!")
    exit(1)

df_all = pd.concat(step_logs, ignore_index=True)

print("="*80)
print("=== Step-Level Analysis ===")
print("="*80)
print(f"Total steps logged: {len(df_all):,}")
print(f"Workers: {df_all['worker_id'].nunique()}")
print(f"Episodes: {df_all['episode'].max()}")
print()

# 1. Action distribution
print("[1] Action Distribution")
print("-"*80)

action_names = {0: 'LOCAL', 1: 'OFFLOAD', 2: 'DISCARD'}
action_counts = df_all['action'].value_counts().sort_index()

for action_id, count in action_counts.items():
    action_name = action_names.get(action_id, f'ACTION_{action_id}')
    percentage = count / len(df_all) * 100
    print(f"  {action_name:10s}: {count:6,} ({percentage:5.2f}%)")

print()

# 2. Worker별 action distribution
print("[2] Worker-wise Action Distribution")
print("-"*80)

for worker_id in sorted(df_all['worker_id'].unique()):
    worker_data = df_all[df_all['worker_id'] == worker_id]
    print(f"\nWorker {worker_id}:")

    action_dist = worker_data['action'].value_counts(normalize=True).sort_index()
    for action_id, prob in action_dist.items():
        action_name = action_names.get(action_id, f'ACTION_{action_id}')
        print(f"  {action_name:10s}: {prob*100:5.2f}%")

print()

# 3. Local vs Cloud comp_units 통계
print("[3] Computation Units Statistics")
print("-"*80)

print(f"Local comp_units:")
print(f"  Mean  : {df_all['local_comp_units'].mean():.2f}")
print(f"  Median: {df_all['local_comp_units'].median():.2f}")
print(f"  Min   : {df_all['local_comp_units'].min():.2f}")
print(f"  Max   : {df_all['local_comp_units'].max():.2f}")
print(f"  Std   : {df_all['local_comp_units'].std():.2f}")

print(f"\nCloud comp_units:")
print(f"  Mean  : {df_all['cloud_comp_units'].mean():.2f}")
print(f"  Median: {df_all['cloud_comp_units'].median():.2f}")
print(f"  Min   : {df_all['cloud_comp_units'].min():.2f}")
print(f"  Max   : {df_all['cloud_comp_units'].max():.2f}")
print(f"  Std   : {df_all['cloud_comp_units'].std():.2f}")

print()

# 4. Action별 평균 reward
print("[4] Average Reward by Action")
print("-"*80)

for action_id in sorted(df_all['action'].unique()):
    action_name = action_names.get(action_id, f'ACTION_{action_id}')
    avg_reward = df_all[df_all['action'] == action_id]['reward'].mean()
    std_reward = df_all[df_all['action'] == action_id]['reward'].std()
    print(f"  {action_name:10s}: {avg_reward:7.4f} ± {std_reward:.4f}")

print()

# 5. Episode별 action 변화 추이 (처음 100 episodes)
print("[5] Action Distribution Over Episodes (First 100)")
print("-"*80)

episode_action_dist = df_all[df_all['episode'] <= 100].groupby(['episode', 'action']).size().unstack(fill_value=0)
episode_action_dist = episode_action_dist.div(episode_action_dist.sum(axis=1), axis=0) * 100

print(f"Episode 1-10 avg:")
for action_id in sorted(df_all['action'].unique()):
    if action_id in episode_action_dist.columns:
        avg_pct = episode_action_dist[action_id].iloc[:10].mean()
        action_name = action_names.get(action_id, f'ACTION_{action_id}')
        print(f"  {action_name:10s}: {avg_pct:5.2f}%")

print(f"\nEpisode 91-100 avg:")
for action_id in sorted(df_all['action'].unique()):
    if action_id in episode_action_dist.columns:
        avg_pct = episode_action_dist[action_id].iloc[90:100].mean()
        action_name = action_names.get(action_id, f'ACTION_{action_id}')
        print(f"  {action_name:10s}: {avg_pct:5.2f}%")

print()

# ========================================
# 시각화
# ========================================

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# (1) Overall Action Distribution
ax1 = axes[0, 0]
action_labels = [action_names.get(i, f'ACTION_{i}') for i in sorted(action_counts.index)]
colors = ['#3498db', '#e74c3c', '#f39c12']
ax1.bar(action_labels, action_counts.values, color=colors)
ax1.set_ylabel('Count', fontsize=12)
ax1.set_title('Overall Action Distribution', fontsize=14, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# (2) Worker-wise Action Distribution (Stacked)
ax2 = axes[0, 1]
worker_action_matrix = []
workers = sorted(df_all['worker_id'].unique())
for worker_id in workers:
    worker_data = df_all[df_all['worker_id'] == worker_id]
    action_dist = worker_data['action'].value_counts(normalize=True).sort_index()
    worker_action_matrix.append([action_dist.get(i, 0) * 100 for i in range(3)])

worker_action_df = pd.DataFrame(worker_action_matrix,
                                 columns=['LOCAL', 'OFFLOAD', 'DISCARD'],
                                 index=[f'W{i}' for i in workers])
worker_action_df.plot(kind='bar', stacked=True, ax=ax2, color=colors)
ax2.set_ylabel('Percentage (%)', fontsize=12)
ax2.set_xlabel('Worker', fontsize=12)
ax2.set_title('Worker-wise Action Distribution', fontsize=14, fontweight='bold')
ax2.legend(title='Action')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)

# (3) Local vs Cloud Comp Units (Scatter)
ax3 = axes[0, 2]
sample_data = df_all.sample(min(5000, len(df_all)))  # 샘플링
scatter = ax3.scatter(sample_data['local_comp_units'],
                     sample_data['cloud_comp_units'],
                     c=sample_data['action'],
                     cmap='viridis',
                     alpha=0.5,
                     s=10)
ax3.set_xlabel('Local Comp Units', fontsize=12)
ax3.set_ylabel('Cloud Comp Units', fontsize=12)
ax3.set_title('Local vs Cloud Comp Units (colored by action)', fontsize=14, fontweight='bold')
cbar = plt.colorbar(scatter, ax=ax3)
cbar.set_label('Action', fontsize=10)
ax3.grid(True, alpha=0.3)

# (4) Action Distribution Over Episodes
ax4 = axes[1, 0]
if len(episode_action_dist) > 0:
    for action_id in sorted(df_all['action'].unique()):
        if action_id in episode_action_dist.columns:
            action_name = action_names.get(action_id, f'ACTION_{action_id}')
            ax4.plot(episode_action_dist.index,
                    episode_action_dist[action_id].rolling(window=10).mean(),
                    label=action_name, linewidth=2)
ax4.set_xlabel('Episode', fontsize=12)
ax4.set_ylabel('Percentage (%)', fontsize=12)
ax4.set_title('Action Distribution Over Episodes (MA-10)', fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# (5) Average Reward by Action (Box plot)
ax5 = axes[1, 1]
reward_by_action = [df_all[df_all['action'] == i]['reward'].values
                   for i in sorted(df_all['action'].unique())]
bp = ax5.boxplot(reward_by_action,
                labels=[action_names.get(i, f'ACTION_{i}') for i in sorted(df_all['action'].unique())],
                patch_artist=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
ax5.set_ylabel('Reward', fontsize=12)
ax5.set_title('Reward Distribution by Action', fontsize=14, fontweight='bold')
ax5.grid(axis='y', alpha=0.3)

# (6) Comp Units Over Time (샘플 episode)
ax6 = axes[1, 2]
sample_episode = df_all[df_all['episode'] == 50]  # Episode 50 예시
if len(sample_episode) > 0:
    for worker_id in sample_episode['worker_id'].unique():
        worker_ep_data = sample_episode[sample_episode['worker_id'] == worker_id]
        ax6.plot(worker_ep_data['step'], worker_ep_data['local_comp_units'],
                label=f'Worker {worker_id} Local', alpha=0.7)
ax6.set_xlabel('Step', fontsize=12)
ax6.set_ylabel('Local Comp Units', fontsize=12)
ax6.set_title('Local Comp Units (Episode 50)', fontsize=14, fontweight='bold')
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3)

plt.tight_layout()

# Save
os.makedirs('figures', exist_ok=True)
output_path = 'figures/step_level_analysis.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"[Visualization saved] {output_path}")
print()

# CSV summary 저장
summary_data = {
    'metric': [],
    'value': []
}

summary_data['metric'].extend(['total_steps', 'total_workers', 'total_episodes'])
summary_data['value'].extend([len(df_all), df_all['worker_id'].nunique(), df_all['episode'].max()])

for action_id, count in action_counts.items():
    action_name = action_names.get(action_id, f'ACTION_{action_id}')
    summary_data['metric'].append(f'action_{action_name}_count')
    summary_data['value'].append(count)
    summary_data['metric'].append(f'action_{action_name}_pct')
    summary_data['value'].append(count / len(df_all) * 100)

summary_df = pd.DataFrame(summary_data)
summary_csv_path = 'figures/step_level_summary.csv'
summary_df.to_csv(summary_csv_path, index=False)
print(f"[Summary saved] {summary_csv_path}")
print()

print("="*80)
print("Analysis Complete!")
print("="*80)
