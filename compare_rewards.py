import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Timestamp for unique directory naming
# timestamp = "20251005_170221"
timestamp = "20251007_022054"

# 최신 run 디렉토리
a3c_dir = f'runs/a3c_{timestamp}'
individual_dir = f'runs/individual_{timestamp}'

print('='*70)
print('=== A3C vs Individual 학습 Reward 비교 분석 ===')
print('='*70)
print()

# 1. .pth 파일에서 메타데이터 확인
print('='*70)
print('=== 모델 메타데이터 비교 ===')
print('='*70)

a3c_model_path = os.path.join(a3c_dir, 'models/global_final.pth')
a3c_checkpoint = torch.load(a3c_model_path, map_location='cpu')

print(f'[A3C Global Model]')
if 'model_state_dict' in a3c_checkpoint:
    print(f'  최종 Episode    : {a3c_checkpoint.get("episode", "N/A")}')
    print(f'  총 Steps        : {a3c_checkpoint.get("total_steps", "N/A"):,}')
    print(f'  최종 Reward     : {a3c_checkpoint.get("final_reward", "N/A"):.2f}')
    print(f'  Workers         : {a3c_checkpoint.get("n_workers", "N/A")}')
    print(f'  Timestamp       : {a3c_checkpoint.get("timestamp", "N/A")}')
else:
    print('  메타데이터 없음 (구버전 모델)')
print()

# Individual 모델들 메타데이터
print(f'[Individual Models]')
for worker_id in range(5):
    ind_model_path = os.path.join(individual_dir, f'models/individual_worker_{worker_id}_final.pth')
    if os.path.exists(ind_model_path):
        ind_checkpoint = torch.load(ind_model_path, map_location='cpu')
        if 'model_state_dict' in ind_checkpoint:
            print(f'  Worker {worker_id}: Episode={ind_checkpoint.get("episode", "N/A")}, '
                  f'Steps={ind_checkpoint.get("total_steps", "N/A"):,}, '
                  f'Final Reward={ind_checkpoint.get("final_reward", "N/A"):.2f}')
        else:
            print(f'  Worker {worker_id}: 메타데이터 없음')
print()

# 2. training_log.csv 로드 및 분석
print('='*70)
print('=== Training Log 기반 Reward 분석 ===')
print('='*70)

a3c_csv = os.path.join(a3c_dir, 'training_log.csv')
ind_csv = os.path.join(individual_dir, 'training_log.csv')

df_a3c = pd.read_csv(a3c_csv)
df_ind = pd.read_csv(ind_csv)

# Episode별 평균 reward 계산
a3c_reward = df_a3c.groupby('episode')['reward'].mean().reset_index()
a3c_reward.columns = ['episode', 'a3c_reward']

ind_reward = df_ind.groupby('episode')['reward'].mean().reset_index()
ind_reward.columns = ['episode', 'ind_reward']

# 통계 요약
print(f'[A3C Global]')
print(f'  평균 Reward     : {a3c_reward["a3c_reward"].mean():.2f}')
print(f'  최종 Reward     : {a3c_reward["a3c_reward"].iloc[-1]:.2f}')
print(f'  최대 Reward     : {a3c_reward["a3c_reward"].max():.2f}')
print(f'  최소 Reward     : {a3c_reward["a3c_reward"].min():.2f}')
print(f'  표준편차        : {a3c_reward["a3c_reward"].std():.2f}')
print()

print(f'[Individual A2C]')
print(f'  평균 Reward     : {ind_reward["ind_reward"].mean():.2f}')
print(f'  최종 Reward     : {ind_reward["ind_reward"].iloc[-1]:.2f}')
print(f'  최대 Reward     : {ind_reward["ind_reward"].max():.2f}')
print(f'  최소 Reward     : {ind_reward["ind_reward"].min():.2f}')
print(f'  표준편차        : {ind_reward["ind_reward"].std():.2f}')
print()

# 3. Worker별 성능 비교
print('='*70)
print('=== Worker별 최종 성능 비교 ===')
print('='*70)

a3c_final = df_a3c[df_a3c['episode'] == df_a3c['episode'].max()].groupby('worker_id')['reward'].mean()
ind_final = df_ind[df_ind['episode'] == df_ind['episode'].max()].groupby('worker_id')['reward'].mean()

print(f'{"Worker":<10} {"A3C Reward":>15} {"Individual Reward":>20} {"차이":>15}')
print('-'*70)
for wid in range(5):
    a3c_r = a3c_final.get(wid, 0)
    ind_r = ind_final.get(wid, 0)
    diff = a3c_r - ind_r
    print(f'{wid:<10} {a3c_r:>15.2f} {ind_r:>20.2f} {diff:>15.2f}')
print()

# 4. 시각화
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# (1) 전체 Reward 곡선 비교
ax1 = axes[0, 0]
ax1.plot(a3c_reward['episode'], a3c_reward['a3c_reward'], label='A3C Global', linewidth=2, alpha=0.8)
ax1.plot(ind_reward['episode'], ind_reward['ind_reward'], label='Individual A2C', linewidth=2, alpha=0.8)
ax1.set_xlabel('Episode', fontsize=12)
ax1.set_ylabel('Average Reward', fontsize=12)
ax1.set_title('A3C vs Individual A2C - Episode Reward', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# (2) Moving Average (window=100)
window = 50
a3c_ma = a3c_reward['a3c_reward'].rolling(window=window, min_periods=1).mean()
ind_ma = ind_reward['ind_reward'].rolling(window=window, min_periods=1).mean()

ax2 = axes[0, 1]
ax2.plot(a3c_reward['episode'], a3c_ma, label=f'A3C (MA-{window})', linewidth=2, alpha=0.8)
ax2.plot(ind_reward['episode'], ind_ma, label=f'Individual (MA-{window})', linewidth=2, alpha=0.8)
ax2.set_xlabel('Episode', fontsize=12)
ax2.set_ylabel('Moving Average Reward', fontsize=12)
ax2.set_title(f'Moving Average Comparison (window={window})', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

# (3) Worker별 개별 곡선 - A3C
ax3 = axes[1, 0]
for wid in sorted(df_a3c['worker_id'].unique()):
    worker_data = df_a3c[df_a3c['worker_id'] == wid].groupby('episode')['reward'].mean()
    ax3.plot(worker_data.index, worker_data.values, label=f'Worker {int(wid)}', alpha=0.6)
ax3.set_xlabel('Episode', fontsize=12)
ax3.set_ylabel('Reward', fontsize=12)
ax3.set_title('A3C - Worker별 Reward', fontsize=14, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# (4) Worker별 개별 곡선 - Individual
ax4 = axes[1, 1]
for wid in sorted(df_ind['worker_id'].unique()):
    worker_data = df_ind[df_ind['worker_id'] == wid].groupby('episode')['reward'].mean()
    ax4.plot(worker_data.index, worker_data.values, label=f'Worker {int(wid)}', alpha=0.6)
ax4.set_xlabel('Episode', fontsize=12)
ax4.set_ylabel('Reward', fontsize=12)
ax4.set_title('Individual A2C - Worker별 Reward', fontsize=14, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
output_path = 'reward_comparison_a3c_vs_individual.png'
plt.savefig(output_path, dpi=180, bbox_inches='tight')
print(f'[시각화 저장] {output_path}')
print()

# Save ax2 (Moving Average Comparison) separately
os.makedirs('figures', exist_ok=True)

fig_ax2, ax_single = plt.subplots(1, 1, figsize=(10, 6))
ax_single.plot(a3c_reward['episode'], a3c_ma, label=f'A3C (MA-{window})', linewidth=2, alpha=0.8)
ax_single.plot(ind_reward['episode'], ind_ma, label=f'Individual (MA-{window})', linewidth=2, alpha=0.8)
ax_single.set_xlabel('Episode', fontsize=12)
ax_single.set_ylabel('Moving Average Reward', fontsize=12)
ax_single.set_title(f'Moving Average Comparison (window={window})', fontsize=14, fontweight='bold')
ax_single.legend(fontsize=11)
ax_single.grid(True, alpha=0.3)

# Save as PNG and EPS
png_path = 'figures/moving_average_comparison.png'
eps_path = 'figures/moving_average_comparison.eps'
fig_ax2.savefig(png_path, dpi=300, bbox_inches='tight')
fig_ax2.savefig(eps_path, format='eps', bbox_inches='tight')
plt.close(fig_ax2)

print(f'[ax2 saved separately]')
print(f'  PNG: {png_path}')
print(f'  EPS: {eps_path}')
print()

# 5. 학습 안정성 분석 (최근 500 episode 기준)
print('='*70)
print('=== 학습 안정성 분석 (최근 500 Episode) ===')
print('='*70)

recent_episodes = 500
a3c_recent = a3c_reward.tail(recent_episodes)['a3c_reward']
ind_recent = ind_reward.tail(recent_episodes)['ind_reward']

print(f'[A3C Global - 최근 {recent_episodes} episodes]')
print(f'  평균            : {a3c_recent.mean():.2f}')
print(f'  표준편차        : {a3c_recent.std():.2f}')
print(f'  변동계수 (CV)   : {a3c_recent.std() / a3c_recent.mean():.4f}')
print()

print(f'[Individual A2C - 최근 {recent_episodes} episodes]')
print(f'  평균            : {ind_recent.mean():.2f}')
print(f'  표준편차        : {ind_recent.std():.2f}')
print(f'  변동계수 (CV)   : {ind_recent.std() / ind_recent.mean():.4f}')
print()

# 6. 수렴 속도 분석 (목표 reward 도달 시점)
target_reward = a3c_reward['a3c_reward'].max() * 0.9  # 최대값의 90%

a3c_converge = a3c_reward[a3c_reward['a3c_reward'] >= target_reward]['episode'].min()
ind_converge = ind_reward[ind_reward['ind_reward'] >= target_reward]['episode'].min()

print('='*70)
print(f'=== 수렴 속도 비교 (목표: {target_reward:.2f}) ===')
print('='*70)
print(f'A3C Global    : Episode {a3c_converge}')
print(f'Individual A2C: Episode {ind_converge}')
if pd.notna(a3c_converge) and pd.notna(ind_converge):
    diff_episodes = int(a3c_converge - ind_converge)
    faster = 'A3C' if diff_episodes < 0 else 'Individual'
    print(f'더 빠른 모델   : {faster} (차이: {abs(diff_episodes)} episodes)')
print()

print('='*70)
print('분석 완료!')
print('='*70)
