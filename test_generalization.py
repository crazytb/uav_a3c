"""
일반화 성능 테스트: 학습 환경과 다른 환경에서 모델 평가
- A3C Global vs Individual Workers
- 학습한 환경 vs 새로운(unseen) 환경
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from drl_framework.networks import RecurrentActorCritic
from drl_framework.custom_env import make_env
from drl_framework.utils import flatten_dict_values
from drl_framework.params import *
from torch.distributions import Categorical

# 최신 학습 결과 타임스탬프
TIMESTAMP = "20251005_170221"

# 환경 설정
temp_env_fn = make_env(**ENV_PARAMS)
temp_env = temp_env_fn()
sample_obs = temp_env.reset()[0]
state_dim = len(flatten_dict_values(sample_obs))
action_dim = temp_env.action_space.n
temp_env.close()

def load_model(model_path):
    """모델 로드 (메타데이터 자동 처리)"""
    model = RecurrentActorCritic(state_dim, action_dim, hidden_dim).to(device)
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model

def evaluate_model(model, env_params, n_episodes=50):
    """단일 환경에서 모델 평가"""
    env_fn = make_env(**env_params, reward_params=REWARD_PARAMS)
    env = env_fn()

    episode_rewards = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        hx = model.init_hidden(batch_size=1, device=device)

        while not done and steps < ENV_PARAMS['max_epoch_size']:
            obs_tensor = torch.as_tensor(
                flatten_dict_values(obs), dtype=torch.float32, device=device
            ).unsqueeze(0)

            with torch.no_grad():
                logits, value, hx = model.step(obs_tensor, hx)
                action = torch.argmax(logits, dim=-1).item()  # Greedy

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            total_reward += float(reward)
            steps += 1

            if done:
                hx = hx * 0.0

        episode_rewards.append(total_reward)

    env.close()
    return np.mean(episode_rewards), np.std(episode_rewards)

# ========================================
# 실험 설정
# ========================================

print("="*80)
print("=== 일반화 성능 테스트: 학습 환경 vs 새로운 환경 ===")
print("="*80)
print()

# 1. 학습 시 사용된 환경 (Seen environments)
SEEN_ENVS = [80, 110, 140, 170, 200]

# 2. 새로운 환경 (Unseen environments)
UNSEEN_ENVS = [50, 95, 125, 155, 185, 220]  # 학습 범위 밖/사이 값

# 3. 전체 테스트 환경
ALL_TEST_ENVS = sorted(SEEN_ENVS + UNSEEN_ENVS)

print(f"학습 환경 (Seen)    : {SEEN_ENVS}")
print(f"새로운 환경 (Unseen): {UNSEEN_ENVS}")
print(f"전체 테스트 환경    : {ALL_TEST_ENVS}")
print()

# ========================================
# 모델 평가
# ========================================

results = []

# A3C Global 모델
a3c_model_path = f"runs/a3c_{TIMESTAMP}/models/global_final.pth"
print(f"[1/6] A3C Global 모델 로드 중...")
a3c_model = load_model(a3c_model_path)

for comp_units in ALL_TEST_ENVS:
    env_params = ENV_PARAMS.copy()
    env_params['max_comp_units'] = comp_units

    print(f"  Testing max_comp_units={comp_units}...", end=" ")
    mean_reward, std_reward = evaluate_model(a3c_model, env_params)

    results.append({
        'model': 'A3C_Global',
        'worker_id': 'Global',
        'trained_on': 'All (80-200)',
        'test_env': comp_units,
        'env_type': 'Seen' if comp_units in SEEN_ENVS else 'Unseen',
        'mean_reward': mean_reward,
        'std_reward': std_reward
    })
    print(f"Reward: {mean_reward:.2f} ± {std_reward:.2f}")

print()

# Individual Workers (각 워커별)
for worker_id in range(5):
    trained_comp_units = SEEN_ENVS[worker_id]
    ind_model_path = f"runs/individual_{TIMESTAMP}/models/individual_worker_{worker_id}_final.pth"

    print(f"[{2+worker_id}/6] Individual Worker {worker_id} (trained on {trained_comp_units}) 로드 중...")
    ind_model = load_model(ind_model_path)

    for comp_units in ALL_TEST_ENVS:
        env_params = ENV_PARAMS.copy()
        env_params['max_comp_units'] = comp_units

        print(f"  Testing max_comp_units={comp_units}...", end=" ")
        mean_reward, std_reward = evaluate_model(ind_model, env_params)

        results.append({
            'model': f'Individual_W{worker_id}',
            'worker_id': worker_id,
            'trained_on': trained_comp_units,
            'test_env': comp_units,
            'env_type': 'Seen' if comp_units in SEEN_ENVS else 'Unseen',
            'mean_reward': mean_reward,
            'std_reward': std_reward
        })
        print(f"Reward: {mean_reward:.2f} ± {std_reward:.2f}")

    print()

# ========================================
# 결과 분석 및 시각화
# ========================================

df = pd.DataFrame(results)

print("="*80)
print("=== 일반화 성능 분석 ===")
print("="*80)
print()

# 1. Seen vs Unseen 환경 성능 비교
print("[1] Seen vs Unseen 환경 평균 성능")
print("-"*80)

a3c_seen = df[(df['model'] == 'A3C_Global') & (df['env_type'] == 'Seen')]['mean_reward'].mean()
a3c_unseen = df[(df['model'] == 'A3C_Global') & (df['env_type'] == 'Unseen')]['mean_reward'].mean()

ind_seen = df[(df['model'] != 'A3C_Global') & (df['env_type'] == 'Seen')]['mean_reward'].mean()
ind_unseen = df[(df['model'] != 'A3C_Global') & (df['env_type'] == 'Unseen')]['mean_reward'].mean()

print(f"A3C Global:")
print(f"  Seen   환경: {a3c_seen:.2f}")
print(f"  Unseen 환경: {a3c_unseen:.2f}")
print(f"  성능 하락: {a3c_seen - a3c_unseen:.2f} ({(a3c_unseen/a3c_seen - 1)*100:.1f}%)")
print()

print(f"Individual Workers (평균):")
print(f"  Seen   환경: {ind_seen:.2f}")
print(f"  Unseen 환경: {ind_unseen:.2f}")
print(f"  성능 하락: {ind_seen - ind_unseen:.2f} ({(ind_unseen/ind_seen - 1)*100:.1f}%)")
print()

# 2. Worker별 일반화 성능
print("[2] Individual Worker별 일반화 성능")
print("-"*80)

for worker_id in range(5):
    trained_env = SEEN_ENVS[worker_id]
    worker_data = df[df['worker_id'] == worker_id]

    # 학습 환경에서의 성능
    trained_perf = worker_data[worker_data['test_env'] == trained_env]['mean_reward'].values[0]

    # Unseen 환경에서의 성능
    unseen_perf = worker_data[worker_data['env_type'] == 'Unseen']['mean_reward'].mean()

    print(f"Worker {worker_id} (trained on {trained_env}):")
    print(f"  학습 환경 성능: {trained_perf:.2f}")
    print(f"  Unseen 평균 : {unseen_perf:.2f}")
    print(f"  성능 하락    : {trained_perf - unseen_perf:.2f} ({(unseen_perf/trained_perf - 1)*100:.1f}%)")

print()

# ========================================
# 시각화
# ========================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# (1) 전체 모델 성능 비교 (Heatmap)
ax1 = axes[0, 0]
pivot_data = df.pivot_table(
    values='mean_reward',
    index='model',
    columns='test_env',
    aggfunc='mean'
)
sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax1, cbar_kws={'label': 'Mean Reward'})
ax1.set_title('모델별 환경별 성능 (Heatmap)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Test Environment (max_comp_units)', fontsize=12)
ax1.set_ylabel('Model', fontsize=12)

# (2) A3C vs Individual: Seen vs Unseen
ax2 = axes[0, 1]
comparison_data = pd.DataFrame({
    'A3C Seen': [a3c_seen],
    'A3C Unseen': [a3c_unseen],
    'Individual Seen': [ind_seen],
    'Individual Unseen': [ind_unseen]
})
comparison_data.T.plot(kind='bar', ax=ax2, legend=False, color=['#2ecc71', '#e74c3c', '#3498db', '#f39c12'])
ax2.set_title('Seen vs Unseen 환경 성능 비교', fontsize=14, fontweight='bold')
ax2.set_ylabel('Mean Reward', fontsize=12)
ax2.set_xlabel('')
ax2.set_xticklabels(['A3C\nSeen', 'A3C\nUnseen', 'Individual\nSeen', 'Individual\nUnseen'], rotation=0)
ax2.grid(axis='y', alpha=0.3)

# (3) 환경별 성능 곡선
ax3 = axes[1, 0]

# A3C Global
a3c_data = df[df['model'] == 'A3C_Global'].sort_values('test_env')
ax3.plot(a3c_data['test_env'], a3c_data['mean_reward'],
         marker='o', linewidth=2, markersize=8, label='A3C Global', color='#2ecc71')

# Individual Workers (각각)
for worker_id in range(5):
    worker_data = df[df['worker_id'] == worker_id].sort_values('test_env')
    trained_env = SEEN_ENVS[worker_id]
    ax3.plot(worker_data['test_env'], worker_data['mean_reward'],
             marker='s', linewidth=1.5, markersize=6, alpha=0.6,
             label=f'Worker {worker_id} (trained@{trained_env})')

# Seen 환경 표시
for env in SEEN_ENVS:
    ax3.axvline(x=env, color='gray', linestyle='--', alpha=0.3, linewidth=1)

ax3.set_title('환경별 성능 곡선', fontsize=14, fontweight='bold')
ax3.set_xlabel('Test Environment (max_comp_units)', fontsize=12)
ax3.set_ylabel('Mean Reward', fontsize=12)
ax3.legend(fontsize=9, loc='best')
ax3.grid(True, alpha=0.3)

# (4) 일반화 갭 비교 (Generalization Gap)
ax4 = axes[1, 1]

generalization_gap = []
for worker_id in range(5):
    worker_data = df[df['worker_id'] == worker_id]
    trained_env = SEEN_ENVS[worker_id]

    trained_perf = worker_data[worker_data['test_env'] == trained_env]['mean_reward'].values[0]
    unseen_perf = worker_data[worker_data['env_type'] == 'Unseen']['mean_reward'].mean()

    gap = trained_perf - unseen_perf
    generalization_gap.append({
        'Model': f'Worker {worker_id}',
        'Gap': gap
    })

# A3C 추가
generalization_gap.append({
    'Model': 'A3C Global',
    'Gap': a3c_seen - a3c_unseen
})

gap_df = pd.DataFrame(generalization_gap)
gap_df.plot(x='Model', y='Gap', kind='bar', ax=ax4, legend=False, color='#e74c3c')
ax4.set_title('일반화 갭 (Seen - Unseen)', fontsize=14, fontweight='bold')
ax4.set_ylabel('Performance Gap', fontsize=12)
ax4.set_xlabel('')
ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax4.grid(axis='y', alpha=0.3)
ax4.set_xticklabels(gap_df['Model'], rotation=45, ha='right')

plt.tight_layout()
output_path = f'generalization_test_{TIMESTAMP}.png'
plt.savefig(output_path, dpi=180, bbox_inches='tight')
print(f"[시각화 저장] {output_path}")
print()

# CSV 저장
csv_output = f'generalization_results_{TIMESTAMP}.csv'
df.to_csv(csv_output, index=False)
print(f"[결과 저장] {csv_output}")
print()

print("="*80)
print("일반화 성능 테스트 완료!")
print("="*80)
