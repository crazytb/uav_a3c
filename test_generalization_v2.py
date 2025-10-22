"""
일반화 성능 테스트 v2: max_comp_units + agent_velocities
- A3C Global vs Individual Workers
- 2D 환경 공간에서 일반화 성능 테스트
- 지원: runs 폴더 또는 archived_experiments 폴더에서 모델 선택
"""

import argparse
import os
import glob

# ========================================
# Command-line argument parsing (먼저 처리)
# ========================================
parser = argparse.ArgumentParser(description='Generalization Performance Test v2')
parser.add_argument('--source', type=str, choices=['runs', 'archived'], default='runs',
                    help='Source folder: "runs" or "archived" (archived_experiments)')
parser.add_argument('--timestamp', type=str, default=None,
                    help='Specific timestamp to evaluate (e.g., 20251021_153805)')
parser.add_argument('--list', action='store_true',
                    help='List available experiments and exit')
args = parser.parse_args()

# ========================================
# Experiment selection logic
# ========================================
def list_available_experiments():
    """사용 가능한 실험 목록 출력"""
    print("=" * 80)
    print("Available Experiments")
    print("=" * 80)
    print()

    # Check runs folder
    print("[1] runs/ folder:")
    runs_a3c = sorted(glob.glob('runs/a3c_*'))
    runs_timestamps = [os.path.basename(p).replace('a3c_', '') for p in runs_a3c]
    if runs_timestamps:
        for ts in runs_timestamps:
            print(f"  - {ts}")
    else:
        print("  (no experiments found)")
    print()

    # Check archived_experiments folder
    print("[2] archived_experiments/ folder:")
    archived_dirs = sorted(glob.glob('archived_experiments/*/'))
    archived_timestamps = [os.path.basename(os.path.dirname(p)) for p in archived_dirs if os.path.isdir(p)]
    if archived_timestamps:
        for ts in archived_timestamps:
            # Check if models exist
            a3c_path = f"archived_experiments/{ts}/a3c_{ts}/models/global_final.pth"
            if os.path.exists(a3c_path):
                print(f"  - {ts}")
    else:
        print("  (no experiments found)")
    print()
    print("=" * 80)
    print("Usage:")
    print("  python test_generalization_v2.py --source runs --timestamp <timestamp>")
    print("  python test_generalization_v2.py --source archived --timestamp <timestamp>")
    print("=" * 80)

if args.list:
    list_available_experiments()
    exit(0)

# ========================================
# 필요한 모듈 임포트 (--list 이후)
# ========================================
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from drl_framework.networks import RecurrentActorCritic
from drl_framework.custom_env import make_env
from drl_framework.utils import flatten_dict_values
from drl_framework.params import *

# Determine base path and timestamp
if args.source == 'runs':
    base_folder = 'runs'
    if args.timestamp is None:
        # Auto-detect latest timestamp
        runs_a3c = sorted(glob.glob('runs/a3c_*'))
        if not runs_a3c:
            print("[ERROR] No experiments found in runs/ folder")
            print("Use --list to see available experiments")
            exit(1)
        TIMESTAMP = os.path.basename(runs_a3c[-1]).replace('a3c_', '')
        print(f"[Auto-detected] Using latest timestamp: {TIMESTAMP}")
    else:
        TIMESTAMP = args.timestamp

    A3C_MODEL_PATH = f"runs/a3c_{TIMESTAMP}/models/global_final.pth"
    IND_MODEL_PATH_TEMPLATE = f"runs/individual_{TIMESTAMP}/models/individual_worker_{{worker_id}}_final.pth"

else:  # archived
    base_folder = 'archived_experiments'
    if args.timestamp is None:
        # Auto-detect latest timestamp
        archived_dirs = sorted(glob.glob('archived_experiments/*/'))
        valid_timestamps = []
        for d in archived_dirs:
            ts = os.path.basename(os.path.dirname(d))
            a3c_path = f"archived_experiments/{ts}/a3c_{ts}/models/global_final.pth"
            if os.path.exists(a3c_path):
                valid_timestamps.append(ts)

        if not valid_timestamps:
            print("[ERROR] No valid experiments found in archived_experiments/ folder")
            print("Use --list to see available experiments")
            exit(1)

        TIMESTAMP = valid_timestamps[-1]
        print(f"[Auto-detected] Using latest archived timestamp: {TIMESTAMP}")
    else:
        TIMESTAMP = args.timestamp

    A3C_MODEL_PATH = f"archived_experiments/{TIMESTAMP}/a3c_{TIMESTAMP}/models/global_final.pth"
    IND_MODEL_PATH_TEMPLATE = f"archived_experiments/{TIMESTAMP}/individual_{TIMESTAMP}/models/individual_worker_{{worker_id}}_final.pth"

# Validate model paths exist
if not os.path.exists(A3C_MODEL_PATH):
    print(f"[ERROR] A3C model not found: {A3C_MODEL_PATH}")
    print("Use --list to see available experiments")
    exit(1)

print()
print("=" * 80)
print(f"Experiment Configuration")
print("=" * 80)
print(f"  Source    : {args.source}")
print(f"  Timestamp : {TIMESTAMP}")
print(f"  A3C Model : {A3C_MODEL_PATH}")
print("=" * 80)
print()

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
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model

def evaluate_model(model, env_params, n_episodes=30):
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
print("=== 일반화 성능 테스트 v2: max_comp_units + agent_velocities ===")
print("="*80)
print()

# 1. 학습 시 사용된 환경 (Seen environments)
# 학습: velocity 5-25 km/h, comp_units 200
# Worker별로 5 + (25-5) * i / (n_workers-1) 공식으로 할당
SEEN_COMP_UNITS = [200]
SEEN_VELOCITIES = [5, 10, 15, 20, 25]  # 실제 학습 velocity

# 학습 시 각 Worker가 경험한 환경 조합 (comp_units는 고정 200)
SEEN_ENVS = [
    (200, 5),      # Worker 0
    (200, 10),     # Worker 1
    (200, 15),     # Worker 2
    (200, 20),     # Worker 3
    (200, 25),     # Worker 4
]

# 2. 테스트 환경 설정
# Seen 환경: 학습한 조합
# intra: 학습 범위 내의 새로운 조합
# extra: 학습 범위 밖의 조합

TEST_SCENARIOS = []

# (1) Seen 환경 (학습한 정확한 조합)
for comp, vel in SEEN_ENVS:
    TEST_SCENARIOS.append({
        'comp_units': comp,
        'velocity': vel,
        'type': 'Seen'
    })

# (2) Intra-Generalization: Velocity만 변경 (comp_units는 seen=200 유지)
intra_configs = [
    (200, 7),   # 사이값
    (200, 12),
    (200, 17),
    (200, 25),  # 범위 약간 밖
    (200, 30),
    (200, 35),
]
for comp, vel in intra_configs:
    TEST_SCENARIOS.append({
        'comp_units': comp,
        'velocity': vel,
        'type': 'Intra'
    })

# (3) Extra-Generalization: 둘 다 변경
extra_configs = [
    (100, 25),   # comp_units + velocity 둘 다 변경
    (100, 30),
    (150, 25),
    (150, 30),
    (150, 35),
    (100, 10),   # comp_units만 변경 (velocity는 seen 범위)
    (150, 15),
]
for comp, vel in extra_configs:
    TEST_SCENARIOS.append({
        'comp_units': comp,
        'velocity': vel,
        'type': 'Extra'
    })

print(f"총 테스트 시나리오: {len(TEST_SCENARIOS)}개")
print(f"  - Seen : {sum(1 for s in TEST_SCENARIOS if s['type'] == 'Seen')}")
print(f"  - Intra: {sum(1 for s in TEST_SCENARIOS if s['type'] == 'Intra')}")
print(f"  - Extra: {sum(1 for s in TEST_SCENARIOS if s['type'] == 'Extra')}")
print()

# ========================================
# 모델 평가
# ========================================

results = []

# A3C Global 모델
print(f"[1/6] A3C Global 모델 로드 중...")
a3c_model = load_model(A3C_MODEL_PATH)

for idx, scenario in enumerate(TEST_SCENARIOS):
    comp = scenario['comp_units']
    vel = scenario['velocity']
    env_type = scenario['type']

    env_params = ENV_PARAMS.copy()
    env_params['max_comp_units'] = comp
    env_params['agent_velocities'] = vel

    print(f"  [{idx+1}/{len(TEST_SCENARIOS)}] Testing (comp={comp}, vel={vel}, {env_type})...", end=" ")
    mean_reward, std_reward = evaluate_model(a3c_model, env_params)

    results.append({
        'model': 'A3C_Global',
        'worker_id': 'Global',
        'trained_comp': 'All',
        'trained_vel': 'All',
        'test_comp': comp,
        'test_vel': vel,
        'env_type': env_type,
        'mean_reward': mean_reward,
        'std_reward': std_reward
    })
    print(f"Reward: {mean_reward:.2f}")

print()

# Individual Workers (각 워커별)
for worker_id in range(n_workers):
    trained_comp, trained_vel = SEEN_ENVS[worker_id]
    ind_model_path = IND_MODEL_PATH_TEMPLATE.format(worker_id=worker_id)

    print(f"[{2+worker_id}/6] Individual Worker {worker_id} (trained on comp={trained_comp}, vel={trained_vel}) 로드 중...")
    ind_model = load_model(ind_model_path)

    for idx, scenario in enumerate(TEST_SCENARIOS):
        comp = scenario['comp_units']
        vel = scenario['velocity']
        env_type = scenario['type']

        env_params = ENV_PARAMS.copy()
        env_params['max_comp_units'] = comp
        env_params['agent_velocities'] = vel

        print(f"  [{idx+1}/{len(TEST_SCENARIOS)}] Testing (comp={comp}, vel={vel}, {env_type})...", end=" ")
        mean_reward, std_reward = evaluate_model(ind_model, env_params)

        results.append({
            'model': f'Individual_W{worker_id}',
            'worker_id': worker_id,
            'trained_comp': trained_comp,
            'trained_vel': trained_vel,
            'test_comp': comp,
            'test_vel': vel,
            'env_type': env_type,
            'mean_reward': mean_reward,
            'std_reward': std_reward
        })
        print(f"Reward: {mean_reward:.2f}")

    print()

# ========================================
# 결과 분석
# ========================================

df = pd.DataFrame(results)

print("="*80)
print("=== 일반화 성능 분석 ===")
print("="*80)
print()

# 1. 환경 타입별 성능 비교
print("[1] 환경 타입별 평균 성능")
print("-"*80)

for env_type in ['Seen', 'Intra', 'Extra']:
    a3c_perf = df[(df['model'] == 'A3C_Global') & (df['env_type'] == env_type)]['mean_reward'].mean()
    ind_perf = df[(df['model'] != 'A3C_Global') & (df['env_type'] == env_type)]['mean_reward'].mean()

    print(f"{env_type:15s}: A3C={a3c_perf:.3f}, Individual={ind_perf:.3f}, Δ={a3c_perf-ind_perf:+.3f}")

print()

# 2. Worker별 일반화 성능
print("[2] Individual Worker별 일반화 성능")
print("-"*80)

for worker_id in range(n_workers):
    trained_comp, trained_vel = SEEN_ENVS[worker_id]
    worker_data = df[df['worker_id'] == worker_id]

    # 학습 환경 성능
    seen_mask = (worker_data['test_comp'] == trained_comp) & (worker_data['test_vel'] == trained_vel)
    seen_perf = worker_data[seen_mask]['mean_reward'].values[0] if seen_mask.any() else 0

    # 다른 환경 성능
    interp_perf = worker_data[worker_data['env_type'] == 'Intra']['mean_reward'].mean()
    extrap_perf = worker_data[worker_data['env_type'] == 'Extra']['mean_reward'].mean()

    print(f"Worker {worker_id} (trained on comp={trained_comp}, vel={trained_vel}):")
    print(f"  Seen         : {seen_perf:.3f}")
    print(f"  Intra: {interp_perf:.3f} ({(interp_perf-seen_perf):+.3f})")
    print(f"  Extra: {extrap_perf:.3f} ({(extrap_perf-seen_perf):+.3f})")

print()

# 3. 최악/최고 시나리오
print("[3] 최악/최고 성능 시나리오")
print("-"*80)

for model_type in ['A3C_Global', 'Individual']:
    if model_type == 'A3C_Global':
        model_data = df[df['model'] == 'A3C_Global']
    else:
        model_data = df[df['model'] != 'A3C_Global']

    worst = model_data.loc[model_data['mean_reward'].idxmin()]
    best = model_data.loc[model_data['mean_reward'].idxmax()]

    print(f"{model_type}:")
    print(f"  최악: comp={worst['test_comp']}, vel={worst['test_vel']}, reward={worst['mean_reward']:.3f}, type={worst['env_type']}")
    print(f"  최고: comp={best['test_comp']}, vel={best['test_vel']}, reward={best['mean_reward']:.3f}, type={best['env_type']}")

print()

# ========================================
# Visualization
# ========================================

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# (1) Performance comparison by environment type
ax1 = fig.add_subplot(gs[0, :])
type_comparison = df.groupby(['model', 'env_type'])['mean_reward'].mean().unstack()
type_comparison.plot(kind='bar', ax=ax1, color=['#2ecc71', '#f39c12', '#e74c3c'])
ax1.set_title('Performance Comparison by Environment Type', fontsize=14, fontweight='bold')
ax1.set_ylabel('Mean Reward', fontsize=12)
ax1.set_xlabel('')
ax1.legend(title='Environment Type', fontsize=10)
ax1.grid(axis='y', alpha=0.3)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')

# (2) A3C 2D Heatmap (comp_units vs velocity)
ax2 = fig.add_subplot(gs[1, 0])
a3c_pivot = df[df['model'] == 'A3C_Global'].pivot_table(
    values='mean_reward',
    index='test_vel',
    columns='test_comp',
    aggfunc='mean'
)
sns.heatmap(a3c_pivot, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax2, cbar_kws={'label': 'Reward'})
ax2.set_title('A3C Global: comp_units vs velocity', fontsize=12, fontweight='bold')
ax2.set_xlabel('max_comp_units')
ax2.set_ylabel('agent_velocities')

# (3-7) Individual Workers 2D Heatmaps
# Grid layout: row 1 has 2 slots (col 1, 2), row 2 has 3 slots (col 0, 1, 2)
# worker_positions = [
#     (1, 1),  # Worker 0
#     (1, 2),  # Worker 1
#     (2, 0),  # Worker 2
#     (2, 1),  # Worker 3
#     (2, 2),  # Worker 4
# ]

# for worker_id in range(n_workers):
#     row, col = worker_positions[worker_id]
#     ax = fig.add_subplot(gs[row, col])

#     worker_data = df[df['worker_id'] == worker_id]
#     worker_pivot = worker_data.pivot_table(
#         values='mean_reward',
#         index='test_vel',
#         columns='test_comp',
#         aggfunc='mean'
#     )

#     trained_comp, trained_vel = SEEN_ENVS[worker_id]

#     sns.heatmap(worker_pivot, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax,
#                 cbar_kws={'label': 'Reward'}, vmin=a3c_pivot.min().min(), vmax=a3c_pivot.max().max())
#     ax.set_title(f'Worker {worker_id} (trained@{trained_comp},{trained_vel})', fontsize=10, fontweight='bold')
#     ax.set_xlabel('max_comp_units', fontsize=9)
#     ax.set_ylabel('agent_velocities', fontsize=9)

output_path = f'generalization_test_v2_{args.source}_{TIMESTAMP}.png'
plt.savefig(output_path, dpi=180, bbox_inches='tight')
print(f"[Visualization saved] {output_path}")
print()

# Save CSV results
csv_output = f'generalization_results_v2_{args.source}_{TIMESTAMP}.csv'
df.to_csv(csv_output, index=False)
print(f"[Results saved] {csv_output}")
print()

print("="*80)
print("Generalization Performance Test v2 Complete!")
print("="*80)
