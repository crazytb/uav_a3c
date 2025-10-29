"""
간단한 Baseline Generalization Test
기존 학습된 모델을 다양한 velocity 환경에서 테스트
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from drl_framework.networks import RecurrentActorCritic
from drl_framework.custom_env import CustomEnv
from drl_framework.utils import flatten_dict_values
from drl_framework import params

device = params.device

def test_model(model_path, test_velocities, n_episodes=100):
    """단일 모델을 다양한 velocity에서 테스트"""

    # Load model
    model = RecurrentActorCritic(
        state_dim=48,
        action_dim=3,
        hidden_dim=params.hidden_dim,
        use_layer_norm=params.use_layer_norm
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    results = {}

    for velocity in test_velocities:
        print(f"  Testing velocity={velocity} km/h...", end=' ')

        # Create environment with this velocity
        env_params = params.ENV_PARAMS.copy()
        env_params['agent_velocities'] = velocity

        env = CustomEnv(**env_params, reward_params=params.REWARD_PARAMS)

        episode_rewards = []

        for ep in range(n_episodes):
            obs, _ = env.reset()
            obs_flat = flatten_dict_values(obs)
            obs_tensor = torch.FloatTensor(obs_flat).unsqueeze(0).to(device)

            hx = model.init_hidden(1, device=device)
            episode_reward = 0
            done = False

            while not done:
                with torch.no_grad():
                    logits, value, hx = model.step(obs_tensor, hx)
                    action = torch.argmax(logits, dim=1).item()

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward

                obs_flat = flatten_dict_values(obs)
                obs_tensor = torch.FloatTensor(obs_flat).unsqueeze(0).to(device)

            episode_rewards.append(episode_reward)

        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)

        results[velocity] = {
            'mean': mean_reward,
            'std': std_reward,
            'min': np.min(episode_rewards),
            'max': np.max(episode_rewards)
        }

        print(f"Mean: {mean_reward:.2f} ± {std_reward:.2f}")

    return results

def main():
    baseline_dir = Path("ablation_results/baseline_20251029_165119")
    output_dir = Path("generalization_results_2000ep")
    output_dir.mkdir(exist_ok=True)

    # Test velocities
    test_velocities = [5, 10, 15, 20, 25, 30, 50, 75, 100]

    # Find all seeds
    seed_dirs = sorted(baseline_dir.glob("seed_*"))

    print("="*80)
    print("BASELINE GENERALIZATION TEST - Velocity Sweep")
    print("="*80)
    print(f"Test velocities: {test_velocities}")
    print(f"Episodes per velocity: 100")
    print(f"Seeds: {len(seed_dirs)}")
    print("="*80)
    print()

    all_results = []

    # Test A3C models
    print("[1] Testing A3C Global Models")
    print("-"*80)

    for seed_dir in seed_dirs:
        seed = int(seed_dir.name.split('_')[1])
        a3c_model_path = seed_dir / "a3c" / "models" / "global_final.pth"

        if not a3c_model_path.exists():
            print(f"Seed {seed}: Model not found!")
            continue

        print(f"Seed {seed}:")
        results = test_model(a3c_model_path, test_velocities)

        for velocity, metrics in results.items():
            all_results.append({
                'model_type': 'A3C',
                'seed': seed,
                'velocity': velocity,
                'mean_reward': metrics['mean'],
                'std_reward': metrics['std'],
                'min_reward': metrics['min'],
                'max_reward': metrics['max']
            })

    print()

    # Test Individual models
    print("[2] Testing Individual Worker Models")
    print("-"*80)

    for seed_dir in seed_dirs:
        seed = int(seed_dir.name.split('_')[1])
        print(f"Seed {seed}:")

        # Test each worker
        for worker_id in range(5):
            ind_model_path = seed_dir / "individual" / "models" / f"individual_worker_{worker_id}_final.pth"

            if not ind_model_path.exists():
                continue

            print(f"  Worker {worker_id}:")
            results = test_model(ind_model_path, test_velocities)

            for velocity, metrics in results.items():
                all_results.append({
                    'model_type': f'Individual_W{worker_id}',
                    'seed': seed,
                    'velocity': velocity,
                    'mean_reward': metrics['mean'],
                    'std_reward': metrics['std'],
                    'min_reward': metrics['min'],
                    'max_reward': metrics['max']
                })

        print()

    # Save results
    df = pd.DataFrame(all_results)
    csv_path = output_dir / "baseline_2000ep_velocity_generalization.csv"
    df.to_csv(csv_path, index=False)
    print(f"[SAVED] Results: {csv_path}")

    # Generate summary
    print()
    print("="*80)
    print("SUMMARY")
    print("="*80)

    # A3C summary
    df_a3c = df[df['model_type'] == 'A3C']
    print("\n[A3C Global]")
    print(f"{'Velocity':<12} {'Mean Reward':>12} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("-"*80)

    for vel in test_velocities:
        df_vel = df_a3c[df_a3c['velocity'] == vel]
        mean_across_seeds = df_vel['mean_reward'].mean()
        std_across_seeds = df_vel['mean_reward'].std()
        min_val = df_vel['mean_reward'].min()
        max_val = df_vel['mean_reward'].max()

        print(f"{vel:>4} km/h    {mean_across_seeds:>12.2f} {std_across_seeds:>8.2f} {min_val:>8.2f} {max_val:>8.2f}")

    # Individual summary (averaged across all workers)
    df_ind = df[df['model_type'].str.startswith('Individual')]
    print("\n[Individual Workers - Averaged]")
    print(f"{'Velocity':<12} {'Mean Reward':>12} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("-"*80)

    for vel in test_velocities:
        df_vel = df_ind[df_ind['velocity'] == vel]
        mean_across_all = df_vel['mean_reward'].mean()
        std_across_all = df_vel['mean_reward'].std()
        min_val = df_vel['mean_reward'].min()
        max_val = df_vel['mean_reward'].max()

        print(f"{vel:>4} km/h    {mean_across_all:>12.2f} {std_across_all:>8.2f} {min_val:>8.2f} {max_val:>8.2f}")

    # Overall comparison
    print("\n[Overall Metrics]")
    print("-"*80)
    a3c_all = df_a3c['mean_reward'].values
    ind_all = df_ind['mean_reward'].values

    print(f"A3C Generalization Score  : {np.mean(a3c_all):.2f} ± {np.std(a3c_all):.2f}")
    print(f"Ind Generalization Score  : {np.mean(ind_all):.2f} ± {np.std(ind_all):.2f}")
    print(f"A3C Advantage             : {np.mean(a3c_all) - np.mean(ind_all):+.2f}")
    print(f"A3C Robustness (CV)       : {np.std(a3c_all)/np.mean(a3c_all):.3f}")
    print(f"Ind Robustness (CV)       : {np.std(ind_all)/np.mean(ind_all):.3f}")
    print(f"A3C Worst-Case            : {np.min(a3c_all):.2f}")
    print(f"Ind Worst-Case            : {np.min(ind_all):.2f}")

if __name__ == "__main__":
    main()
