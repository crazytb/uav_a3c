#!/usr/bin/env python3
"""
Test Generalization Performance for Neither RNN nor LN
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from drl_framework import params
from drl_framework.custom_env import CustomEnv
from drl_framework.networks import ActorCritic
from drl_framework.trainer import make_env, flatten_dict_values


def load_model(model_path, device):
    """Load feedforward ActorCritic model"""

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Get state dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Detect hidden_dim from checkpoint
    if 'feature.weight' in state_dict:
        hidden_dim = state_dict['feature.weight'].shape[0]
    else:
        hidden_dim = 128  # Default

    # Create feedforward ActorCritic
    model = ActorCritic(
        state_dim=48,
        action_dim=3,
        hidden_dim=hidden_dim
    ).to(device)

    model.load_state_dict(state_dict)
    model.eval()

    return model


def test_velocity(model, velocity, n_episodes=100, device='cpu'):
    """Test model performance at specific velocity"""

    # Create environment with specified velocity
    env_params = params.ENV_PARAMS.copy()
    env_params['agent_velocities'] = velocity
    env_params['reward_params'] = params.REWARD_PARAMS
    env_fn = make_env(**env_params)
    env = env_fn()

    episode_rewards = []

    for _ in range(n_episodes):
        state_dict, _ = env.reset()
        state = flatten_dict_values(state_dict)
        done = False
        episode_reward = 0

        # Feedforward model - no hidden state
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

            with torch.no_grad():
                policy_logits, value = model(state_tensor)
                # Greedy action selection
                action = torch.argmax(policy_logits, dim=1).item()

            state_dict, reward, done, _, _ = env.step(action)
            state = flatten_dict_values(state_dict)
            episode_reward += reward

        episode_rewards.append(episode_reward)

    return {
        'velocity': velocity,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards)
    }


def test_neither_generalization(base_dir, velocities, n_episodes=100):
    """Test generalization for neither configuration"""

    device = torch.device('cpu')
    results = []

    base_path = Path(base_dir)

    # Get all seed directories
    seed_dirs = sorted(base_path.glob("seed_*"))

    if not seed_dirs:
        print(f"ERROR: No seed directories found in {base_dir}")
        return None

    print(f"\nFound {len(seed_dirs)} seeds")
    print(f"Testing velocities: {velocities}")
    print(f"Episodes per velocity: {n_episodes}\n")

    # Test each seed
    for seed_dir in seed_dirs:
        seed = seed_dir.name.split('_')[1]
        print(f"{'='*80}")
        print(f"Seed {seed}")
        print(f"{'='*80}")

        # Test A3C model
        a3c_model_path = seed_dir / "a3c" / "models" / "global_final.pth"
        if a3c_model_path.exists():
            print("Testing A3C model...")
            model = load_model(a3c_model_path, device)

            for velocity in tqdm(velocities, desc="  Velocities"):
                result = test_velocity(model, velocity, n_episodes, device)
                result['seed'] = int(seed)
                result['method'] = 'A3C'
                results.append(result)
        else:
            print(f"WARNING: A3C model not found: {a3c_model_path}")

        # Test Individual models
        individual_dir = seed_dir / "individual" / "models"
        if individual_dir.exists():
            print("Testing Individual models...")

            # Find all individual worker models
            worker_models = sorted(individual_dir.glob("individual_worker_*_final.pth"))

            for worker_path in worker_models:
                worker_id = worker_path.stem.split('_')[-2]  # Extract worker number
                model = load_model(worker_path, device)

                for velocity in tqdm(velocities, desc=f"  Worker {worker_id}"):
                    result = test_velocity(model, velocity, n_episodes, device)
                    result['seed'] = int(seed)
                    result['method'] = 'Individual'
                    result['worker'] = int(worker_id)
                    results.append(result)
        else:
            print(f"WARNING: Individual models not found: {individual_dir}")

        print()

    # Create DataFrame
    df = pd.DataFrame(results)

    # Calculate summary statistics
    print("="*80)
    print("GENERALIZATION PERFORMANCE SUMMARY")
    print("="*80)
    print()

    # Overall performance (average across all velocities)
    summary = df.groupby('method').agg({
        'mean_reward': ['mean', 'std'],
        'std_reward': 'mean'
    }).round(2)

    print("Overall Performance (all velocities):")
    print(summary)
    print()

    # Calculate gap
    a3c_mean = df[df['method'] == 'A3C']['mean_reward'].mean()
    ind_mean = df[df['method'] == 'Individual']['mean_reward'].mean()
    gap = ((a3c_mean - ind_mean) / ind_mean) * 100

    print(f"A3C Mean: {a3c_mean:.2f}")
    print(f"Individual Mean: {ind_mean:.2f}")
    print(f"Gap: {gap:.1f}%")
    print()

    # Per-velocity performance
    print("Per-Velocity Performance:")
    pivot = df.pivot_table(
        values='mean_reward',
        index='velocity',
        columns='method',
        aggfunc='mean'
    ).round(2)
    print(pivot)
    print()

    # Coefficient of Variation (CV)
    a3c_cv = df[df['method'] == 'A3C']['mean_reward'].std() / df[df['method'] == 'A3C']['mean_reward'].mean()
    ind_cv = df[df['method'] == 'Individual']['mean_reward'].std() / df[df['method'] == 'Individual']['mean_reward'].mean()

    print(f"Coefficient of Variation:")
    print(f"  A3C CV: {a3c_cv:.3f}")
    print(f"  Individual CV: {ind_cv:.3f}")
    print()

    # Worst-case performance
    a3c_worst = df[df['method'] == 'A3C']['mean_reward'].min()
    ind_worst = df[df['method'] == 'Individual']['mean_reward'].min()

    print(f"Worst-case Performance:")
    print(f"  A3C: {a3c_worst:.2f}")
    print(f"  Individual: {ind_worst:.2f}")
    print()

    # Save results
    output_path = base_path / "generalization_results.csv"
    df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")
    print("="*80)

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test Neither RNN nor LN generalization')
    parser.add_argument('--base-dir', type=str, required=True,
                        help='Base directory with seed_* subdirectories')
    parser.add_argument('--velocities', type=int, nargs='+',
                        default=[5, 10, 20, 30, 50, 70, 80, 90, 100],
                        help='Velocities to test (km/h)')
    parser.add_argument('--n-episodes', type=int, default=100,
                        help='Episodes per velocity')

    args = parser.parse_args()

    test_neither_generalization(
        args.base_dir,
        args.velocities,
        args.n_episodes
    )
