#!/usr/bin/env python3
"""
Test Generalization Performance for Ablation Studies
Runs velocity sweep tests for all completed ablations
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from drl_framework import params
from drl_framework.custom_env import CustomEnv
from drl_framework.networks import ActorCritic, RecurrentActorCritic
from drl_framework.trainer import make_env, flatten_dict_values
from ablation_configs import get_config


def load_model(model_path, ablation_name, device):
    """Load trained model with proper architecture for ablation"""

    # Load checkpoint first to inspect structure
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Get state dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Detect architecture from checkpoint keys
    has_rnn = any('rnn' in key for key in state_dict.keys())
    has_layer_norm = any('ln_' in key for key in state_dict.keys())

    # Detect hidden_dim from checkpoint (more reliable than config)
    # Check feature layer size to determine actual hidden_dim used during training
    if 'feature.weight' in state_dict:
        hidden_dim = state_dict['feature.weight'].shape[0]  # Output dim of feature layer
    else:
        # Fallback to config if can't detect from checkpoint
        config = get_config(ablation_name)
        hidden_dim = config.get('hidden_dim', 128)

    # Fixed dimensions for this environment (from trainer.py)
    input_dim = 48  # state is flattened dict with 48 features
    output_dim = 3  # 3 actions: local, cloud, drop

    # IMPORTANT: trainer.py always creates RecurrentActorCritic
    # The actual architecture is determined by use_layer_norm parameter
    # So we always use RecurrentActorCritic, but with correct use_layer_norm
    model = RecurrentActorCritic(
        state_dim=input_dim,
        action_dim=output_dim,
        hidden_dim=hidden_dim,
        use_layer_norm=has_layer_norm  # Use detected value from checkpoint
    ).to(device)

    # Load state dict
    model.load_state_dict(state_dict)
    model.eval()

    return model, has_rnn


def test_velocity(model, velocity, n_episodes=100, device='cpu', use_recurrent=True):
    """Test model performance at specific velocity"""

    # Create environment with specified velocity
    env_params = params.ENV_PARAMS.copy()
    env_params['agent_velocities'] = velocity
    env_params['reward_params'] = params.REWARD_PARAMS  # IMPORTANT: Add reward params for proper scaling
    env_fn = make_env(**env_params)
    env = env_fn()

    episode_rewards = []

    for _ in range(n_episodes):
        state_dict, _ = env.reset()
        state = flatten_dict_values(state_dict)  # Flatten dict to vector
        done = False
        episode_reward = 0

        if use_recurrent:
            hidden = None

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

            with torch.no_grad():
                if use_recurrent:
                    # Use step() method for single-step inference
                    policy_logits, value, hidden = model.step(state_tensor, hidden)
                else:
                    # Feedforward model (if we ever have one)
                    policy_logits, value = model(state_tensor)

                # Greedy action selection
                action = torch.argmax(policy_logits, dim=1).item()

            state_dict, reward, done, _, _ = env.step(action)
            state = flatten_dict_values(state_dict)  # Flatten dict to vector
            episode_reward += reward

        episode_rewards.append(episode_reward)

    return {
        'velocity': velocity,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards)
    }


def test_ablation_generalization(ablation_dir, ablation_name, velocities, n_episodes=100):
    """Test generalization for one ablation across all seeds"""

    print(f"\n{'='*80}")
    print(f"Testing: {ablation_name}")
    print(f"{'='*80}\n")

    config = get_config(ablation_name)
    use_recurrent = config.get('use_recurrent', True)
    device = torch.device('cpu')

    ablation_path = Path(ablation_dir) / ablation_name
    if not ablation_path.exists():
        print(f"Warning: {ablation_path} does not exist")
        return None

    results = []

    # Test each seed
    for seed_dir in sorted(ablation_path.glob("seed_*")):
        seed = seed_dir.name.split('_')[1]
        print(f"Seed {seed}:")

        # Test A3C model
        a3c_model_path = seed_dir / "a3c" / "models" / "global_final.pth"
        if a3c_model_path.exists():
            print("  Testing A3C model...")
            model, use_recurrent = load_model(a3c_model_path, ablation_name, device)

            for velocity in tqdm(velocities, desc="  Velocities"):
                result = test_velocity(model, velocity, n_episodes, device, use_recurrent)
                result['seed'] = int(seed)
                result['model'] = 'a3c'
                results.append(result)
        else:
            print(f"  Warning: A3C model not found: {a3c_model_path}")

        # Test Individual models
        individual_dir = seed_dir / "individual" / "models"
        if individual_dir.exists():
            print("  Testing Individual models...")
            worker_models = sorted(individual_dir.glob("individual_worker_*_final.pth"))

            for worker_id, worker_model_path in enumerate(worker_models):
                model, use_recurrent_ind = load_model(worker_model_path, ablation_name, device)

                for velocity in velocities:
                    result = test_velocity(model, velocity, n_episodes, device, use_recurrent_ind)
                    result['seed'] = int(seed)
                    result['model'] = 'individual'
                    result['worker'] = worker_id
                    results.append(result)
        else:
            print(f"  Warning: Individual models not found: {individual_dir}")

        print()

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description='Test ablation generalization performance')
    parser.add_argument('--ablation-dir', type=str,
                        default='ablation_results/high_priority',
                        help='Directory containing ablation results')
    parser.add_argument('--output-dir', type=str,
                        default='ablation_results/analysis',
                        help='Output directory for results')
    parser.add_argument('--velocities', type=int, nargs='+',
                        default=[5, 10, 20, 30, 50, 70, 80, 90, 100],
                        help='Velocities to test (km/h)')
    parser.add_argument('--n-episodes', type=int, default=100,
                        help='Episodes per velocity')
    parser.add_argument('--ablations', type=str, nargs='+',
                        default=['ablation_1_no_rnn', 'ablation_2_no_layer_norm',
                                'ablation_15_few_workers', 'ablation_16_many_workers'],
                        help='Ablations to test')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("="*80)
    print("Ablation Generalization Testing")
    print("="*80)
    print(f"Velocities: {args.velocities}")
    print(f"Episodes per velocity: {args.n_episodes}")
    print(f"Ablations: {args.ablations}")
    print("="*80)

    # Test each ablation
    all_results = {}
    for ablation in args.ablations:
        results_df = test_ablation_generalization(
            args.ablation_dir,
            ablation,
            args.velocities,
            args.n_episodes
        )

        if results_df is not None:
            all_results[ablation] = results_df

            # Save individual ablation results
            output_file = Path(args.output_dir) / f"{ablation}_generalization.csv"
            results_df.to_csv(output_file, index=False)
            print(f"Saved: {output_file}\n")

    # Compute summary statistics
    print("="*80)
    print("Generalization Performance Summary")
    print("="*80)
    print()

    summary_rows = []

    for ablation, results_df in all_results.items():
        # A3C generalization
        a3c_results = results_df[results_df['model'] == 'a3c']
        if len(a3c_results) > 0:
            a3c_mean = a3c_results['mean_reward'].mean()
            a3c_std = a3c_results['mean_reward'].std()
            a3c_cv = a3c_std / a3c_mean if a3c_mean > 0 else 0
            a3c_worst = a3c_results['mean_reward'].min()

            # Individual generalization
            individual_results = results_df[results_df['model'] == 'individual']
            if len(individual_results) > 0:
                ind_mean = individual_results['mean_reward'].mean()
                ind_std = individual_results['mean_reward'].std()
                ind_cv = ind_std / ind_mean if ind_mean > 0 else 0
                ind_worst = individual_results['mean_reward'].min()

                gap = a3c_mean - ind_mean
                gap_pct = (gap / ind_mean * 100) if ind_mean > 0 else 0

                summary_rows.append({
                    'Ablation': ablation,
                    'A3C_Mean': a3c_mean,
                    'A3C_Std': a3c_std,
                    'A3C_CV': a3c_cv,
                    'A3C_Worst': a3c_worst,
                    'Individual_Mean': ind_mean,
                    'Individual_Std': ind_std,
                    'Individual_CV': ind_cv,
                    'Individual_Worst': ind_worst,
                    'Gap': gap,
                    'Gap_Pct': gap_pct
                })

                print(f"{ablation}")
                print(f"  A3C: {a3c_mean:.2f} ± {a3c_std:.2f} (CV: {a3c_cv:.3f}, Worst: {a3c_worst:.2f})")
                print(f"  Individual: {ind_mean:.2f} ± {ind_std:.2f} (CV: {ind_cv:.3f}, Worst: {ind_worst:.2f})")
                print(f"  Gap: {gap:+.2f} ({gap_pct:+.1f}%)")
                print()

    # Save summary
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_file = Path(args.output_dir) / "generalization_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"Saved summary: {summary_file}")

    print("="*80)
    print("Generalization testing complete!")
    print("="*80)


if __name__ == "__main__":
    main()
