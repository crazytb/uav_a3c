#!/usr/bin/env python3
"""
Single Ablation Execution Script
Runs one ablation experiment with specified configuration and seed
"""

import os
import sys
import argparse
import shutil
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ablation_configs import get_config
from drl_framework import params


def apply_ablation_config(ablation_name):
    """Apply ablation configuration to params module"""
    config = get_config(ablation_name)

    print(f"\n{'='*80}")
    print(f"Applying configuration: {ablation_name}")
    print(f"Description: {config['description']}")
    print(f"{'='*80}\n")

    # Apply network architecture settings
    if 'use_recurrent' in config:
        params.use_recurrent = config['use_recurrent']
        print(f"  use_recurrent: {config['use_recurrent']}")

    if 'use_layer_norm' in config:
        params.use_layer_norm = config['use_layer_norm']
        print(f"  use_layer_norm: {config['use_layer_norm']}")

    if 'hidden_dim' in config:
        params.hidden_dim = config['hidden_dim']
        print(f"  hidden_dim: {config['hidden_dim']}")

    # Apply hyperparameters
    if 'entropy_coef' in config:
        params.entropy_coef = config['entropy_coef']
        print(f"  entropy_coef: {config['entropy_coef']}")

    if 'value_loss_coef' in config:
        params.value_loss_coef = config['value_loss_coef']
        print(f"  value_loss_coef: {config['value_loss_coef']}")

    if 'lr' in config:
        params.lr = config['lr']
        print(f"  lr: {config['lr']}")

    if 'max_grad_norm' in config:
        params.max_grad_norm = config['max_grad_norm']
        print(f"  max_grad_norm: {config['max_grad_norm']}")

    # Apply environment settings
    if 'n_workers' in config:
        params.n_workers = config['n_workers']
        print(f"  n_workers: {config['n_workers']}")

    if 'target_episode_count' in config:
        params.target_episode_count = config['target_episode_count']
        print(f"  target_episode_count: {config['target_episode_count']}")

    if 'max_comp_units_for_cloud' in config:
        params.ENV_PARAMS['max_comp_units_for_cloud'] = config['max_comp_units_for_cloud']
        print(f"  max_comp_units_for_cloud: {config['max_comp_units_for_cloud']}")

    if 'agent_velocities' in config:
        params.ENV_PARAMS['agent_velocities'] = config['agent_velocities']
        print(f"  agent_velocities: {config['agent_velocities']}")

    if 'reward_scale' in config:
        params.REWARD_PARAMS['REWARD_SCALE'] = config['reward_scale']
        print(f"  reward_scale: {config['reward_scale']}")

    print()
    return config


def organize_results(ablation_name, seed, result_base_dir):
    """Organize training results into structured directory"""

    # Find most recent runs directories
    runs_dir = Path("runs")

    # Find most recent a3c and individual directories
    a3c_dirs = sorted(runs_dir.glob("a3c_*"), key=lambda x: x.stat().st_mtime, reverse=True)
    individual_dirs = sorted(runs_dir.glob("individual_*"), key=lambda x: x.stat().st_mtime, reverse=True)

    if not a3c_dirs or not individual_dirs:
        print(f"Warning: Could not find recent training results")
        return False

    latest_a3c = a3c_dirs[0]
    latest_individual = individual_dirs[0]

    # Create destination directory
    dest_dir = result_base_dir / f"seed_{seed}"
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Copy A3C results
    a3c_dest = dest_dir / "a3c"
    if a3c_dest.exists():
        shutil.rmtree(a3c_dest)
    shutil.copytree(latest_a3c, a3c_dest)
    print(f"  Copied A3C results: {latest_a3c} -> {a3c_dest}")

    # Copy Individual results
    individual_dest = dest_dir / "individual"
    if individual_dest.exists():
        shutil.rmtree(individual_dest)
    shutil.copytree(latest_individual, individual_dest)
    print(f"  Copied Individual results: {latest_individual} -> {individual_dest}")

    # Save configuration
    config_file = dest_dir / "config.txt"
    config = get_config(ablation_name)
    with open(config_file, 'w') as f:
        f.write(f"Ablation: {ablation_name}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\nConfiguration:\n")
        for key, value in config.items():
            f.write(f"  {key}: {value}\n")

    print(f"  Saved configuration: {config_file}")

    return True


def main():
    parser = argparse.ArgumentParser(description='Run single ablation experiment')
    parser.add_argument('--ablation', type=str, required=True,
                        help='Ablation name (e.g., ablation_1_no_rnn)')
    parser.add_argument('--seed', type=int, required=True,
                        help='Random seed')
    parser.add_argument('--output-dir', type=str, default='ablation_results/high_priority',
                        help='Base output directory')

    args = parser.parse_args()

    # Set random seed environment variable
    os.environ['RANDOM_SEED'] = str(args.seed)

    # Apply ablation configuration
    config = apply_ablation_config(args.ablation)

    # Prepare output directory
    result_base_dir = Path(args.output_dir) / args.ablation
    result_base_dir.mkdir(parents=True, exist_ok=True)

    print(f"Results will be saved to: {result_base_dir}/seed_{args.seed}\n")

    # Run training
    print(f"{'='*80}")
    print(f"Starting Training: {args.ablation} (seed={args.seed})")
    print(f"{'='*80}\n")

    # Run main training as subprocess
    import subprocess
    result = subprocess.run(
        [sys.executable, "main_train.py"],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        env=os.environ.copy()
    )

    if result.returncode != 0:
        print(f"\n✗ Training failed with exit code: {result.returncode}")
        return 1

    print(f"\n{'='*80}")
    print(f"Training Completed")
    print(f"{'='*80}\n")

    # Organize results
    print(f"Organizing results...")
    success = organize_results(args.ablation, args.seed, result_base_dir)

    if success:
        print(f"\n✓ Ablation experiment completed successfully!")
        print(f"  Results saved to: {result_base_dir}/seed_{args.seed}")
    else:
        print(f"\n✗ Warning: Could not organize results properly")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
