"""
Ablation Study Runner
체계적으로 ablation study 실험을 수행하고 결과를 수집
"""

import os
import sys
import subprocess
import datetime
import pandas as pd
import numpy as np
import json
import shutil
from pathlib import Path
import argparse

# Import configurations
from ablation_configs import (
    BASELINE_CONFIG,
    ALL_ABLATIONS,
    get_config,
    get_ablations_by_phase,
    get_ablations_by_priority
)

class AblationStudyRunner:
    def __init__(self, output_dir='ablation_results', n_seeds=3, conda_env='torch-cert'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.n_seeds = n_seeds
        self.conda_env = conda_env
        self.seeds = [42, 123, 456, 789, 1024][:n_seeds]

        # Timestamp for this ablation study run
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.study_dir = self.output_dir / f"study_{self.timestamp}"
        self.study_dir.mkdir(exist_ok=True)

        print(f"[INFO] Ablation study results will be saved to: {self.study_dir}")

    def apply_config_to_params(self, config):
        """
        config를 params.py 파일에 적용
        원본 params.py를 백업하고 수정된 버전을 생성
        """
        params_path = Path('drl_framework/params.py')
        backup_path = Path('drl_framework/params.py.backup')

        # Backup original params.py
        if not backup_path.exists():
            shutil.copy(params_path, backup_path)

        # Read original params.py
        with open(backup_path, 'r') as f:
            lines = f.readlines()

        # Modify parameters
        modified_lines = []
        for line in lines:
            modified_line = line

            # Network architecture
            if 'use_layer_norm =' in line and 'use_layer_norm' in config:
                modified_line = f"use_layer_norm = {config['use_layer_norm']}         # Modified by ablation study\n"
            elif 'hidden_dim =' in line and 'hidden_dim' in config:
                modified_line = f"hidden_dim = {config['hidden_dim']}               # Modified by ablation study\n"

            # Hyperparameters
            elif line.strip().startswith('entropy_coef =') and 'entropy_coef' in config:
                modified_line = f"entropy_coef = {config['entropy_coef']}             # Modified by ablation study\n"
            elif line.strip().startswith('value_loss_coef =') and 'value_loss_coef' in config:
                modified_line = f"value_loss_coef = {config['value_loss_coef']}          # Modified by ablation study\n"
            elif line.strip().startswith('lr =') and 'lr' in config:
                modified_line = f"lr = {config['lr']}                       # Modified by ablation study\n"
            elif 'max_grad_norm =' in line and 'max_grad_norm' in config:
                modified_line = f"max_grad_norm = {config['max_grad_norm']}             # Modified by ablation study\n"

            # Worker count
            elif line.strip().startswith('n_workers =') and 'n_workers' in config:
                modified_line = f"n_workers = {config['n_workers']}                  # Modified by ablation study\n"
            elif 'target_episode_count =' in line and 'target_episode_count' in config:
                modified_line = f"target_episode_count = {config['target_episode_count']}      # Modified by ablation study\n"

            # Environment parameters (inside ENV_PARAMS dict)
            elif "'max_comp_units_for_cloud':" in line and 'max_comp_units_for_cloud' in config:
                modified_line = f"    'max_comp_units_for_cloud': {config['max_comp_units_for_cloud']},  # Modified by ablation study\n"
            elif "'agent_velocities':" in line and 'agent_velocities' in config:
                modified_line = f"    'agent_velocities': {config['agent_velocities']}  # Modified by ablation study\n"

            # Reward parameters
            elif "'REWARD_SCALE':" in line and 'reward_scale' in config:
                modified_line = f"    'REWARD_SCALE': {config['reward_scale']},      # Modified by ablation study\n"

            modified_lines.append(modified_line)

        # Write modified params.py
        with open(params_path, 'w') as f:
            f.writelines(modified_lines)

    def restore_params(self):
        """원본 params.py 복원"""
        params_path = Path('drl_framework/params.py')
        backup_path = Path('drl_framework/params.py.backup')

        if backup_path.exists():
            shutil.copy(backup_path, params_path)
            print("[INFO] Restored original params.py")

    def modify_main_train(self, config):
        """
        main_train.py를 수정하여 use_recurrent 설정 반영
        RNN/Feedforward 선택을 위해
        """
        main_train_path = Path('main_train.py')
        backup_path = Path('main_train.py.backup')

        # Backup if not exists
        if not backup_path.exists():
            shutil.copy(main_train_path, backup_path)

        # This is a placeholder - actual implementation would require
        # modifying trainer.py to accept use_recurrent parameter
        # For now, we'll document this in the config
        pass

    def run_single_experiment(self, config_name, config, seed):
        """단일 실험 수행"""
        print(f"\n{'='*80}")
        print(f"Running: {config_name} with seed={seed}")
        print(f"Description: {config.get('description', 'N/A')}")
        print(f"{'='*80}\n")

        # Apply configuration
        self.apply_config_to_params(config)

        # Set environment variable for seed
        env = os.environ.copy()
        env['RANDOM_SEED'] = str(seed)

        try:
            # Run training
            result = subprocess.run(
                ['conda', 'run', '-n', self.conda_env, 'python', 'main_train.py'],
                env=env,
                capture_output=True,
                text=True,
                timeout=7200  # 2 hour timeout
            )

            print(result.stdout[-2000:])  # Print last 2000 chars

            if result.stderr:
                print("STDERR:", result.stderr[-1000:])

            # Find latest run directories
            import glob
            a3c_runs = sorted(glob.glob('runs/a3c_*'))
            ind_runs = sorted(glob.glob('runs/individual_*'))

            if not a3c_runs or not ind_runs:
                print(f"[WARNING] No run directories found for {config_name} seed={seed}")
                return None

            latest_a3c = a3c_runs[-1]
            latest_ind = ind_runs[-1]
            timestamp = os.path.basename(latest_a3c).replace('a3c_', '')

            # Extract results
            result_data = {
                'config_name': config_name,
                'seed': seed,
                'timestamp': timestamp,
            }

            # A3C results
            a3c_summary_path = os.path.join(latest_a3c, 'summary_global.csv')
            if os.path.exists(a3c_summary_path):
                a3c_summary = pd.read_csv(a3c_summary_path)
                result_data['a3c_final_reward'] = a3c_summary['reward'].iloc[-1]
                result_data['a3c_final_value_loss'] = a3c_summary['value_loss'].iloc[-1]
                result_data['a3c_final_policy_loss'] = a3c_summary['policy_loss'].iloc[-1]

            # Individual results (average across workers)
            n_workers = config.get('n_workers', 5)
            ind_rewards = []
            ind_value_losses = []
            ind_policy_losses = []

            for worker_id in range(n_workers):
                ind_summary_path = os.path.join(latest_ind, f'summary_Individual_{worker_id}.csv')
                if os.path.exists(ind_summary_path):
                    ind_summary = pd.read_csv(ind_summary_path)
                    ind_rewards.append(ind_summary['reward'].iloc[-1])
                    ind_value_losses.append(ind_summary['value_loss'].iloc[-1])
                    ind_policy_losses.append(ind_summary['policy_loss'].iloc[-1])

            if ind_rewards:
                result_data['individual_final_reward'] = np.mean(ind_rewards)
                result_data['individual_final_value_loss'] = np.mean(ind_value_losses)
                result_data['individual_final_policy_loss'] = np.mean(ind_policy_losses)
                result_data['individual_reward_std'] = np.std(ind_rewards)

            # Move results to study directory
            exp_dir = self.study_dir / config_name / f"seed_{seed}"
            exp_dir.mkdir(parents=True, exist_ok=True)

            shutil.move(latest_a3c, exp_dir / 'a3c')
            shutil.move(latest_ind, exp_dir / 'individual')

            print(f"\n[Results for {config_name}, seed={seed}]")
            print(f"  A3C Final Reward       : {result_data.get('a3c_final_reward', 'N/A'):.3f}")
            print(f"  Individual Final Reward: {result_data.get('individual_final_reward', 'N/A'):.3f}")

            return result_data

        except subprocess.TimeoutExpired:
            print(f"[WARNING] {config_name} with seed={seed} timed out")
            return None
        except Exception as e:
            print(f"[ERROR] {config_name} with seed={seed} failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run_ablation_study(self, ablation_names=None, phase=None, priority=None):
        """
        Ablation study 실행

        Args:
            ablation_names: 실행할 ablation 이름 리스트 (None이면 전체)
            phase: 특정 phase만 실행 (None이면 전체)
            priority: 특정 priority만 실행 (None이면 전체)
        """
        # Determine which ablations to run
        if ablation_names:
            ablations_to_run = {name: ALL_ABLATIONS[name] for name in ablation_names}
        elif phase:
            ablations_to_run = get_ablations_by_phase(phase)
        elif priority:
            ablations_to_run = get_ablations_by_priority(priority)
        else:
            ablations_to_run = ALL_ABLATIONS

        # Always include baseline
        ablations_to_run = {'baseline': BASELINE_CONFIG, **ablations_to_run}

        print(f"\n{'='*80}")
        print(f"Starting Ablation Study")
        print(f"{'='*80}")
        print(f"Total ablations to run: {len(ablations_to_run)}")
        print(f"Seeds per ablation: {self.n_seeds}")
        print(f"Total experiments: {len(ablations_to_run) * self.n_seeds}")
        print(f"{'='*80}\n")

        all_results = []

        for idx, (ablation_name, ablation_info) in enumerate(ablations_to_run.items()):
            print(f"\n[{idx+1}/{len(ablations_to_run)}] Processing: {ablation_name}")

            # Get full config
            config = get_config(ablation_name)

            # Save config to study directory
            config_file = self.study_dir / ablation_name / 'config.json'
            config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)

            # Run with multiple seeds
            for seed in self.seeds:
                result = self.run_single_experiment(ablation_name, config, seed)
                if result:
                    all_results.append(result)

        # Restore original parameters
        self.restore_params()

        # Aggregate and save results
        if all_results:
            self.aggregate_results(all_results)
        else:
            print("\n[ERROR] No results collected!")

    def aggregate_results(self, all_results):
        """결과 집계 및 분석"""
        df = pd.DataFrame(all_results)

        # Save raw results
        raw_csv = self.study_dir / 'raw_results.csv'
        df.to_csv(raw_csv, index=False)
        print(f"\n[Saved] Raw results: {raw_csv}")

        # Aggregate by configuration
        agg_metrics = {
            'a3c_final_reward': ['mean', 'std', 'min', 'max'],
            'individual_final_reward': ['mean', 'std', 'min', 'max'],
            'a3c_final_value_loss': ['mean', 'std'],
            'individual_final_value_loss': ['mean', 'std'],
        }

        summary = df.groupby('config_name').agg(agg_metrics)
        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]

        # Add performance metrics
        summary['a3c_advantage_mean'] = (
            summary['a3c_final_reward_mean'] - summary['individual_final_reward_mean']
        )
        summary['a3c_advantage_pct'] = (
            100 * summary['a3c_advantage_mean'] / summary['individual_final_reward_mean']
        )

        # Sort by A3C performance
        summary = summary.sort_values('a3c_final_reward_mean', ascending=False)

        # Save summary
        summary_csv = self.study_dir / 'summary_results.csv'
        summary.to_csv(summary_csv)
        print(f"[Saved] Summary results: {summary_csv}")

        # Print summary
        print("\n" + "="*80)
        print("ABLATION STUDY SUMMARY")
        print("="*80)
        print("\nTop 5 Configurations (by A3C performance):")
        print(summary[['a3c_final_reward_mean', 'a3c_final_reward_std',
                      'a3c_advantage_mean', 'a3c_advantage_pct']].head(10))

        # Compare to baseline
        if 'baseline' in summary.index:
            baseline_a3c = summary.loc['baseline', 'a3c_final_reward_mean']
            baseline_ind = summary.loc['baseline', 'individual_final_reward_mean']

            print("\n" + "="*80)
            print("COMPARISON TO BASELINE")
            print("="*80)
            print(f"Baseline A3C reward: {baseline_a3c:.3f}")
            print(f"Baseline Individual reward: {baseline_ind:.3f}")
            print()

            comparison = pd.DataFrame({
                'config': summary.index,
                'a3c_delta': summary['a3c_final_reward_mean'].values - baseline_a3c,
                'a3c_delta_pct': 100 * (summary['a3c_final_reward_mean'].values - baseline_a3c) / baseline_a3c,
                'ind_delta': summary['individual_final_reward_mean'].values - baseline_ind,
                'ind_delta_pct': 100 * (summary['individual_final_reward_mean'].values - baseline_ind) / baseline_ind,
            })

            comparison_csv = self.study_dir / 'baseline_comparison.csv'
            comparison.to_csv(comparison_csv, index=False)
            print(f"[Saved] Baseline comparison: {comparison_csv}")

            print("\nTop improvements over baseline:")
            print(comparison.sort_values('a3c_delta_pct', ascending=False).head(10))
            print("\nTop degradations from baseline:")
            print(comparison.sort_values('a3c_delta_pct', ascending=True).head(10))

def main():
    parser = argparse.ArgumentParser(description='Run ablation study experiments')
    parser.add_argument('--phase', type=int, choices=[1, 2, 3, 4],
                       help='Run specific phase only')
    parser.add_argument('--priority', choices=['high', 'medium', 'low'],
                       help='Run specific priority only')
    parser.add_argument('--ablations', nargs='+',
                       help='Specific ablation names to run')
    parser.add_argument('--n-seeds', type=int, default=3,
                       help='Number of random seeds per ablation (default: 3)')
    parser.add_argument('--conda-env', type=str, default='torch-cert',
                       help='Conda environment name (default: torch-cert)')
    parser.add_argument('--output-dir', type=str, default='ablation_results',
                       help='Output directory for results (default: ablation_results)')

    args = parser.parse_args()

    runner = AblationStudyRunner(
        output_dir=args.output_dir,
        n_seeds=args.n_seeds,
        conda_env=args.conda_env
    )

    runner.run_ablation_study(
        ablation_names=args.ablations,
        phase=args.phase,
        priority=args.priority
    )

if __name__ == "__main__":
    main()
