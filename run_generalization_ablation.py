"""
Generalization-based Ablation Study Runner
학습된 모델을 다양한 환경에서 테스트하여 일반화 성능 비교
"""

import os
import sys
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
import json

# Test environment configurations
TEST_CONFIGS = {
    'velocity_sweep': {
        'name': 'Velocity Sweep',
        'description': 'Test across different UAV velocities (channel dynamics)',
        'variations': [
            {'agent_velocities': 5, 'label': 'vel_5'},
            {'agent_velocities': 10, 'label': 'vel_10'},
            {'agent_velocities': 15, 'label': 'vel_15'},
            {'agent_velocities': 20, 'label': 'vel_20'},
            {'agent_velocities': 25, 'label': 'vel_25'},
            {'agent_velocities': 30, 'label': 'vel_30'},
            {'agent_velocities': 50, 'label': 'vel_50'},
            {'agent_velocities': 75, 'label': 'vel_75'},
            {'agent_velocities': 100, 'label': 'vel_100'},
        ]
    },
    'cloud_resource_sweep': {
        'name': 'Cloud Resource Sweep',
        'description': 'Test across different cloud resource availability',
        'variations': [
            {'max_comp_units_for_cloud': 500, 'label': 'cloud_500'},
            {'max_comp_units_for_cloud': 750, 'label': 'cloud_750'},
            {'max_comp_units_for_cloud': 1000, 'label': 'cloud_1000'},
            {'max_comp_units_for_cloud': 1500, 'label': 'cloud_1500'},
            {'max_comp_units_for_cloud': 2000, 'label': 'cloud_2000'},
        ]
    },
    'local_resource_sweep': {
        'name': 'Local Resource Sweep',
        'description': 'Test across different local computation resources',
        'variations': [
            {'max_comp_units': 100, 'label': 'local_100'},
            {'max_comp_units': 150, 'label': 'local_150'},
            {'max_comp_units': 200, 'label': 'local_200'},
            {'max_comp_units': 300, 'label': 'local_300'},
            {'max_comp_units': 400, 'label': 'local_400'},
        ]
    },
}

class GeneralizationAblationRunner:
    def __init__(self, baseline_dir, output_dir='generalization_ablation_results'):
        self.baseline_dir = Path(baseline_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.python_path = os.path.expanduser('~/miniconda3/envs/torch-cert/bin/python')

        print(f"[INFO] Baseline directory: {self.baseline_dir}")
        print(f"[INFO] Output directory: {self.output_dir}")

    def run_single_test(self, model_dir, test_config_name, variation, seed):
        """단일 테스트 환경에서 모델 평가"""

        # 임시 params.py 수정을 위한 스크립트 작성
        test_script = f"""
import sys
import os
sys.path.insert(0, os.getcwd())

# Modify params temporarily
from drl_framework import params

# Apply test variation
"""
        for key, value in variation.items():
            if key != 'label':
                if key in ['agent_velocities', 'max_comp_units', 'max_comp_units_for_cloud']:
                    if key == 'agent_velocities':
                        test_script += f"params.ENV_PARAMS['{key}'] = {value}\n"
                    elif key == 'max_comp_units':
                        test_script += f"params.ENV_PARAMS['{key}'] = {value}\n"
                    elif key == 'max_comp_units_for_cloud':
                        test_script += f"params.ENV_PARAMS['{key}'] = {value}\n"

        test_script += """
# Now run evaluation
import torch
import numpy as np
from drl_framework.networks import RecurrentActorCritic
from drl_framework.custom_env import make_env
from drl_framework.utils import flatten_dict_values

device = params.device
n_test_episodes = 100

# Load models
a3c_model_path = sys.argv[1]
seed_dir = sys.argv[2]

# Test A3C
a3c_model = RecurrentActorCritic(
    state_dim=48,
    action_dim=3,
    hidden_dim=params.hidden_dim,
    use_layer_norm=params.use_layer_norm
).to(device)
a3c_model.load_state_dict(torch.load(a3c_model_path, map_location=device))
a3c_model.eval()

# Create test environment
env_fn = make_env(**params.ENV_PARAMS, reward_params=params.REWARD_PARAMS)
test_env = env_fn()

# Run A3C test
a3c_rewards = []
for ep in range(n_test_episodes):
    obs, _ = test_env.reset()
    obs_flat = flatten_dict_values(obs)
    obs_tensor = torch.FloatTensor(obs_flat).unsqueeze(0).to(device)

    hx = a3c_model.init_hidden(1, device=device)
    episode_reward = 0
    done = False

    while not done:
        with torch.no_grad():
            logits, value, hx = a3c_model.step(obs_tensor, hx)
            action = torch.argmax(logits, dim=1).item()

        obs, reward, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated
        episode_reward += reward

        obs_flat = flatten_dict_values(obs)
        obs_tensor = torch.FloatTensor(obs_flat).unsqueeze(0).to(device)

    a3c_rewards.append(episode_reward)

a3c_mean = np.mean(a3c_rewards)
a3c_std = np.std(a3c_rewards)

# Test Individual workers
ind_rewards_all = []
for worker_id in range(5):
    ind_model_path = f"{seed_dir}/individual/models/individual_worker_{worker_id}_final.pth"

    if not os.path.exists(ind_model_path):
        continue

    ind_model = RecurrentActorCritic(
        state_dim=48,
        action_dim=3,
        hidden_dim=params.hidden_dim,
        use_layer_norm=params.use_layer_norm
    ).to(device)
    ind_model.load_state_dict(torch.load(ind_model_path, map_location=device))
    ind_model.eval()

    worker_rewards = []
    for ep in range(n_test_episodes):
        obs, _ = test_env.reset()
        obs_flat = flatten_dict_values(obs)
        obs_tensor = torch.FloatTensor(obs_flat).unsqueeze(0).to(device)

        hx = ind_model.init_hidden(1, device=device)
        episode_reward = 0
        done = False

        while not done:
            with torch.no_grad():
                logits, value, hx = ind_model.step(obs_tensor, hx)
                action = torch.argmax(logits, dim=1).item()

            obs, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated
            episode_reward += reward

            obs_flat = flatten_dict_values(obs)
            obs_tensor = torch.FloatTensor(obs_flat).unsqueeze(0).to(device)

        worker_rewards.append(episode_reward)

    ind_rewards_all.extend(worker_rewards)

ind_mean = np.mean(ind_rewards_all) if ind_rewards_all else 0.0
ind_std = np.std(ind_rewards_all) if ind_rewards_all else 0.0

# Output results
print(f"A3C:{a3c_mean:.4f},{a3c_std:.4f}")
print(f"IND:{ind_mean:.4f},{ind_std:.4f}")
"""

        # Write test script
        script_path = self.output_dir / f"test_script_{variation['label']}.py"
        with open(script_path, 'w') as f:
            f.write(test_script)

        # Run test
        a3c_model_path = model_dir / 'a3c' / 'models' / 'global_final.pth'

        try:
            result = subprocess.run(
                [self.python_path, str(script_path), str(a3c_model_path), str(model_dir)],
                capture_output=True,
                text=True,
                timeout=600
            )

            # Parse output
            lines = result.stdout.strip().split('\n')
            a3c_line = [l for l in lines if l.startswith('A3C:')]
            ind_line = [l for l in lines if l.startswith('IND:')]

            if a3c_line and ind_line:
                a3c_parts = a3c_line[0].split(':')[1].split(',')
                ind_parts = ind_line[0].split(':')[1].split(',')

                return {
                    'seed': seed,
                    'test_config': test_config_name,
                    'variation_label': variation['label'],
                    'variation_params': {k: v for k, v in variation.items() if k != 'label'},
                    'a3c_mean': float(a3c_parts[0]),
                    'a3c_std': float(a3c_parts[1]),
                    'ind_mean': float(ind_parts[0]),
                    'ind_std': float(ind_parts[1]),
                }
            else:
                print(f"[WARNING] Could not parse output for {variation['label']}")
                return None

        except Exception as e:
            print(f"[ERROR] Test failed for {variation['label']}: {e}")
            return None

    def run_generalization_tests(self, test_config_names=None):
        """모든 seed와 test configuration에 대해 generalization test 수행"""

        if test_config_names is None:
            test_config_names = list(TEST_CONFIGS.keys())

        # Find all seed directories
        seed_dirs = sorted(self.baseline_dir.glob('seed_*'))

        if not seed_dirs:
            print(f"[ERROR] No seed directories found in {self.baseline_dir}")
            return

        print(f"[INFO] Found {len(seed_dirs)} seed directories")
        print(f"[INFO] Test configurations: {test_config_names}")
        print()

        all_results = []

        for seed_dir in seed_dirs:
            seed = int(seed_dir.name.split('_')[1])
            print(f"\n{'='*80}")
            print(f"Testing Seed {seed}")
            print(f"{'='*80}")

            for test_config_name in test_config_names:
                if test_config_name not in TEST_CONFIGS:
                    print(f"[WARNING] Unknown test config: {test_config_name}")
                    continue

                test_config = TEST_CONFIGS[test_config_name]
                print(f"\n[{test_config['name']}] {test_config['description']}")

                for i, variation in enumerate(test_config['variations']):
                    print(f"  [{i+1}/{len(test_config['variations'])}] Testing {variation['label']}...", end=' ')

                    result = self.run_single_test(seed_dir, test_config_name, variation, seed)

                    if result:
                        all_results.append(result)
                        print(f"A3C: {result['a3c_mean']:.2f}±{result['a3c_std']:.2f}, "
                              f"Ind: {result['ind_mean']:.2f}±{result['ind_std']:.2f}")
                    else:
                        print("FAILED")

        # Save results
        if all_results:
            df_results = pd.DataFrame(all_results)
            results_file = self.output_dir / 'generalization_ablation_results.csv'
            df_results.to_csv(results_file, index=False)
            print(f"\n[SAVED] Results: {results_file}")

            # Generate summary
            self.generate_summary(df_results)

        return all_results

    def generate_summary(self, df_results):
        """결과 요약 생성"""
        print("\n" + "="*80)
        print("GENERALIZATION ABLATION SUMMARY")
        print("="*80)

        # Group by test configuration
        for test_config in df_results['test_config'].unique():
            df_test = df_results[df_results['test_config'] == test_config]

            print(f"\n[{test_config.upper()}]")
            print("-" * 80)

            # Aggregate across seeds
            summary = df_test.groupby('variation_label').agg({
                'a3c_mean': ['mean', 'std'],
                'ind_mean': ['mean', 'std']
            })

            print(f"{'Variation':<15} {'A3C Mean':>12} {'A3C Std':>10} {'Ind Mean':>12} {'Ind Std':>10} {'Gap':>8}")
            print("-" * 80)

            for var_label in summary.index:
                a3c_m = summary.loc[var_label, ('a3c_mean', 'mean')]
                a3c_s = summary.loc[var_label, ('a3c_mean', 'std')]
                ind_m = summary.loc[var_label, ('ind_mean', 'mean')]
                ind_s = summary.loc[var_label, ('ind_mean', 'std')]
                gap = a3c_m - ind_m

                print(f"{var_label:<15} {a3c_m:>12.2f} {a3c_s:>10.2f} "
                      f"{ind_m:>12.2f} {ind_s:>10.2f} {gap:>+8.2f}")

        # Overall metrics
        print("\n" + "="*80)
        print("OVERALL METRICS")
        print("="*80)

        a3c_all = df_results['a3c_mean'].values
        ind_all = df_results['ind_mean'].values

        print(f"A3C Generalization Score  : {np.mean(a3c_all):.2f} ± {np.std(a3c_all):.2f}")
        print(f"Ind Generalization Score  : {np.mean(ind_all):.2f} ± {np.std(ind_all):.2f}")
        print(f"A3C Advantage             : {np.mean(a3c_all) - np.mean(ind_all):+.2f}")
        print(f"A3C Robustness (std)      : {np.std(a3c_all):.2f}")
        print(f"Ind Robustness (std)      : {np.std(ind_all):.2f}")
        print(f"A3C Worst-Case            : {np.min(a3c_all):.2f}")
        print(f"Ind Worst-Case            : {np.min(ind_all):.2f}")

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Run generalization-based ablation study')
    parser.add_argument('--baseline-dir', type=str, required=True,
                       help='Directory containing baseline seed results')
    parser.add_argument('--test-configs', nargs='+', choices=list(TEST_CONFIGS.keys()),
                       default=['velocity_sweep'],
                       help='Test configurations to run')
    parser.add_argument('--output-dir', type=str, default='generalization_ablation_results',
                       help='Output directory for results')

    args = parser.parse_args()

    runner = GeneralizationAblationRunner(
        baseline_dir=args.baseline_dir,
        output_dir=args.output_dir
    )

    runner.run_generalization_tests(test_config_names=args.test_configs)

if __name__ == "__main__":
    main()
