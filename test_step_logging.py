"""
Step logging 기능 테스트 (5 episodes만 실행)
"""
from drl_framework.trainer import train
from drl_framework.network_state import NetworkState
import drl_framework.params as params
import copy
import numpy as np
import torch
import random
import os

# Random seed
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

print("[INFO] Testing step logging with 5 episodes and 2 workers")
print()

# 테스트 설정
n_workers = 2
test_episodes = 5

# 환경 파라미터
ENV_PARAMS = params.ENV_PARAMS
env_param_list = []
for i in range(n_workers):
    e = copy.deepcopy(ENV_PARAMS)
    e["max_comp_units"] = int(80 + (200 - 80) * i / (n_workers - 1)) if n_workers > 1 else 140
    e["agent_velocities"] = int(20 + (100 - 20) * i / (n_workers - 1)) if n_workers > 1 else 60
    env_param_list.append(e)
    print(f"Worker {i}: comp_units={e['max_comp_units']}, velocities={e['agent_velocities']}")

print()
print("Starting training...")
print()

try:
    train(n_workers=n_workers,
          total_episodes=test_episodes,
          env_param_list=env_param_list)

    print()
    print("="*80)
    print("Training completed successfully!")
    print("="*80)
    print()

    # Check step log files
    import glob
    step_logs = glob.glob('runs/a3c_*/worker_*_step_log.csv')
    if step_logs:
        print(f"Step log files created: {len(step_logs)}")
        for log_file in step_logs:
            import pandas as pd
            df = pd.read_csv(log_file)
            print(f"  {log_file}: {len(df)} steps logged")
    else:
        print("No step log files found")

except Exception as e:
    print(f"Error occurred: {e}")
    import traceback
    traceback.print_exc()
