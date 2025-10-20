from drl_framework.trainer import train, train_individual
from drl_framework.network_state import NetworkState
import drl_framework.params as params
import copy
import numpy as np
import torch
import random
import os

# Random seed setting (for reproducibility)
seed = int(os.environ.get('RANDOM_SEED', 42))  # Default seed: 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

print(f"[INFO] Using random seed: {seed}")
print()

n_workers = params.n_workers
target_episode_count = params.target_episode_count  # trainer.py에서도 동일하게 사용됨
ENV_PARAMS = params.ENV_PARAMS

# 난수 기반 환경 리스트 생성
env_param_list = []
for i in range(n_workers):
    e = copy.deepcopy(ENV_PARAMS)
    # e["max_comp_units"] = np.random.randint(80, 121)
    # n_worker개 워커용: 80~200을 n_worker개로 나누기
    # e["max_comp_units"] = int(80 + (200 - 80) * i / (n_workers - 1)) if n_workers > 1 else 140
    # n_worker개 워커용: 5~25을 n_worker개로 나누기
    e["agent_velocities"] = int(5 + (25 - 5) * i / (n_workers - 1)) if n_workers > 1 else 10
    env_param_list.append(e)
# Do action masking!!
if __name__ == "__main__":
    train(n_workers=n_workers,
          total_episodes=target_episode_count,
          env_param_list=env_param_list)
    train_individual(n_workers=n_workers,
                     total_episodes=target_episode_count,
                     env_param_list=env_param_list)
