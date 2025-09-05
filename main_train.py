from drl_framework.trainer import train, train_individual
from drl_framework.network_state import NetworkState
import drl_framework.params as params
import copy
import numpy as np

n_workers = params.n_workers
target_episode_count = params.target_episode_count  # trainer.py에서도 동일하게 사용됨
ENV_PARAMS = params.ENV_PARAMS

# 난수 기반 환경 리스트 생성
env_param_list = []
for i in range(n_workers):
    e = copy.deepcopy(ENV_PARAMS)
    # e["max_comp_units"] = np.random.randint(80, 121)
    e["max_comp_units"] = np.arange(40, 161, 10)[i % n_workers]  # 40, 50, ..., 160
    # e["agent_velocities"] = np.arange(30, 101)[i % n_workers] 
    env_param_list.append(e)
# Do action masking!!
if __name__ == "__main__":
    train(n_workers=n_workers, 
          total_episodes=target_episode_count, 
          env_param_list=env_param_list)
    train_individual(n_workers=n_workers, 
                     total_episodes=target_episode_count, 
                     env_param_list=env_param_list)
