from drl_framework.trainer import train, train_individual
from drl_framework.params import ENV_PARAMS
import copy
import numpy as np

n_workers = 5
target_episode_count = 5000  # trainer.py에서도 동일하게 사용됨

# 난수 기반 환경 리스트 생성
env_param_list = []
for i in range(n_workers):
    e = copy.deepcopy(ENV_PARAMS)
    e["max_comp_units"] = np.random.randint(80, 121)
    e["agent_velocities"] = np.random.randint(30, 101)
    env_param_list.append(e)

if __name__ == "__main__":
    train(n_workers=n_workers, 
          total_episodes=target_episode_count, 
          env_param_list=env_param_list)
    train_individual(n_workers=n_workers, 
                     total_episodes=target_episode_count, 
                     env_param_list=env_param_list)
