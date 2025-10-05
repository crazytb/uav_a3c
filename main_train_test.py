from drl_framework.trainer import train, train_individual
from drl_framework.network_state import NetworkState
import drl_framework.params as params
import copy
import numpy as np

n_workers = params.n_workers
test_episode_count = 200  # 테스트용으로 200 episodes만 실행
ENV_PARAMS = params.ENV_PARAMS

# 난수 기반 환경 리스트 생성
env_param_list = []
for i in range(n_workers):
    e = copy.deepcopy(ENV_PARAMS)
    e["max_comp_units"] = np.arange(80, 201, 30)[i % n_workers]  # 80, 110, 140, 170, 200
    env_param_list.append(e)

if __name__ == "__main__":
    print("="*70)
    print("테스트 학습 시작 (value_loss_coef=0.1, episodes=200)")
    print("="*70)

    # A3C만 실행 (빠른 테스트를 위해)
    train(n_workers=n_workers,
          total_episodes=test_episode_count,
          env_param_list=env_param_list)

    print("="*70)
    print("테스트 학습 완료!")
    print("="*70)
