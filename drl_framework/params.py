import os
from datetime import datetime
import torch
import numpy as np

# Device configuration
# if torch.cuda.is_available():
#     device = torch.device("cuda")
# elif torch.backends.mps.is_available():
#     device = torch.device("mps")
# else:
#     device = torch.device("cpu")
device = torch.device("cpu")  # Force CPU for compatibility

# Logging directory
log_dir = "logs"

# ===== A3C 구조 관련 파라미터 =====
n_workers = 5                  # 병렬 에이전트(worker) 수
# update_interval = 10            # 몇 스텝마다 global model을 업데이트할지
target_episode_count = 5000      # worker 당 총 에피소드 수

# Env params
ENV_PARAMS = {
    # 'max_comp_units': np.random.randint(1, 101),  # Max computation units
    'max_comp_units': 200,  # Max computation units
    'max_comp_units_for_cloud': 1000,  # Max computation units for cloud
    'max_epoch_size': 100,  # Max epoch size
    'max_queue_size': 20,  # Max queue size
    'reward_weights': 1,  # Reward weights
    'agent_velocities': 50  # Agent velocities
}

# Reward 관련 파라미터
REWARD_PARAMS = {
    'ALPHA': 1,                # 로컬 처리 에너지 비용 계수
    'BETA': 0.5,               # 오프로드 시간 비용 계수
    'GAMMA': 2.0,              # 전송 지연 계수
    'REWARD_SCALE': 0.05,      # 보상 스케일링 (1/20) - 중간 값
    'ENERGY_COST_COEFF': 0.0,  # 에너지 비용 감소 (시도 유도)
    'CONGESTION_COST_COEFF': 0.1
}

# ===== 학습 관련 파라미터 =====
gamma = 0.99                   # discount factor
entropy_coef = 0.05             # policy entropy 가중치 - 중간 값 (탐색/exploitation 균형)
value_loss_coef = 0.25          # value loss 가중치 - 중간 값 (안정성과 학습 균형)
lr = 1e-4                       # learning rate (적절한 학습 속도 유지)
max_grad_norm = 2.0             # gradient clipping 임계값 (5.0 -> 2.0, 안정화와 학습 속도의 균형)
hidden_dim = 128               # hidden layer 노드 수

# Set parameters
batch_size = 1
learning_rate = 1e-4 #
buffer_len = int(100000)
min_buffer_len = 20
min_epi_num = 20 # Start moment to train the Q network
episodes = 200 #
print_per_iter = 20
target_update_period = 10 #
eps_start = 0.1
eps_end = 0.01 #
eps_decay = 0.998 #
tau = 1e-2
max_step = 20

# DRQN param
random_update = False # If you want to do random update instead of sequential update
lookup_step = 20 # If you want to do random update instead of sequential update
max_epi_len = 100 
max_epi_step = max_step

