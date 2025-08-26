import time
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from numpy.random import default_rng
import math
import random
from .params import *
from functools import partial

# https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/
# https://www.youtube.com/@cartoonsondemand

class CustomEnv(gym.Env):
    def __init__(self, max_comp_units, max_epoch_size, max_queue_size, max_comp_units_for_cloud, 
                 reward_weights=1, agent_velocities=None, seed=None, reward_params=None, network_state=None,
                 worker_id=None):
        super().__init__()
        self.max_comp_units = max_comp_units
        self.max_epoch_size = max_epoch_size
        self.max_queue_size = max_queue_size
        self.reward_weight = reward_weights
        self.agent_velocities = agent_velocities if agent_velocities else 10
        self.max_available_computation_units = max_comp_units
        self.max_available_computation_units_for_cloud = max_comp_units_for_cloud
        self.max_channel_quality = 2
        self.max_remain_epochs = max_epoch_size
        self.max_proc_times = int(np.ceil(max_epoch_size / 10))
        # --- Context for normalization (fixed bounds you use when sampling) ---
        self._VEL_MIN, self._VEL_MAX = 30.0, 100.0
        self._COMP_MAX = 120.0  # your randint upper bound
        # Network state
        self.network_state = network_state
        self.worker_id = worker_id

        self.action_space = spaces.Discrete(3)  # 0: LOCAL, 1: OFFLOAD, 2: DISCARD
        self.reward = 0

        # self.observation_space = spaces.Dict({
        #     "available_computation_units": spaces.Discrete(self.max_available_computation_units),
        #     # "channel_quality": spaces.Discrete(self.max_channel_quality),
        #     "remain_epochs": spaces.Discrete(self.max_remain_epochs),
        #     "mec_comp_units": spaces.MultiDiscrete([max_comp_units] * max_queue_size),
        #     "mec_proc_times": spaces.MultiDiscrete([max_epoch_size] * max_queue_size),
        #     "queue_comp_units": spaces.Discrete(max_comp_units, start=1),
        #     "queue_proc_times": spaces.Discrete(max_epoch_size, start=1),
        #     "offload_success": spaces.Discrete(2),
        #     # ---- Context features (continuous) ----
        #     "ctx_vel":  spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
        #     "ctx_comp": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
        #  })

        self.observation_space = spaces.Dict({
            "available_computation_units": spaces.Box(0.0, 1.0, (1,), dtype=np.float32),
            # "channel_quality": spaces.Discrete(self.max_channel_quality),
            "remain_epochs": spaces.Box(0.0, 1.0, (1,), dtype=np.float32),
            "mec_comp_units": spaces.Box(0.0, 1.0, (max_queue_size,), dtype=np.float32),
            "mec_proc_times": spaces.Box(0.0, 1.0, (max_queue_size,), dtype=np.float32),
            # "cloud_comp_units": spaces.Box(0.0, 1.0, (max_queue_size,), dtype=np.float32),
            # "cloud_proc_times": spaces.Box(0.0, 1.0, (max_queue_size,), dtype=np.float32),
            "queue_comp_units": spaces.Box(0.0, 1.0, (1,), dtype=np.float32),
            "queue_proc_times": spaces.Box(0.0, 1.0, (1,), dtype=np.float32),
            "local_success": spaces.Discrete(2),
            "offload_success": spaces.Discrete(2),
            # ---- Context features (continuous) ----
            "ctx_vel":  spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "ctx_comp": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
         })

        import time
        from numpy.random import default_rng
        if seed is None:
            seed = int(time.time() * 1000) % 2**32
        self.rng = default_rng(seed)
        self.reward_params = reward_params or {}

    def _ctx(self):
        v = float(self.agent_velocities)
        v_norm = (v - self._VEL_MIN) / max(1e-6, (self._VEL_MAX - self._VEL_MIN))
        v_norm = float(np.clip(v_norm, 0.0, 1.0))
        c_norm = float(np.clip(self.max_comp_units / self._COMP_MAX, 0.0, 1.0))
        return np.array([v_norm], dtype=np.float32), np.array([c_norm], dtype=np.float32)

    def get_obs(self):
        ctx_vel, ctx_comp = self._ctx()
        return {"available_computation_units": self.available_computation_units / self.max_available_computation_units,
                # "number_of_associated_terminals": self.number_of_associated_terminals,
                "channel_quality": self.channel_quality,
                "remain_epochs": self.remain_epochs / self.max_remain_epochs,
                "mec_comp_units": self.mec_comp_units / self.max_comp_units,
                "mec_proc_times": self.mec_proc_times / self.max_proc_times,
                # "cloud_comp_units": self.cloud_comp_units / self.max_comp_units,
                # "cloud_proc_times": self.cloud_proc_times / self.max_proc_times,
                "queue_comp_units": self.queue_comp_units / self.max_comp_units,
                "queue_proc_times": self.queue_proc_times / self.max_proc_times,
                "local_success": self.local_success,
                "offload_success": self.offload_success,
                "ctx_vel": ctx_vel,
                "ctx_comp": ctx_comp
        }

    def stepfunc(self, thres, x):
        if x > thres:
            return 1
        else:
            return 0
    
    def change_channel_quality(self):
        # State settings
        velocity = self.agent_velocities # km/h
        snr_thr = 15
        snr_ave = snr_thr + 10
        f_0 = 5.9e9 # Carrier freq = 5.9GHz, IEEE 802.11bd
        speedoflight = 300000   # km/sec
        f_d = velocity/(3600*speedoflight)*f_0  # Hz
        packettime = 100*1000/ENV_PARAMS['max_epoch_size']
        # packettime = 5000    # us
        fdtp = f_d*packettime/1e6
        TRAN_01 = (fdtp*math.sqrt(2*math.pi*snr_thr/snr_ave))/(np.exp(snr_thr/snr_ave)-1)
        TRAN_00 = 1 - TRAN_01
        # TRAN_11 = fdtp*math.sqrt((2*math.pi*snr_thr)/snr_ave)
        TRAN_10 = fdtp*math.sqrt((2*math.pi*snr_thr)/snr_ave)
        TRAN_11 = 1 - TRAN_10

        if self.channel_quality == 0:  # Was in Bad state
            return 1 if random.random() > TRAN_00 else 0
        else:   # Good state
            return 0 if random.random() > TRAN_11 else 1

    
    def fill_first_zero(self, arr, value):
        for i in range(len(arr)):
            if arr[i] == 0:
                arr[i] = value
                break
        return arr
    
    def fill_first_zero_pair(self, arr1, arr2, value1, value2):
        """두 배열에 동시에 값을 채우는 헬퍼 함수"""
        for i in range(len(arr1)):
            if arr1[i] == 0:
                arr1[i] = value1
                arr2[i] = value2
                break
        return arr1, arr2

    def reset(self, seed=None):
        """
        Returns: observation
        """
        super().reset(seed=seed)

        self.available_computation_units = self.max_available_computation_units
        self.available_computation_units_for_cloud = self.max_available_computation_units_for_cloud
        # self.number_of_associated_terminals = self.rng.integers(1, self.max_number_of_associated_terminals + 1, size=1)
        self.channel_quality = self.rng.integers(0, self.max_channel_quality)
        self.remain_epochs = self.max_remain_epochs
        self.remain_processing = 0
        self.mec_comp_units = np.zeros(self.max_queue_size, dtype=int)
        self.mec_proc_times = np.zeros(self.max_queue_size, dtype=int)
        self.mec_original_times = np.zeros(self.max_queue_size, dtype=int)
        self.cloud_comp_units = np.zeros(self.max_queue_size, dtype=int)
        self.cloud_proc_times = np.zeros(self.max_queue_size, dtype=int)
        self.cloud_original_times = np.zeros(self.max_queue_size, dtype=int)
        self.queue_comp_units = self.rng.integers(1, self.max_comp_units + 1)
        self.queue_proc_times = self.rng.integers(1, self.max_proc_times + 1)
        self.local_success = 0
        self.offload_success = 0

        self.reward = 0

        observation = self.get_obs()
        
        return observation, {}
    
    def step(self, action):
        """
        Returns: observation, reward, terminated, truncated, info
        """
        LOCAL, OFFLOAD, DISCARD = 0, 1, 2
        ALPHA = self.reward_params.get('ALPHA')
        BETA = self.reward_params.get('BETA')
        GAMMA = self.reward_params.get('GAMMA')
        SCALE = self.reward_params.get('REWARD_SCALE')
        PENALTY = self.reward_params.get('FAILURE_PENALTY')
        ENERGY_COST_COEFF = self.reward_params.get('ENERGY_COST_COEFF')
        CONGESTION_COST_COEFF = self.reward_params.get('CONGESTION_COST_COEFF')

        comp_units = self.queue_comp_units
        proc_time = self.queue_proc_times
        success = False
        energy = 0.0
        congestion_penalty = 0.0
        self.reward = 0
        # forwarding phase
        # 0: local process, 1: offload
        if action == LOCAL:  # Local process
            case_action = ((self.available_computation_units >= self.queue_comp_units) and 
                           (self.mec_comp_units[self.mec_comp_units == 0].size > 0) and
                           (self.queue_comp_units > 0))
            self.reward -= (self.queue_comp_units / self.max_comp_units) * ENERGY_COST_COEFF
            if case_action:
                self.available_computation_units -= self.queue_comp_units
                # 🆕 comp_units와 proc_times를 함께 저장
                self.mec_comp_units = self.fill_first_zero(self.mec_comp_units, self.queue_comp_units)
                self.mec_proc_times = self.fill_first_zero(self.mec_proc_times, self.queue_proc_times)
                self.mec_original_times = self.fill_first_zero(self.mec_original_times, self.queue_proc_times)
                success = True
                # energy = ALPHA * comp_units
                latency = proc_time
                self.local_success = 1
            else:
                success = False
                # energy = 0.0
                latency = proc_time
                self.local_success = 0
        elif action == OFFLOAD:   # Offload
            case_action = ((self.available_computation_units_for_cloud >= self.queue_comp_units) and
                            (self.cloud_comp_units[self.cloud_comp_units == 0].size > 0) and
                            (self.queue_comp_units > 0) and
                            (self.channel_quality == 1))  # Only offload if channel quality is good
            self.reward -= (self.queue_comp_units / self.max_comp_units) * ENERGY_COST_COEFF
            if self.network_state:
                congestion = self.network_state.get_congestion_level()
                congestion_penalty = congestion * 10.0  # 간단한 페널티
                self.network_state.add_offloading_load(self.worker_id, self.queue_comp_units)  # worker_id는 나중에 전달
            self.reward -= congestion_penalty * CONGESTION_COST_COEFF
            if case_action:
                self.available_computation_units_for_cloud -= self.queue_comp_units
                # 🆕 comp_units와 proc_times를 함께 저장
                self.cloud_comp_units = self.fill_first_zero(self.cloud_comp_units, self.queue_comp_units)
                self.cloud_proc_times = self.fill_first_zero(self.cloud_proc_times, self.queue_proc_times)
                self.cloud_original_times = self.fill_first_zero(self.cloud_original_times, self.queue_proc_times)
                success = True
                # energy = BETA * comp_units
                latency = proc_time
                self.offload_success = 1
            else:
                success = False
                # energy = BETA * comp_units
                latency = proc_time
                self.offload_success = 0
        elif action == DISCARD:  # Discard
            success = False
        else:
            raise ValueError("Invalid action")
        
        if not success:
            self.reward -= PENALTY
        
        # 새로운 작업 생성
        self.queue_comp_units = self.rng.integers(1, self.max_comp_units + 1)
        self.queue_proc_times = self.rng.integers(1, self.max_proc_times + 1)
            
        self.channel_quality = self.change_channel_quality()
        self.remain_epochs = self.remain_epochs - 1

        # Processing phase
        zeroed_mec = (self.mec_proc_times == 1)
        if zeroed_mec.any():
            done_comp = self.mec_comp_units[zeroed_mec].sum()
            # 🆕 정확한 원본 처리 시간 사용
            consumed_time = self.mec_original_times[zeroed_mec].sum()
            self.reward += done_comp
            self.available_computation_units += done_comp
            self.mec_proc_times = np.concatenate([self.mec_proc_times[zeroed_mec == False], 
                                                  np.zeros(zeroed_mec.sum(), dtype=int)])
            self.mec_comp_units = np.concatenate([self.mec_comp_units[zeroed_mec == False], 
                                                  np.zeros(zeroed_mec.sum(), dtype=int)])
            self.mec_original_times = np.concatenate([self.mec_original_times[zeroed_mec == False], 
                                                      np.zeros(zeroed_mec.sum(), dtype=int)])

        zeroed_cloud = (self.cloud_proc_times == 1)
        if zeroed_cloud.any():
            done_comp = self.cloud_comp_units[zeroed_cloud].sum()
            # 🆕 정확한 원본 처리 시간 사용
            consumed_time = self.cloud_original_times[zeroed_cloud].sum()
            self.reward += (done_comp - BETA*consumed_time)
            # self.available_computation_units += done_comp
            if self.network_state:
                self.network_state.release_cloud_resource(done_comp)
            self.cloud_proc_times = np.concatenate([self.cloud_proc_times[zeroed_cloud == False], 
                                                    np.zeros(zeroed_cloud.sum(), dtype=int)])
            self.cloud_comp_units = np.concatenate([self.cloud_comp_units[zeroed_cloud == False], 
                                                    np.zeros(zeroed_cloud.sum(), dtype=int)])
            self.cloud_original_times = np.concatenate([self.cloud_original_times[zeroed_cloud == False], 
                                                        np.zeros(zeroed_cloud.sum(), dtype=int)])

        self.mec_proc_times = np.clip(self.mec_proc_times - 1, 0, self.max_proc_times)
        self.cloud_proc_times = np.clip(self.cloud_proc_times - 1, 0, self.max_proc_times)

        next_obs = self.get_obs()

        return next_obs, self.reward, self.remain_epochs == 0, False, {}


    def render(self):
        """
        Returns: None
        """
        pass

    def close(self):
        """
        Returns: None
        """
        pass

def make_env(**kwargs):
    # network_state = kwargs.pop('network_state', None)
    # def _make():
        # return CustomEnv(**kwargs, network_state=network_state)
    # return _make
    return partial(CustomEnv, **kwargs)