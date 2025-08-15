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
    def __init__(self, max_comp_units, max_epoch_size, max_queue_size, reward_weights=1,
                 agent_velocities=None, seed=None, reward_params=None):
        super().__init__()
        self.max_comp_units = max_comp_units
        self.max_epoch_size = max_epoch_size
        self.max_queue_size = max_queue_size
        self.reward_weight = reward_weights
        self.agent_velocities = agent_velocities if agent_velocities else 10
        self.max_available_computation_units = max_comp_units
        self.max_channel_quality = 2
        self.max_remain_epochs = max_epoch_size
        self.max_proc_times = int(np.ceil(max_epoch_size / 2))

        self.action_space = spaces.Discrete(2)
        self.reward = 0

        self.observation_space = spaces.Dict({
            "available_computation_units": spaces.Discrete(self.max_available_computation_units),
            "channel_quality": spaces.Discrete(self.max_channel_quality),
            "remain_epochs": spaces.Discrete(self.max_remain_epochs),
            "mec_comp_units": spaces.MultiDiscrete([max_comp_units] * max_queue_size),
            "mec_proc_times": spaces.MultiDiscrete([max_epoch_size] * max_queue_size),
            "queue_comp_units": spaces.Discrete(max_comp_units, start=1),
            "queue_proc_times": spaces.Discrete(max_epoch_size, start=1),
        })

        import time
        from numpy.random import default_rng
        if seed is None:
            seed = int(time.time() * 1000) % 2**32
        self.rng = default_rng(seed)
        self.reward_params = reward_params or {}
    
    def get_obs(self):
        return {"available_computation_units": self.available_computation_units,
                # "number_of_associated_terminals": self.number_of_associated_terminals,
                "channel_quality": self.channel_quality,
                "remain_epochs": self.remain_epochs,
                "mec_comp_units": self.mec_comp_units,
                "mec_proc_times": self.mec_proc_times,
                "queue_comp_units": self.queue_comp_units,
                "queue_proc_times": self.queue_proc_times}
    
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

        if self.channel_quality == 0:  # Bad state
            return 1 if random.random() > TRAN_00 else 0
        else:   # Good state
            return 0 if random.random() > TRAN_11 else 1

    
    def fill_first_zero(self, arr, value):
        for i in range(len(arr)):
            if arr[i] == 0:
                arr[i] = value
                break
        return arr
    
    

    def reset(self, seed=None, options=None):
        """
        Returns: observation
        """
        super().reset(seed=seed)

        self.available_computation_units = self.max_available_computation_units
        # self.number_of_associated_terminals = self.rng.integers(1, self.max_number_of_associated_terminals + 1, size=1)
        self.channel_quality = self.rng.integers(0, self.max_channel_quality)
        self.remain_epochs = self.max_remain_epochs
        self.remain_processing = 0
        self.mec_comp_units = np.zeros(self.max_queue_size, dtype=int)
        self.mec_proc_times = np.zeros(self.max_queue_size, dtype=int)
        self.queue_comp_units = self.rng.integers(1, self.max_comp_units + 1)
        self.queue_proc_times = self.rng.integers(1, self.max_proc_times + 1)

        self.reward = 0

        observation = self.get_obs()
        
        return observation, {}
    
    def step(self, action):
        """
        Returns: observation, reward, terminated, truncated, info
        """
        ALPHA = self.reward_params.get('ALPHA', 0.1)
        BETA = self.reward_params.get('BETA', 0.5)
        GAMMA = self.reward_params.get('GAMMA', 2)
        SCALE = self.reward_params.get('REWARD_SCALE', 1.0)
        PENALTY = self.reward_params.get('FAILURE_PENALTY', 5.0)
        comp_units = self.queue_comp_units
        proc_time = self.queue_proc_times
        success = False
        energy = 0.0
        self.reward = 0
        # forwarding phase
        # 0: local process, 1: offload
        if action == 0:  # Local process
            case_action = ((self.available_computation_units >= self.queue_comp_units) and 
                           (self.mec_comp_units[self.mec_comp_units == 0].size > 0) and
                           (self.queue_comp_units > 0))
            if case_action:
                self.available_computation_units -= self.queue_comp_units
                self.mec_comp_units = self.fill_first_zero(self.mec_comp_units, self.queue_comp_units)
                self.mec_proc_times = self.fill_first_zero(self.mec_proc_times, self.queue_proc_times)
                success = True
                energy = ALPHA * comp_units
                latency = proc_time
            else:
                success = False
                energy = 0.0
                latency = proc_time
        elif action == 1:   # Offload
            if self.queue_comp_units > 0:
                if self.channel_quality == 1:
                    success = True
                    energy = BETA * comp_units
                    latency = GAMMA * proc_time
                elif self.channel_quality == 0:
                    success = False
                    energy = BETA * comp_units
                    latency = GAMMA * proc_time
        else:
            raise ValueError("Invalid action")

        if success:
            efficiency = comp_units / (latency + 1e-8)  # Avoid division by zero
            self.reward = SCALE * (efficiency - energy)
        else:
            efficiency = 0
            self.reward = SCALE * (efficiency - energy) - PENALTY

        self.queue_comp_units = self.rng.integers(1, self.max_comp_units + 1)
        self.queue_proc_times = self.rng.integers(1, self.max_proc_times + 1)
            
        self.channel_quality = self.change_channel_quality()
        self.remain_epochs = self.remain_epochs - 1

        # Processing phase
        zeroed_index = (self.mec_proc_times == 1)
        if zeroed_index.any():
            self.reward += self.mec_comp_units[zeroed_index].sum()
            
            self.available_computation_units += self.mec_comp_units[zeroed_index].sum()
            self.mec_proc_times = np.concatenate([self.mec_proc_times[zeroed_index == False], np.zeros(zeroed_index.sum(), dtype=int)])
            self.mec_comp_units = np.concatenate([self.mec_comp_units[zeroed_index == False], np.zeros(zeroed_index.sum(), dtype=int)])
            
        self.mec_proc_times = np.clip(self.mec_proc_times - 1, 0, self.max_proc_times)

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
    return partial(CustomEnv, **kwargs)