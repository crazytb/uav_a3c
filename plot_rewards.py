import matplotlib.pyplot as plt
import numpy as np
import os
import csv
from drl_framework.params import n_workers, log_dir, target_episode_count
from drl_framework.trainer import train
import pandas as pd


reward_logs = pd.DataFrame(index=np.arange(1, target_episode_count + 1))

for i in range(n_workers):
    path = os.path.join(log_dir, f"worker_{i}_rewards.csv")
    rewards = []
    with open(path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) == 2:
                try:
                    rewards.append(float(row[1]))
                except ValueError:
                    rewards.append(np.nan)
    # 워커 열 추가
    reward_logs[f"Worker {i}"] = pd.Series(rewards, index=np.arange(1, len(rewards) + 1))

# Plot
plt.figure(figsize=(12, 6))
for worker in reward_logs.columns:
    avg = reward_logs[worker].rolling(window=100).mean()
    plt.plot(avg, label=worker)

plt.title("Average Episode Reward per Worker")
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
