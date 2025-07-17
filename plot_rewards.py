import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import drl_framework.params as params

# 설정
log_dir = params.log_dir
n_workers = params.n_workers
target_episode_count = params.target_episode_count
window_size = params.target_episode_count // 10  # rolling 평균 윈도우 크기

# reward와 loss를 담을 DataFrame
reward_logs = pd.DataFrame(index=np.arange(1, target_episode_count + 1))

# CSV 파일에서 각 워커의 reward 및 loss 읽기
for i in range(n_workers):
    path = os.path.join(log_dir, f"worker_{i}_rewards.csv")
    df = pd.read_csv(path)
    df = df.set_index("episode")
    reward_logs[f"Reward {i}"] = df["reward"]
    reward_logs[f"Loss {i}"] = df["loss"]

# Plot: Reward
plt.figure(figsize=(12, 6))
for i in range(n_workers):
    avg_reward = reward_logs[f"Reward {i}"].rolling(window=window_size).mean()
    plt.plot(avg_reward, label=f"Worker {i} Reward")

plt.title(f"Average Episode Reward per Worker (rolling {window_size})")
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("logs/average_episode_reward.png")

# Plot: Loss
plt.figure(figsize=(12, 6))
for i in range(n_workers):
    avg_loss = reward_logs[f"Loss {i}"].rolling(window=window_size).mean()
    plt.plot(avg_loss, label=f"Worker {i} Loss")

plt.title(f"Average Episode Loss per Worker (rolling {window_size})")
plt.xlabel("Episode")
plt.ylabel("Average Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("logs/average_episode_loss.png")
