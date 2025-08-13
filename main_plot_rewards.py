import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import drl_framework.params as params

# 설정
log_dir = params.log_dir
n_workers = params.n_workers
target_episode_count = params.target_episode_count
window_size = params.target_episode_count // 5  # rolling 평균 윈도우 크기

# reward와 loss를 담을 DataFrame
A3C_logs = pd.DataFrame(index=np.arange(1, target_episode_count + 1))
indiv_logs = pd.DataFrame(index=np.arange(1, target_episode_count + 1))

# A3C 워커들의 reward/loss
for i in range(n_workers):
    path = os.path.join(log_dir, f"A3C_worker_{i}_rewards.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        df = df.set_index("episode")
        A3C_logs[f"A3C_Reward_{i}"] = df["reward"]
        A3C_logs[f"A3C_Loss_{i}"] = df["loss"]

# Individual 워커들의 reward/loss
for i in range(n_workers):
    path = os.path.join(log_dir, f"individual_worker_{i}_rewards.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        df = df.set_index("episode")
        indiv_logs[f"Indiv_Reward_{i}"] = df["reward"]
        indiv_logs[f"Indiv_Loss_{i}"] = df["loss"]

# Plot: Reward 비교
plt.figure(figsize=(12, 6))
for i in range(n_workers):
    if f"A3C_Reward_{i}" in A3C_logs:
        plt.plot(A3C_logs[f"A3C_Reward_{i}"].rolling(window=window_size).mean(), label=f"A3C Worker {i}")
    if f"Indiv_Reward_{i}" in indiv_logs:
        plt.plot(indiv_logs[f"Indiv_Reward_{i}"].rolling(window=window_size).mean(), linestyle="--", label=f"Indiv Worker {i}")

plt.title(f"Average Episode Reward per Worker (rolling {window_size})")
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/compare_episode_reward.png")

# Plot: Loss 비교
plt.figure(figsize=(12, 6))
for i in range(n_workers):
    if f"A3C_Loss_{i}" in A3C_logs:
        plt.plot(A3C_logs[f"A3C_Loss_{i}"].rolling(window=window_size).mean(), label=f"A3C Worker {i}")
    if f"Indiv_Loss_{i}" in indiv_logs:
        plt.plot(indiv_logs[f"Indiv_Loss_{i}"].rolling(window=window_size).mean(), linestyle="--", label=f"Indiv Worker {i}")

plt.title(f"Average Episode Loss per Worker (rolling {window_size})")
plt.xlabel("Episode")
plt.ylabel("Average Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/compare_episode_loss.png")
