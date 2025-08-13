import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.distributions import Categorical
import threading
import datetime as dt
import pandas as pd
import os, csv, time, queue
from .networks import ActorCritic
from .params import *
from .custom_env import make_env
from .utils import flatten_dict_values
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
import matplotlib.pyplot as plt
from collections import defaultdict
from queue import Queue

temp_env_fn = make_env(**ENV_PARAMS)
temp_env = temp_env_fn()
sample_obs = temp_env.reset()[0]
state_dim = len(flatten_dict_values(sample_obs))
action_dim = temp_env.action_space.n

@dataclass
class EpisodeLog:
    step: int                   # 전역 스텝(옵션) 또는 에피소드 번호
    worker_id: int
    episode: int
    reward: float
    length: int
    policy_loss: Optional[float] = None
    value_loss: Optional[float] = None
    entropy: Optional[float] = None
    total_loss: Optional[float] = None

MASTER_CSV = os.path.join("runs", "all_training_metrics.csv")

def _ensure_master_csv_header():
    os.makedirs("runs", exist_ok=True)
    if not os.path.exists(MASTER_CSV):
        with open(MASTER_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["run_id","label","episode","reward","policy_loss","value_loss","entropy","total_loss"])

def _write_summary_csv(df: pd.DataFrame, label: str, out_csv_path: str, run_id: str):
    """
    df: columns = step, worker_id, episode, reward, length, policy_loss, value_loss, entropy, total_loss
    label: "Global" 또는 "Individual_{n}"
    out_csv_path: 이번 실행(run) 전용 요약 CSV 저장 경로
    run_id: 예) a3c_20250813_153012
    """
    # 에피소드 단위 평균으로 요약
    epi = df.groupby("episode").agg(
        reward=("reward","mean"),
        policy_loss=("policy_loss","mean"),
        value_loss=("value_loss","mean"),
        entropy=("entropy","mean"),
        total_loss=("total_loss","mean"),
    ).reset_index()

    epi.insert(0, "label", label)
    epi.to_csv(out_csv_path, index=False)

    # 마스터 CSV에도 append
    _ensure_master_csv_header()
    epi_master = epi.copy()
    epi_master.insert(0, "run_id", run_id)
    epi_master[["run_id","label","episode","reward","policy_loss","value_loss","entropy","total_loss"]].to_csv(
        MASTER_CSV, mode="a", header=False, index=False
    )

class TrainingLogger:
    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.csv_path = os.path.join(log_dir, "training_log.csv")
        self.fieldnames = ["step","worker_id","episode","reward","length",
                           "policy_loss","value_loss","entropy","total_loss"]
        # CSV 헤더
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
        self.buffer: List[Dict[str, Any]] = []

    def log(self, ep: EpisodeLog):
        self.buffer.append(asdict(ep))
        # 버퍼를 너무 키우지 않도록 주기적으로 flush
        if len(self.buffer) >= 100:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerows(self.buffer)
        self.buffer.clear()

    def finalize(self):
        self.flush()

def plot_curves_from_csv(csv_path: str, out_path_prefix: str):
    import pandas as pd
    df = pd.read_csv(csv_path)
    # 워커 여러 개일 경우 에피소드 기준 집계(평균)
    epi = df.groupby("episode").agg(
        reward=("reward","mean"),
        length=("length","mean"),
        policy_loss=("policy_loss","mean"),
        value_loss=("value_loss","mean"),
        entropy=("entropy","mean"),
        total_loss=("total_loss","mean")
    ).reset_index()

    # 1) 리워드
    plt.figure()
    plt.plot(epi["episode"], epi["reward"])
    plt.xlabel("Episode")
    plt.ylabel("Avg Reward")
    plt.title("Training Curve - Reward")
    plt.tight_layout()
    plt.savefig(out_path_prefix + "_reward.png", dpi=180)
    plt.close()

    # 2) 로스(선택: 값이 있으면 그림)
    for k,ylabel in [("total_loss","Total Loss"),
                     ("policy_loss","Policy Loss"),
                     ("value_loss","Value Loss"),
                     ("entropy","Entropy")]:
        if k in epi.columns and epi[k].notna().any():
            plt.figure()
            plt.plot(epi["episode"], epi[k])
            plt.xlabel("Episode")
            plt.ylabel(ylabel)
            plt.title(f"Training Curve - {ylabel}")
            plt.tight_layout()
            plt.savefig(out_path_prefix + f"_{k}.png", dpi=180)
            plt.close()


class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-4):
        super(SharedAdam, self).__init__(params, lr=lr)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)
                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

# def universal_worker(worker_id, model, optimizer, env_fn, log_path, 
#                      use_global_model=True, 
#                      total_episodes=1000,
#                      metrics_queue: Optional[Queue] = None
#                      ):
#     torch.manual_seed(123 + worker_id)
#     env = env_fn()
    
#     if use_global_model:
#         local_model = ActorCritic(state_dim, action_dim, hidden_dim).to(device)
#         local_model.load_state_dict(model.state_dict())
#         working_model = local_model
#     else:
#         working_model = model

#     for episode in range(1, total_episodes + 1):
#         state, _ = env.reset()
#         done = False
#         total_reward = 0.0
#         episode_steps = 0

#         # 에피소드 동안 데이터 수집
#         states, actions, rewards, values, log_probs, entropies = [], [], [], [], [], []

#         while not done and episode_steps < ENV_PARAMS['max_epoch_size']:
#             state_tensor = torch.FloatTensor(flatten_dict_values(state)).unsqueeze(0).to(device)
#             logits, value = working_model(state_tensor)
#             probs = torch.softmax(logits, dim=-1)
#             log_prob = torch.log(probs + 1e-8)
#             entropy = -(log_prob * probs).sum(1, keepdim=True)

#             action = probs.multinomial(num_samples=1).detach()
#             selected_log_prob = log_prob.gather(1, action)

#             next_state, reward, done, _, _ = env.step(action.item())

#             # 데이터 저장
#             states.append(state_tensor)
#             actions.append(action)
#             rewards.append(reward)
#             values.append(value)
#             log_probs.append(selected_log_prob)
#             entropies.append(entropy)

#             total_reward += reward
#             episode_steps += 1
#             state = next_state

#         # 에피소드 종료 후 returns 계산
#         returns = []
#         R = 0.0 if done else working_model(torch.FloatTensor(flatten_dict_values(state)).unsqueeze(0).to(device))[1].item()
        
#         # Discounted returns 계산 (역순으로)
#         for i in reversed(range(len(rewards))):
#             R = rewards[i] + gamma * R
#             returns.insert(0, R)
        
#         returns = torch.FloatTensor(returns).to(device)
        
#         # Advantage 계산
#         values_tensor = torch.cat(values).squeeze()
#         advantages = returns - values_tensor
        
#         # 정규화 (선택적)
#         if len(advantages) > 1:
#             advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

#         # Loss 계산
#         policy_loss = 0
#         value_loss = 0
#         entropy_loss = 0
        
#         for i in range(len(log_probs)):
#             # Policy loss (REINFORCE with baseline)
#             policy_loss -= log_probs[i] * advantages[i].detach()
            
#             # Value loss (MSE)
#             value_loss += (returns[i] - values[i]).pow(2)
            
#             # Entropy loss (exploration 장려)
#             entropy_loss -= entropies[i]

#         # 총 loss
#         total_loss = (policy_loss + 
#                      value_loss_coef * value_loss + 
#                      entropy_coef * entropy_loss)

#         # Gradient 업데이트
#         optimizer.zero_grad()
#         total_loss.backward()

#         if use_global_model:
#             # A3C: global model 업데이트
#             torch.nn.utils.clip_grad_norm_(working_model.parameters(), max_grad_norm)
#             for lp, gp in zip(working_model.parameters(), model.parameters()):
#                 if gp._grad is None:
#                     gp._grad = lp.grad.clone()
#                 else:
#                     gp._grad += lp.grad
#             optimizer.step()
#             working_model.load_state_dict(model.state_dict())
#         else:
#             # Individual: 자체 모델 업데이트
#             torch.nn.utils.clip_grad_norm_(working_model.parameters(), max_grad_norm)
#             optimizer.step()

#         # 로깅
#         avg_loss = total_loss.item() / len(rewards)  # 스텝당 평균 loss
        
#         with open(log_path, mode="a", newline="") as f:
#             writer = csv.writer(f)
#             writer.writerow([episode, total_reward, avg_loss])

#         if episode % 100 == 0:
#             print(f"[Worker {worker_id}] Episode {episode}, Reward: {total_reward:.2f}, Avg Loss: {avg_loss:.4f}")

def universal_worker(worker_id, model, optimizer, env_fn, log_path, 
                     use_global_model=True, 
                     total_episodes=1000,
                     metrics_queue: Optional[Queue] = None
                     ):
    torch.manual_seed(123 + worker_id)
    env = env_fn()
    
    if use_global_model:
        local_model = ActorCritic(state_dim, action_dim, hidden_dim).to(device)
        local_model.load_state_dict(model.state_dict())
        working_model = local_model
    else:
        working_model = model

    global_step = 0  # (옵션) 전역 스텝 카운트

    for episode in range(1, total_episodes + 1):
        state, _ = env.reset()
        done = False
        total_reward = 0.0
        episode_steps = 0

        states, actions, rewards, values, log_probs, entropies = [], [], [], [], [], []

        while not done and episode_steps < ENV_PARAMS['max_epoch_size']:
            state_tensor = torch.FloatTensor(flatten_dict_values(state)).unsqueeze(0).to(device)
            # logits, value = working_model(state_tensor)
            # probs = torch.softmax(logits, dim=-1)
            # log_prob = torch.log(probs + 1e-8)
            # entropy = -(log_prob * probs).sum(1, keepdim=True)

            # action = probs.multinomial(num_samples=1).detach()
            # selected_log_prob = log_prob.gather(1, action)
            # ★ 수치 안정적 버전 (logits로 직접 작업)
            logits, value = working_model(state_tensor)
            log_prob = torch.log_softmax(logits, dim=-1)   # ← 수치안정 (기존: torch.log(probs + 1e-8))
            probs    = torch.softmax(logits, dim=-1)
            entropy  = -(log_prob * probs).sum(dim=1)      # shape: [1]

            action = probs.multinomial(num_samples=1).detach()     # shape: [1,1]
            selected_log_prob = log_prob.gather(1, action).squeeze(1)  # shape: [1]

            next_state, reward, done, _, _ = env.step(action.item())

            # 저장
            states.append(state_tensor)
            actions.append(action)
            rewards.append(reward)
            # values.append(value)
            # log_probs.append(selected_log_prob)
            # entropies.append(entropy)
            values.append(value.view(-1))                  # (1,1) -> (1,)
            log_probs.append(selected_log_prob.view(-1))   # (1,)   -> (1,)
            entropies.append(entropy.view(-1))             # (1,)   -> (1,)

            total_reward += reward
            episode_steps += 1
            global_step += 1
            state = next_state

        # Returns
        R_bootstrap = 0.0 if done else working_model(
            torch.FloatTensor(flatten_dict_values(state)).unsqueeze(0).to(device)
        )[1].item()

        returns = []
        R = R_bootstrap
        for i in reversed(range(len(rewards))):
            R = rewards[i] + gamma * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns).to(device)

        # Advantage
        values_tensor = torch.cat(values).squeeze()
        advantages = returns - values_tensor
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Loss
        # policy_loss = 0
        # value_loss = 0
        # entropy_loss = 0
        # for i in range(len(log_probs)):
        #     policy_loss -= log_probs[i] * advantages[i].detach()
        #     value_loss += (returns[i] - values[i]).pow(2)
        #     entropy_loss -= entropies[i]

        # total_loss = (policy_loss +
        #               value_loss_coef * value_loss +
        #               entropy_coef * entropy_loss)
        # ★ 리스트 -> 텐서로 스택
        log_probs_t = torch.stack(log_probs)          # [T]
        values_t    = torch.stack(values)             # [T]
        entropies_t = torch.stack(entropies)          # [T]

        # ★ mean 기반 손실 (스케일 안정)
        policy_loss   = -(log_probs_t * advantages.detach()).mean()
        value_loss    = F.mse_loss(values_t, returns)         # == ((returns-values_t)**2).mean()
        entropy_bonus =  entropies_t.mean()

        # ★ 엔트로피는 보너스(탐색 증가) → 마이너스 부호로 더함
        total_loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy_bonus


        # Update
        optimizer.zero_grad()
        total_loss.backward()
        if use_global_model:
            torch.nn.utils.clip_grad_norm_(working_model.parameters(), max_grad_norm)
            for lp, gp in zip(working_model.parameters(), model.parameters()):
                if gp._grad is None:
                    gp._grad = lp.grad.clone()
                else:
                    gp._grad += lp.grad
            optimizer.step()
            working_model.load_state_dict(model.state_dict())
        else:
            torch.nn.utils.clip_grad_norm_(working_model.parameters(), max_grad_norm)
            optimizer.step()

        # ====== CSV 로깅(기존) ======
        # avg_loss = total_loss.item() / max(1, len(rewards))
        avg_loss = float(total_loss.detach().item())

        with open(log_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([episode, total_reward, avg_loss])

        # ====== 에피소드 메트릭을 큐로 전송(신규) ======
        if metrics_queue is not None:
            # 에피소드 평균 손실/엔트로피 계산
            # avg_policy_loss = float(policy_loss.detach().item()) / max(1, len(rewards))
            # avg_value_loss  = float(value_loss.detach().item())  / max(1, len(rewards))
            # # entropy_loss는 -(entropy)를 누적 → 평균 엔트로피는 부호 반전
            # avg_entropy     = float(-entropy_loss.detach().item()) / max(1, len(rewards))
            # avg_total_loss  = float(total_loss.detach().item())   / max(1, len(rewards))
            avg_policy_loss = float(policy_loss.detach().item())
            avg_value_loss  = float(value_loss.detach().item())
            avg_entropy     = float(entropy_bonus.detach().item())     # ★ 양수
            avg_total_loss  = float(total_loss.detach().item())

            metrics_queue.put({
                "step": global_step,          # (옵션) 전역 스텝
                "worker_id": worker_id,
                "episode": episode,
                "reward": float(total_reward),
                "length": int(episode_steps),
                "policy_loss": avg_policy_loss,
                "value_loss": avg_value_loss,
                "entropy": avg_entropy,
                "total_loss": avg_total_loss,
            })

        if episode % 100 == 0:
            print(f"[Worker {worker_id}] Episode {episode}, Reward: {total_reward:.2f}, Avg Loss: {avg_loss:.4f}")


# def train(n_workers, total_episodes, env_param_list=None):
#     mp.set_start_method("spawn", force=True)

#     global_model = ActorCritic(state_dim, action_dim, hidden_dim)
#     global_model.share_memory()
#     global_model = global_model.to(device)
#     optimizer = SharedAdam(global_model.parameters(), lr=lr)

#     os.makedirs("models", exist_ok=True)
#     os.makedirs("logs", exist_ok=True)

#     for worker_id in range(n_workers):
#         log_path = os.path.join("logs", f"A3C_worker_{worker_id}_rewards.csv")
#         with open(log_path, mode="w", newline="") as f:
#             writer = csv.writer(f)
#             writer.writerow(["episode", "reward", "loss"])

#     processes = []
#     for worker_id in range(n_workers):
#         env_params = env_param_list[worker_id] if env_param_list else ENV_PARAMS
#         env_fn = make_env(**env_params, reward_params=REWARD_PARAMS)
#         log_path = os.path.join("logs", f"A3C_worker_{worker_id}_rewards.csv")
#         p = mp.Process(target=universal_worker, args=(worker_id, global_model, optimizer, env_fn, log_path, True, total_episodes))
#         p.start()
#         processes.append(p)

#     for p in processes:
#         p.join()

#     torch.save(global_model.state_dict(), "models/global_final.pth")
#     print("Training complete. Final model saved.")


# def train_individual(n_workers, total_episodes, env_param_list=None):
#     # os.makedirs("models_individual", exist_ok=True)
#     # os.makedirs("logs_individual", exist_ok=True)

#     for worker_id in range(n_workers):
#         env_params = env_param_list[worker_id] if env_param_list else ENV_PARAMS
#         env_fn = make_env(**env_params, reward_params=REWARD_PARAMS)

#         model = ActorCritic(state_dim, action_dim, hidden_dim).to(device)
#         optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#         log_path = os.path.join("logs", f"individual_worker_{worker_id}_rewards.csv")
#         with open(log_path, mode="w", newline="") as f:
#             writer = csv.writer(f)
#             writer.writerow(["episode", "reward", "loss"])

#         print(f"[Worker {worker_id}] Starting individual training")
#         universal_worker(worker_id, model, optimizer, env_fn, log_path, use_global_model=False, total_episodes=total_episodes)

#         torch.save(model.state_dict(), f"models/individual_worker_{worker_id}_final.pth")
#         print(f"[Worker {worker_id}] Training complete. Model saved.")


# ---- 보조 함수: CSV -> 곡선 저장 ----
def _plot_curves_from_csv(csv_path: str, out_prefix: str):
    df = pd.read_csv(csv_path)
    # 에피소드별 평균(여러 워커 평균)
    epi = df.groupby("episode").agg(
        reward=("reward","mean"),
        length=("length","mean"),
        policy_loss=("policy_loss","mean"),
        value_loss=("value_loss","mean"),
        entropy=("entropy","mean"),
        total_loss=("total_loss","mean")
    ).reset_index()

    # Reward
    plt.figure()
    plt.plot(epi["episode"], epi["reward"])
    plt.xlabel("Episode"); plt.ylabel("Avg Reward"); plt.title("Training Curve - Reward")
    plt.tight_layout(); plt.savefig(out_prefix + "_reward.png", dpi=180); plt.close()

    # Loss curves (존재하는 것만)
    for k, ylabel in [("total_loss","Total Loss"),
                      ("policy_loss","Policy Loss"),
                      ("value_loss","Value Loss"),
                      ("entropy","Entropy")]:
        if k in epi.columns and epi[k].notna().any():
            plt.figure()
            plt.plot(epi["episode"], epi[k])
            plt.xlabel("Episode"); plt.ylabel(ylabel); plt.title(f"Training Curve - {ylabel}")
            plt.tight_layout(); plt.savefig(out_prefix + f"_{k}.png", dpi=180); plt.close()


# ---- 멀티프로세싱 수집 루프(부모 프로세스에서 실행) ----
def _collector_process(metrics_queue: mp.Queue, csv_path: str, worker_ps: list):
    fieldnames = ["step","worker_id","episode","reward","length","policy_loss","value_loss","entropy","total_loss"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        alive = True
        while alive or not metrics_queue.empty():
            alive = any(p.is_alive() for p in worker_ps)
            try:
                item = metrics_queue.get(timeout=0.2)
                # 누락 키 보정(없으면 None)
                for k in fieldnames:
                    item.setdefault(k, None)
                writer.writerow(item)
            except queue.Empty:
                pass


# ---- 스레드 기반 수집 루프(train_individual 용) ----
def _collector_thread(metrics_queue: "queue.Queue", csv_path: str, stop_event: threading.Event):
    fieldnames = ["step","worker_id","episode","reward","length","policy_loss","value_loss","entropy","total_loss"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        while not (stop_event.is_set() and metrics_queue.empty()):
            try:
                item = metrics_queue.get(timeout=0.2)
                for k in fieldnames:
                    item.setdefault(k, None)
                writer.writerow(item)
            except queue.Empty:
                pass

# 새로 추가
def _collector_thread_for_mp(metrics_queue, csv_path, any_alive_fn):
    import csv, queue
    fieldnames = ["step","worker_id","episode","reward","length","policy_loss","value_loss","entropy","total_loss"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        while any_alive_fn() or not metrics_queue.empty():
            try:
                item = metrics_queue.get(timeout=0.2)
                for k in fieldnames: item.setdefault(k, None)
                w.writerow(item)
            except queue.Empty:
                pass


# ==========================
# 수정된 train()
# ==========================
def train(n_workers, total_episodes, env_param_list=None):
    mp.set_start_method("spawn", force=True)

    # 타임스탬프 기반 로그/모델 폴더
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", f"a3c_{stamp}")
    logs_dir = run_dir
    models_dir = os.path.join(run_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # 공통 CSV 경로
    agg_csv = os.path.join(logs_dir, "training_log.csv")

    global_model = ActorCritic(state_dim, action_dim, hidden_dim)
    global_model.share_memory()
    global_model = global_model.to(device)
    optimizer = SharedAdam(global_model.parameters(), lr=lr)

    # (선택) 기존 per-worker csv 초기화가 필요하면 유지
    for worker_id in range(n_workers):
        log_path = os.path.join(logs_dir, f"A3C_worker_{worker_id}_rewards.csv")
        with open(log_path, mode="w", newline="") as f:
            writer = csv.writer(f); writer.writerow(["episode", "reward", "loss"])

    # 메트릭 수집용 큐 & 콜렉터 프로세스
    metrics_queue = mp.Queue(maxsize=10000)
    processes = []

    # 워커 시작
    for worker_id in range(n_workers):
        env_params = env_param_list[worker_id] if env_param_list else ENV_PARAMS
        env_fn = make_env(**env_params, reward_params=REWARD_PARAMS)
        log_path = os.path.join(logs_dir, f"A3C_worker_{worker_id}_rewards.csv")

        p = mp.Process(
            target=universal_worker,
            args=(worker_id, global_model, optimizer, env_fn, log_path),
            kwargs=dict(
                use_global_model=True,
                total_episodes=total_episodes,
                metrics_queue=metrics_queue,   # ★ 큐 전달
            )
        )
        p.start()
        processes.append(p)

    # 콜렉터 시작(부모 프로세스에서 실행)
    collector = threading.Thread(
        target=_collector_thread_for_mp,      # 새로 정의
        args=(metrics_queue, agg_csv, lambda: any(p.is_alive() for p in processes)),
        daemon=True
    )
    collector.start()

    # 워커 종료 대기
    for p in processes:
        p.join()

    # 콜렉터 종료 대기
    collector.join()

    # 곡선 저장
    _plot_curves_from_csv(agg_csv, os.path.join(logs_dir, "curves"))

    # 최종 모델 저장
    final_model_path = os.path.join(models_dir, "global_final.pth")
    torch.save(global_model.state_dict(), final_model_path)
    print(f"Training complete. Final model saved at: {final_model_path}")
    print(f"Logs & curves saved under: {logs_dir}")

    # 곡선 저장 직후 (이미 agg_csv = os.path.join(logs_dir, "training_log.csv")가 있음)
    df = pd.read_csv(agg_csv)

    # 이 run의 고유 ID (logs_dir 이름이 곧 run_id)
    run_id = os.path.basename(logs_dir)

    # Global 요약 CSV 저장 (에피소드 평균)
    summary_csv = os.path.join(logs_dir, "summary_global.csv")
    _write_summary_csv(df, label="Global", out_csv_path=summary_csv, run_id=run_id)

    print(f"[Summary] Global metrics saved: {summary_csv}")
    print(f"[Master] Appended to {MASTER_CSV}")



# ==========================
# 수정된 train_individual()
# ==========================
def train_individual(n_workers, total_episodes, env_param_list=None):
    # 타임스탬프 기반 로그/모델 폴더
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", f"individual_{stamp}")
    logs_dir = run_dir
    models_dir = os.path.join(run_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    agg_csv = os.path.join(logs_dir, "training_log.csv")

    # 수집용 queue & 스레드 콜렉터 시작
    metrics_queue = queue.Queue(maxsize=10000)
    stop_event = threading.Event()
    collector = threading.Thread(target=_collector_thread,
                                 args=(metrics_queue, agg_csv, stop_event),
                                 daemon=True)
    collector.start()

    # 워커들을 같은 프로세스에서 순차 학습
    for worker_id in range(n_workers):
        env_params = env_param_list[worker_id] if env_param_list else ENV_PARAMS
        env_fn = make_env(**env_params, reward_params=REWARD_PARAMS)

        model = ActorCritic(state_dim, action_dim, hidden_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # (선택) 개별 csv도 유지하려면 아래 유지
        log_path = os.path.join(logs_dir, f"individual_worker_{worker_id}_rewards.csv")
        with open(log_path, mode="w", newline="") as f:
            writer = csv.writer(f); writer.writerow(["episode", "reward", "loss"])

        print(f"[Worker {worker_id}] Starting individual training")
        universal_worker(
            worker_id, model, optimizer, env_fn, log_path,
            use_global_model=False,
            total_episodes=total_episodes,
            metrics_queue=metrics_queue   # ★ 큐 전달
        )

        # 워커별 최종 모델 저장
        model_path = os.path.join(models_dir, f"individual_worker_{worker_id}_final.pth")
        torch.save(model.state_dict(), model_path)
        print(f"[Worker {worker_id}] Training complete. Model saved at: {model_path}")

    # 콜렉터 종료
    stop_event.set()
    collector.join()

    # 곡선 저장
    _plot_curves_from_csv(agg_csv, os.path.join(logs_dir, "curves"))
    print(f"Logs & curves saved under: {logs_dir}")

    df = pd.read_csv(agg_csv)
    run_id = os.path.basename(logs_dir)

    # worker_id 별로 개별 요약 CSV 생성 (에피소드 평균: 사실상 그 워커의 기록이므로 평균=해당 워커 값)
    for wid in sorted(df["worker_id"].dropna().unique()):
        sub = df[df["worker_id"] == wid].copy()
        if sub.empty:
            continue
        label = f"Individual_{int(wid)}"
        out_csv = os.path.join(logs_dir, f"summary_{label}.csv")
        _write_summary_csv(sub, label=label, out_csv_path=out_csv, run_id=run_id)
        print(f"[Summary] {label} metrics saved: {out_csv}")

    print(f"[Master] Appended to {MASTER_CSV}")
