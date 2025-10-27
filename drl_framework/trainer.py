from email import policy
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.distributions import Categorical
import threading
import datetime as dt
import pandas as pd
import os, csv, time, queue
# from .networks import ActorCritic
from .networks import RecurrentActorCritic
from .network_state import NetworkState
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
ROLL_OUT_LEN = 20
tbtt_steps = ROLL_OUT_LEN

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

stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
MASTER_CSV = os.path.join("runs", f"all_training_metrics_{stamp}.csv")

def _ensure_master_csv_header():
    os.makedirs("runs", exist_ok=True)
    if not os.path.exists(MASTER_CSV):
        with open(MASTER_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["run_id","label","episode","reward","policy_loss","value_loss","entropy","total_loss"])
    
def _write_summary_csv(df: pd.DataFrame, label: str, out_csv_path: str, run_id: str):
    epi = df.groupby("episode").agg(
        reward=("reward","mean"),
        policy_loss=("policy_loss","mean"),
        value_loss=("value_loss","mean"),
        entropy=("entropy","mean"),
        total_loss=("total_loss","mean"),
    ).reset_index()

    epi.insert(0, "label", label)
    epi.to_csv(out_csv_path, index=False)

    epi_master = epi.copy()
    epi_master.insert(0, "run_id", run_id)

    # ✨ 여기가 핵심
    write_header = not os.path.exists(MASTER_CSV) or os.stat(MASTER_CSV).st_size == 0

    epi_master.to_csv(
        MASTER_CSV, mode="a", header=write_header, index=False
    )

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

def detach_hidden(hx):
    # GRU: Tensor, LSTM: (h, c)
    if isinstance(hx, tuple):
        return tuple(h.detach() for h in hx)
    return hx.detach()

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


def universal_worker(worker_id, model, optimizer, env_fn, log_path,
                     use_global_model=True,
                     total_episodes=1000,
                     metrics_queue: Optional[Queue] = None,
                     episode_barrier=None,
                     network_state=None
                     ):
    torch.manual_seed(123 + worker_id)
    env = env_fn()

    # --- (A) 로컬/글로벌 모델 준비 (RNN) ---
    if use_global_model:
        local_model = RecurrentActorCritic(state_dim, action_dim, hidden_dim, use_layer_norm=use_layer_norm).to(device)
        local_model.load_state_dict(model.state_dict())
        working_model = local_model
    else:
        working_model = model.to(device)

    global_step = 0
    # 로깅용 파일 초기화 유지
    with open(log_path, mode="a", newline="") as f:
        pass

    # Step-level logging 파일 초기화
    step_log_path = log_path.replace('.csv', '_step_log.csv')
    with open(step_log_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['worker_id', 'episode', 'step', 'action',
                        'local_comp_units', 'cloud_comp_units', 'reward'])

    for episode in range(1, total_episodes + 1):
        if episode_barrier:
            episode_barrier.wait()
        if network_state:
            network_state.register_worker_start(worker_id)

        state, _ = env.reset()
        done = False
        total_reward = 0.0
        episode_steps = 0
        accum_policy_loss = 0.0
        accum_value_loss = 0.0
        accum_entropy = 0.0

        # --- (B) 에피소드 시작 시 hidden state 초기화 ---
        hx = working_model.init_hidden(batch_size=1, device=device)  # (L=1, B=1, H)

        while not done and episode_steps < ENV_PARAMS['max_epoch_size']:
            # --------- (C) T-step 롤아웃 수집 ----------
            obs_seq, act_seq, rew_seq, done_seq = [], [], [], []
            hx_roll_start = hx.detach()  # 이 시점의 hidden을 저장(rollout 역전파용)

            t = 0
            last_probs_np = None  # 디버그 출력용
            while t < ROLL_OUT_LEN and (not done) and episode_steps < ENV_PARAMS['max_epoch_size']:
                # 관측 전처리
                obs_tensor = torch.as_tensor(
                    flatten_dict_values(state), dtype=torch.float32, device=device
                ).unsqueeze(0)  # (1, state_dim)

                # --- (C-1) 1-step RNN 전파 및 행동 샘플 (액션 마스킹 제거) ---
                with torch.no_grad():
                    logits, value, hx = working_model.step(obs_tensor, hx)

                    # 액션 마스킹 제거: 모든 action을 시도 가능하도록 함
                    dist = Categorical(logits=logits)
                    action = dist.sample()                      # (1,)
                    logp  = dist.log_prob(action)               # (1,)  # 저장은 이후 rollout 재계산에서 함
                    probs = torch.softmax(logits, dim=-1)       # (1, A)
                last_probs_np = probs.detach().cpu().numpy()

                # 환경 한 스텝
                next_state, reward, done, _, _ = env.step(action.item())

                # Step-level logging: 현재 상태 기록 (원본 값, 정규화 안 된 값)
                with open(step_log_path, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    # 로컬 comp_units (환경에서 원본 값 가져오기 - 정규화 안 된 값)
                    local_comp = env.available_computation_units if hasattr(env, 'available_computation_units') else 0
                    # 클라우드 comp_units (network_state에서 가져오기)
                    cloud_comp = network_state.available_cloud_capacity.value if network_state else 0
                    writer.writerow([worker_id, episode, episode_steps + 1, action.item(),
                                    local_comp, cloud_comp, float(reward)])

                # 버퍼 저장 (학습 시점에 rollout()으로 다시 평가)
                obs_seq.append(obs_tensor.squeeze(0))               # (state_dim,)
                act_seq.append(action.squeeze(0))                   # ()
                rew_seq.append(torch.as_tensor(reward, dtype=torch.float32, device=device))
                done_seq.append(torch.as_tensor(done, dtype=torch.bool, device=device))

                total_reward += float(reward)
                episode_steps += 1
                global_step += 1
                state = next_state
                t += 1

                # done이면 hidden 즉시 리셋 (수집 안정화)
                if done:
                    with torch.no_grad():
                        hx = hx * 0.0

            # --------- (D) 텐서 스택 및 부트스트랩 값 ---------
            # (B,T,...) 형태로 만들기 위해 B=1 차원 추가
            B, T = 1, len(obs_seq)
            if T == 0:
                break  # 보호

            obs_seq_t  = torch.stack(obs_seq, dim=0).unsqueeze(0)    # (1,T,state_dim)
            act_seq_t  = torch.stack(act_seq, dim=0).unsqueeze(0)    # (1,T)
            rew_seq_t  = torch.stack(rew_seq, dim=0).unsqueeze(0)    # (1,T)
            done_seq_t = torch.stack(done_seq, dim=0).unsqueeze(0)   # (1,T)

            with torch.no_grad():
                # 마지막 상태의 V(s_T)로 부트스트랩 (done이면 0)
                if done:
                    v_last = torch.zeros(B, device=device)
                else:
                    last_obs_tensor = torch.as_tensor(
                        flatten_dict_values(state), dtype=torch.float32, device=device
                    ).unsqueeze(0)  # (1,state_dim)
                    _, v_last_, _ = working_model.step(last_obs_tensor, hx)
                    v_last = v_last_.squeeze(-1)  # (1,)

            # --------- (E) 롤아웃 재평가: logits/values 시퀀스 ----------
            logits_seq, values_seq, _ = working_model.rollout(
                x_seq=obs_seq_t, hx=hx_roll_start, done_seq=done_seq_t
            )  # (1,T,A), (1,T,1)

            # 액션 마스킹 제거: 원본 logits로 학습
            dist_seq = Categorical(logits=logits_seq)
            logp_seq = dist_seq.log_prob(act_seq_t)            # (1,T)
            entropy_seq = dist_seq.entropy()                   # (1,T)

            values_seq = values_seq.squeeze(-1)                # (1,T)
            done_f = done_seq_t.float()                        # (1,T)
            mask = 1.0 - done_f                                # (1,T)

            # --------- (F) Returns & Advantage (부트스트랩) ----------
            returns = torch.zeros_like(rew_seq_t)              # (1,T)
            R = v_last                                        # (1,)
            for i in reversed(range(T)):
                R = rew_seq_t[:, i] + gamma * R * mask[:, i]
                returns[:, i] = R

            advantages = returns - values_seq                  # (1,T)

            # (선택) 어드밴티지 정규화
            if T > 1:
                adv_mean = advantages.mean()
                adv_std  = advantages.std().clamp_min(1e-8)
                advantages = (advantages - adv_mean) / adv_std

            # --------- (G) Loss 계산 ----------
            policy_loss   = -(logp_seq * advantages.detach()).mean()
            value_loss    = F.mse_loss(values_seq, returns)
            entropy_bonus =  entropy_seq.mean()

            total_loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy_bonus

            # --------- (H) 업데이트 ----------
            optimizer.zero_grad()
            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(working_model.parameters(), max_grad_norm)


            for name, p in working_model.named_parameters():
                if p.grad is None:
                    print(f"[Worker {worker_id}] ⚠️ Gradient missing: {name}")

            if use_global_model:
                torch.nn.utils.clip_grad_norm_(working_model.parameters(), max_grad_norm)
                # 글로벌 모델 그래디언트 초기화 후 복사 (누적 방지)
                optimizer.zero_grad()
                for lp, gp in zip(working_model.parameters(), model.parameters()):
                    if lp.grad is not None:
                        if gp.grad is None:
                            gp.grad = lp.grad.clone()
                        else:
                            gp.grad.copy_(lp.grad)  # 누적이 아닌 직접 복사
                optimizer.step()
                working_model.load_state_dict(model.state_dict())
            else:
                torch.nn.utils.clip_grad_norm_(working_model.parameters(), max_grad_norm)
                optimizer.step()            
        # while 끝
        # 로깅(에피소드 누적 기준)
        if metrics_queue is not None:
            metrics_queue.put({
                "step": global_step,
                "worker_id": worker_id,
                "episode": episode,
                "reward": float(total_reward),
                "length": int(episode_steps),
                "policy_loss": float(policy_loss.detach().item()),
                "value_loss": float(value_loss.detach().item()),
                "entropy": float(entropy_bonus.detach().item()),
                "total_loss": float(total_loss.detach().item()),
            })

        # 디버그: 최근 확률
        if episode % 100 == 0 and last_probs_np is not None:
            print(f"[Worker {worker_id}] Episode {episode}, Reward: {total_reward:.2f}, "
                    f"Loss: {float(total_loss.detach().item()):.4f}, Example action probs: {last_probs_np.round(3)}")
            # print(f"  Local vs Offload preference: {last_probs_np[0][0]:.3f} vs {last_probs_np[0][1]:.3f}")

        # per-worker csv (기존 유지)
        with open(log_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([episode, total_reward, float(total_loss.detach().item())])

        if network_state:
            network_state.register_worker_end(worker_id)
        if episode_barrier:
            episode_barrier.wait()
            if worker_id == 0:
                network_state.reset_for_new_episode()

# ==========================
# 수정된 train()
# ==========================
def train(n_workers, total_episodes, env_param_list=None):
    mp.set_start_method("spawn", force=True)

    # 타임스탬프 기반 로그/모델 폴더
    # stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", f"a3c_{stamp}")
    logs_dir = run_dir
    models_dir = os.path.join(run_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # 공통 CSV 경로
    agg_csv = os.path.join(logs_dir, "training_log.csv")

    global_model = RecurrentActorCritic(state_dim, action_dim, hidden_dim, use_layer_norm=use_layer_norm)
    global_model.share_memory()
    global_model = global_model.to(device)
    optimizer = SharedAdam(global_model.parameters(), lr=lr)

    # (선택) 기존 per-worker csv 초기화가 필요하면 유지
    for worker_id in range(n_workers):
        log_path = os.path.join(logs_dir, f"A3C_worker_{worker_id}_rewards.csv")
        with open(log_path, mode="w", newline="") as f:
            writer = csv.writer(f); writer.writerow(["episode", "reward", "loss"])

    # 동기화 객체들 생성
    episode_barrier = mp.Barrier(n_workers) if n_workers > 1 else None
    # max_cloud_capacity = ENV_PARAMS['max_comp_units_for_cloud', 10000]
    network_state = NetworkState(n_workers) if n_workers > 1 else None

    # 메트릭 수집용 큐 & 콜렉터 프로세스
    metrics_queue = mp.Queue(maxsize=10000)
    processes = []

    # 워커 시작
    for worker_id in range(n_workers):
        env_params = env_param_list[worker_id] if env_param_list else ENV_PARAMS
        # env_params['network_state'] = network_state
        env_fn = make_env(**env_params, reward_params=REWARD_PARAMS, 
                          network_state=network_state, worker_id=worker_id)
        log_path = os.path.join(logs_dir, f"A3C_worker_{worker_id}_rewards.csv")

        p = mp.Process(
            target=universal_worker,
            args=(worker_id, global_model, optimizer, env_fn, log_path),
            kwargs=dict(
                use_global_model=True,
                total_episodes=total_episodes,
                metrics_queue=metrics_queue,   # ★ 큐 전달
                episode_barrier=episode_barrier,
                network_state=network_state
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

    # 최종 모델 저장 (메타데이터 포함)
    final_model_path = os.path.join(models_dir, "global_final.pth")

    # training_log.csv에서 최종 통계 추출
    df = pd.read_csv(agg_csv)
    if len(df) > 0:
        final_episode = df['episode'].max()
        final_reward = df.groupby('episode')['reward'].mean().iloc[-1]
        total_steps = df['step'].max()
    else:
        final_episode = 0
        final_reward = 0.0
        total_steps = 0

    torch.save({
        'model_state_dict': global_model.state_dict(),
        'episode': int(final_episode),
        'total_steps': int(total_steps),
        'final_reward': float(final_reward),
        'n_workers': n_workers,
        'timestamp': stamp,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'hidden_dim': hidden_dim
    }, final_model_path)

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



# # ==========================
# # 수정된 train_individual()
# # ==========================
# def train_individual(n_workers, total_episodes, env_param_list=None):
#     # 타임스탬프 기반 로그/모델 폴더
#     # stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
#     run_dir = os.path.join("runs", f"individual_{stamp}")
#     logs_dir = run_dir
#     models_dir = os.path.join(run_dir, "models")
#     os.makedirs(models_dir, exist_ok=True)
#     os.makedirs(logs_dir, exist_ok=True)

#     agg_csv = os.path.join(logs_dir, "training_log.csv")

#     # 수집용 queue & 스레드 콜렉터 시작
#     metrics_queue = queue.Queue(maxsize=10000)
#     stop_event = threading.Event()
#     collector = threading.Thread(target=_collector_thread,
#                                  args=(metrics_queue, agg_csv, stop_event),
#                                  daemon=True)
#     collector.start()

#     # 워커들을 같은 프로세스에서 순차 학습
#     for worker_id in range(n_workers):
#         env_params = env_param_list[worker_id] if env_param_list else ENV_PARAMS
#         env_fn = make_env(**env_params, reward_params=REWARD_PARAMS)

#         model = RecurrentActorCritic(state_dim, action_dim, hidden_dim).to(device)
#         optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#         # (선택) 개별 csv도 유지하려면 아래 유지
#         log_path = os.path.join(logs_dir, f"individual_worker_{worker_id}_rewards.csv")
#         with open(log_path, mode="w", newline="") as f:
#             writer = csv.writer(f); writer.writerow(["episode", "reward", "loss"])

#         print(f"[Worker {worker_id}] Starting individual training")
#         universal_worker(
#             worker_id, model, optimizer, env_fn, log_path,
#             use_global_model=False,
#             total_episodes=total_episodes,
#             metrics_queue=metrics_queue   # ★ 큐 전달
#         )

#         # 워커별 최종 모델 저장
#         model_path = os.path.join(models_dir, f"individual_worker_{worker_id}_final.pth")
#         torch.save(model.state_dict(), model_path)
#         print(f"[Worker {worker_id}] Training complete. Model saved at: {model_path}")

#     # 콜렉터 종료
#     stop_event.set()
#     collector.join()

#     # 곡선 저장
#     _plot_curves_from_csv(agg_csv, os.path.join(logs_dir, "curves"))
#     print(f"Logs & curves saved under: {logs_dir}")

#     df = pd.read_csv(agg_csv)
#     run_id = os.path.basename(logs_dir)

#     # worker_id 별로 개별 요약 CSV 생성 (에피소드 평균: 사실상 그 워커의 기록이므로 평균=해당 워커 값)
#     for wid in sorted(df["worker_id"].dropna().unique()):
#         sub = df[df["worker_id"] == wid].copy()
#         if sub.empty:
#             continue
#         label = f"Individual_{int(wid)}"
#         out_csv = os.path.join(logs_dir, f"summary_{label}.csv")
#         _write_summary_csv(sub, label=label, out_csv_path=out_csv, run_id=run_id)
#         print(f"[Summary] {label} metrics saved: {out_csv}")

#     print(f"[Master] Appended to {MASTER_CSV}")

# trainer.py의 train_individual 함수를 multiprocessing으로 수정

def individual_worker(worker_id, env_fn, log_path, total_episodes, 
                     metrics_queue=None, episode_barrier=None, network_state=None):
    """Individual A2C worker (모델 공유 없음)"""
    torch.manual_seed(123 + worker_id)
    env = env_fn()

    # 개별 모델 생성 (공유하지 않음)
    local_model = RecurrentActorCritic(state_dim, action_dim, hidden_dim, use_layer_norm=use_layer_norm).to(device)
    optimizer = torch.optim.Adam(local_model.parameters(), lr=lr)

    global_step = 0
    with open(log_path, mode="a", newline="") as f:
        pass

    for episode in range(1, total_episodes + 1):
        # A3C와 동일한 에피소드 동기화
        if episode_barrier:
            episode_barrier.wait()
        if network_state:
            network_state.register_worker_start(worker_id)

        state, _ = env.reset()
        done = False
        total_reward = 0.0
        episode_steps = 0

        # RNN hidden state 초기화
        hx = local_model.init_hidden(batch_size=1, device=device)

        while not done and episode_steps < ENV_PARAMS['max_epoch_size']:
            # ... A3C와 동일한 rollout 및 학습 로직 ...
            # (universal_worker의 코드와 거의 동일, 단지 local_model만 업데이트)
            
            obs_seq, act_seq, rew_seq, done_seq = [], [], [], []
            hx_roll_start = hx.detach()

            t = 0
            while t < ROLL_OUT_LEN and (not done) and episode_steps < ENV_PARAMS['max_epoch_size']:
                obs_tensor = torch.as_tensor(
                    flatten_dict_values(state), dtype=torch.float32, device=device
                ).unsqueeze(0)

                with torch.no_grad():
                    logits, value, hx = local_model.step(obs_tensor, hx)

                    # 액션 마스킹 제거: 모든 action을 시도 가능하도록 함
                    dist = Categorical(logits=logits)
                    action = dist.sample()
                    probs = torch.softmax(logits, dim=-1)       # (1, A)
                last_probs_np = probs.detach().cpu().numpy()

                next_state, reward, done, _, _ = env.step(action.item())

                obs_seq.append(obs_tensor.squeeze(0))
                act_seq.append(action.squeeze(0))
                rew_seq.append(torch.as_tensor(reward, dtype=torch.float32, device=device))
                done_seq.append(torch.as_tensor(done, dtype=torch.bool, device=device))

                total_reward += float(reward)
                episode_steps += 1
                global_step += 1
                state = next_state
                t += 1

                if done:
                    with torch.no_grad():
                        hx = hx * 0.0

            # 학습 (개별 모델만 업데이트)
            if len(obs_seq) > 0:
                # ... 동일한 loss 계산 및 업데이트 로직 ...
                # 단, local_model만 업데이트하고 global model은 건드리지 않음
                
                B, T = 1, len(obs_seq)
                obs_seq_t = torch.stack(obs_seq, dim=0).unsqueeze(0)
                act_seq_t = torch.stack(act_seq, dim=0).unsqueeze(0)
                rew_seq_t = torch.stack(rew_seq, dim=0).unsqueeze(0)
                done_seq_t = torch.stack(done_seq, dim=0).unsqueeze(0)

                with torch.no_grad():
                    if done:
                        v_last = torch.zeros(B, device=device)
                    else:
                        last_obs_tensor = torch.as_tensor(
                            flatten_dict_values(state), dtype=torch.float32, device=device
                        ).unsqueeze(0)
                        _, v_last_, _ = local_model.step(last_obs_tensor, hx)
                        v_last = v_last_.squeeze(-1)

                logits_seq, values_seq, _ = local_model.rollout(
                    x_seq=obs_seq_t, hx=hx_roll_start, done_seq=done_seq_t
                )

                # 액션 마스킹 제거: 원본 logits로 학습
                dist_seq = Categorical(logits=logits_seq)
                logp_seq = dist_seq.log_prob(act_seq_t)
                entropy_seq = dist_seq.entropy()
                values_seq = values_seq.squeeze(-1)
                done_f = done_seq_t.float()
                mask = 1.0 - done_f

                returns = torch.zeros_like(rew_seq_t)
                R = v_last
                for i in reversed(range(T)):
                    R = rew_seq_t[:, i] + gamma * R * mask[:, i]
                    returns[:, i] = R

                advantages = returns - values_seq
                if T > 1:
                    adv_mean = advantages.mean()
                    adv_std = advantages.std().clamp_min(1e-8)
                    advantages = (advantages - adv_mean) / adv_std

                policy_loss = -(logp_seq * advantages.detach()).mean()
                value_loss = F.mse_loss(values_seq, returns)
                entropy_bonus = entropy_seq.mean()
                total_loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy_bonus

                # 개별 모델만 업데이트 (글로벌 모델 없음)
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_grad_norm)
                optimizer.step()

        # 메트릭 로깅
        if metrics_queue is not None:
            metrics_queue.put({
                "step": global_step,
                "worker_id": worker_id,
                "episode": episode,
                "reward": float(total_reward),
                "length": int(episode_steps),
                "policy_loss": float(policy_loss.detach().item()) if 'policy_loss' in locals() else None,
                "value_loss": float(value_loss.detach().item()) if 'value_loss' in locals() else None,
                "entropy": float(entropy_bonus.detach().item()) if 'entropy_bonus' in locals() else None,
                "total_loss": float(total_loss.detach().item()) if 'total_loss' in locals() else None,
            })

        # A3C와 동일한 에피소드 종료 동기화
        if network_state:
            network_state.register_worker_end(worker_id)
        if episode_barrier:
            episode_barrier.wait()
            if worker_id == 0:
                network_state.reset_for_new_episode()

        # per-worker csv 로깅
        with open(log_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([episode, total_reward, float(total_loss.detach().item()) if 'total_loss' in locals() else 0])

        # 디버그: 최근 확률
        if episode % 100 == 0 and last_probs_np is not None:
            print(f"[Worker {worker_id}] Episode {episode}, Reward: {total_reward:.2f}, "
                    f"Loss: {float(total_loss.detach().item()):.4f}, Example action probs: {last_probs_np.round(3)}")

    # 학습 완료 후 모델 저장 (메타데이터 포함)
    run_dir = os.path.dirname(log_path)  # runs/individual_{stamp}
    models_dir = os.path.join(run_dir, "models")  # runs/individual_{stamp}/models
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, f"individual_worker_{worker_id}_final.pth")

    # per-worker CSV 읽어서 최종 reward 계산
    df_worker = pd.read_csv(log_path)
    final_reward_worker = df_worker['reward'].iloc[-1] if len(df_worker) > 0 else 0.0

    torch.save({
        'model_state_dict': local_model.state_dict(),
        'worker_id': worker_id,
        'episode': total_episodes,
        'total_steps': global_step,
        'final_reward': float(final_reward_worker),
        'timestamp': stamp,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'hidden_dim': hidden_dim
    }, model_path)

    print(f"[Worker {worker_id}] Individual training complete. Model saved at: {model_path}")


def train_individual(n_workers, total_episodes, env_param_list=None):
    """Individual A2C with multiprocessing and network sharing"""
    mp.set_start_method("spawn", force=True)

    # A3C와 동일한 동기화 구조 사용
    run_dir = os.path.join("runs", f"individual_{stamp}")
    logs_dir = run_dir
    models_dir = os.path.join(run_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    # 공통 CSV 경로
    agg_csv = os.path.join(logs_dir, "training_log.csv")

    # A3C와 동일한 동기화 객체들 생성
    episode_barrier = mp.Barrier(n_workers)
    network_state = NetworkState(n_workers)

    # 메트릭 수집
    metrics_queue = mp.Queue(maxsize=10000)
    processes = []

    # Individual 워커들을 병렬로 실행
    for worker_id in range(n_workers):
        env_params = env_param_list[worker_id] if env_param_list else ENV_PARAMS
        env_fn = make_env(**env_params, reward_params=REWARD_PARAMS, 
                         network_state=network_state, worker_id=worker_id)

        log_path = os.path.join(logs_dir, f"individual_worker_{worker_id}_rewards.csv")
        with open(log_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "reward", "loss"])

        p = mp.Process(
            target=individual_worker,
            args=(worker_id, env_fn, log_path, total_episodes),
            kwargs=dict(
                metrics_queue=metrics_queue,
                episode_barrier=episode_barrier,  # A3C와 동일한 동기화
                network_state=network_state,      # A3C와 동일한 네트워크 상태
            )
        )
        p.start()
        processes.append(p)

    # A3C와 동일한 collector 구조
    collector = threading.Thread(
        target=_collector_thread_for_mp,
        args=(metrics_queue, agg_csv, lambda: any(p.is_alive() for p in processes)),
        daemon=True
    )
    collector.start()

    # 워커 종료 대기
    for p in processes:
        p.join()

    collector.join()

    # 나머지 로깅 및 정리는 기존과 동일
    _plot_curves_from_csv(agg_csv, os.path.join(logs_dir, "curves"))
    
    # 개별 모델들 저장
    print(f"Individual A2C training complete. Logs saved under: {logs_dir}")

    # 워커별 최종 모델 저장
    # model_path = os.path.join(models_dir, f"individual_worker_{worker_id}_final.pth")
    # torch.save(model.state_dict(), model_path)
    # print(f"[Worker {worker_id}] Training complete. Model saved at: {model_path}")
    
    # 요약 CSV 생성
    df = pd.read_csv(agg_csv)
    run_id = os.path.basename(logs_dir)
    for wid in sorted(df["worker_id"].dropna().unique()):
        sub = df[df["worker_id"] == wid].copy()
        if sub.empty:
            continue
        label = f"Individual_{int(wid)}"
        out_csv = os.path.join(logs_dir, f"summary_{label}.csv")
        _write_summary_csv(sub, label=label, out_csv_path=out_csv, run_id=run_id)
        print(f"[Summary] {label} metrics saved: {out_csv}")

    print(f"[Master] Appended to {MASTER_CSV}")