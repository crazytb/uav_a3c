# main_evaluation.py  (RNN 평가용)

import torch
import numpy as np
import os
import csv
from torch.distributions import Categorical
from typing import Callable, Dict, Any

from drl_framework.networks import RecurrentActorCritic  # 변경: RNN 모델 사용
from drl_framework.custom_env import make_env
from drl_framework.utils import flatten_dict_values
import drl_framework.params as params
import copy

# 타임스탬프
stamp = "20250905_145022"  # 예시 타임스탬프, 필요에 따라 변경

device = params.device
ENV_PARAMS = params.ENV_PARAMS
REWARD_PARAMS = params.REWARD_PARAMS
hidden_dim = params.hidden_dim
n_workers = params.n_workers

# 임시 env로 상태/행동 차원 파악
temp_env_fn = make_env(**ENV_PARAMS)
temp_env = temp_env_fn()
sample_obs, _ = temp_env.reset()
state_dim = len(flatten_dict_values(sample_obs))
action_dim = temp_env.action_space.n
temp_env.close()

# 평가용 환경 파라미터 리스트 (원본 로직 유지)
env_param_list = []
for _ in range(n_workers):
    e = copy.deepcopy(ENV_PARAMS)
    e["max_comp_units"] = np.random.randint(40, 161)
    # e["agent_velocities"] = np.random.randint(30, 101)
    env_param_list.append(e)

# @torch.no_grad()
def evaluate_model_on_env(model_path, env_kwargs, n_episodes=100, greedy=True, render=False, log_actions=False, log_prefix=""):
    """단일 환경에서 모델 평가 및 액션 로깅"""
    model = RecurrentActorCritic(state_dim, action_dim, hidden_dim).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    env_fn = make_env(**env_kwargs, reward_params=REWARD_PARAMS)
    env = env_fn()

    episode_rewards, episode_lengths = [], []
    
    # 액션 로깅용 데이터
    action_logs = []

    for ep_idx in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        
        hx = model.init_hidden(batch_size=1, device=device)

        while not done and steps < env.max_epoch_size:
            obs_tensor = torch.as_tensor(
                flatten_dict_values(obs), dtype=torch.float32, device=device
            ).unsqueeze(0)

            logits, value, hx = model.step(obs_tensor, hx)

            if greedy:
                action = torch.argmax(logits, dim=-1).item()
            else:
                dist = Categorical(logits=logits)
                action = dist.sample().item()

            # 액션 로깅 - 자동으로 모든 observation 처리
            if log_actions:
                log_entry = {
                    'episode': ep_idx,
                    'step': steps,
                    'action': action,
                    'distribution': logits[0].softmax(dim=-1).cpu().tolist(),
                }
                
                # obs의 모든 키-값을 자동으로 추가
                for key, value in obs.items():
                    if isinstance(value, np.ndarray):
                        # 배열인 경우 요약 정보만
                        log_entry[f'{key}_sum'] = float(np.sum(value))
                        log_entry[f'{key}_nonzero'] = int(np.count_nonzero(value))
                    elif hasattr(value, 'item'):
                        log_entry[key] = value.item()
                    else:
                        log_entry[key] = float(value) if not isinstance(value, int) else value
                
                # 환경 파라미터 추가
                log_entry['reward'] = 0  # 다음 스텝에서 업데이트
                log_entry['env_max_comp_units'] = env_kwargs.get('max_comp_units', 'N/A')
                log_entry['env_agent_velocities'] = env_kwargs.get('agent_velocities', 'N/A')
                
                action_logs.append(log_entry)


            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 이전 스텝의 reward 업데이트
            if log_actions and action_logs:
                action_logs[-1]['reward'] = reward

            total_reward += float(reward)
            steps += 1
            obs = next_obs

            if done:
                hx = hx * 0.0

            if render:
                env.render()

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

    env.close()

    # CSV 저장
    if log_actions and action_logs:
        import csv
        csv_filename = f"{log_prefix}_actions.csv"
        
        # 첫 번째 로그 엔트리에서 필드명 자동 추출
        if action_logs:
            fieldnames = list(action_logs[0].keys())
            
            with open(csv_filename, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(action_logs)
            print(f"  Saved action logs to {csv_filename}")

    return {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "min_reward": float(np.min(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
        "rewards": episode_rewards,
    }


def evaluate_over_envs(model_path, env_param_list, n_episodes=100, greedy=True):
    """여러 환경 파라미터에서 같은 모델을 평가하고 평균"""
    per_env = []
    for idx, e in enumerate(env_param_list):
        print(f"  - evaluating on env {idx+1}/{len(env_param_list)} ...")
        res = evaluate_model_on_env(model_path, e, n_episodes=n_episodes, greedy=greedy)
        res["env_id"] = idx
        per_env.append(res)

    # 환경 간 평균(요약)
    all_rewards = np.concatenate([np.array(r["rewards"]) for r in per_env]) if per_env else np.array([0.0])
    summary = {
        "mean_reward": float(np.mean([r["mean_reward"] for r in per_env])) if per_env else 0.0,
        "std_reward": float(np.std(all_rewards)) if per_env else 0.0,
        "min_reward": float(np.min(all_rewards)) if per_env else 0.0,
        "max_reward": float(np.max(all_rewards)) if per_env else 0.0,
        "mean_length": float(np.mean([r["mean_length"] for r in per_env])) if per_env else 0.0,
        "per_env": per_env,
    }
    return summary


def compare_all_models(env_param_list, n_episodes=100, greedy=True):
    """여러 모델(pth)들을 동일한 env 세트에서 비교"""
    results = {}

    # A3C 글로벌 모델
    gpath = f"runs/a3c_{stamp}/models/global_final.pth"
    if os.path.exists(gpath):
        print("Evaluating A3C Global Model...")
        results["A3C_Global"] = evaluate_over_envs(gpath, env_param_list, n_episodes, greedy)

    # 개별(Individual) 모델들
    individual_results = []
    for i in range(params.n_workers):
        mpath = f"runs/individual_{stamp}/models/individual_worker_{i}_final.pth"
        if os.path.exists(mpath):
            print(f"Evaluating Individual Worker {i}...")
            r = evaluate_over_envs(mpath, env_param_list, n_episodes, greedy)
            r["worker_id"] = i
            individual_results.append(r)
    results["Individual_Workers"] = individual_results

    return results


def print_results(results):
    print("\n" + "=" * 60)
    print("MODEL EVALUATION RESULTS")
    print("=" * 60)

    # A3C Global
    if "A3C_Global" in results:
        r = results["A3C_Global"]
        print(f"\n📊 A3C Global Model (avg over envs):")
        print(f"  Mean Reward: {r['mean_reward']:.2f} (std over all eps: {r['std_reward']:.2f})")
        print(f"  Range: [{r['min_reward']:.2f}, {r['max_reward']:.2f}]")
        print(f"  Mean Episode Length: {r['mean_length']:.1f}")

    # Individual Workers
    if "Individual_Workers" in results and results["Individual_Workers"]:
        print(f"\n📊 Individual Workers (avg over envs):")
        all_individual_means = []
        for r in results["Individual_Workers"]:
            wid = r["worker_id"]
            print(f"  Worker {wid}: mean {r['mean_reward']:.2f}")
            all_individual_means.append(r["mean_reward"])

        if all_individual_means:
            print(f"\n  📈 Individual Workers Overall:")
            print(f"    Mean of means: {np.mean(all_individual_means):.2f}")
            print(f"    Std of means : {np.std(all_individual_means):.2f}")

    # 비교
    if "A3C_Global" in results and results["Individual_Workers"]:
        a3c_mean = results["A3C_Global"]["mean_reward"]
        individual_means = [r["mean_reward"] for r in results["Individual_Workers"]]
        best_ind = max(individual_means) if individual_means else 0.0
        avg_ind = float(np.mean(individual_means)) if individual_means else 0.0

        print(f"\n🏆 COMPARISON (avg over envs):")
        print(f"  A3C Global vs Best Individual: {a3c_mean:.2f} vs {best_ind:.2f}")
        print(f"  A3C Global vs Avg Individual : {a3c_mean:.2f} vs {avg_ind:.2f}")
        improvement = ((a3c_mean - avg_ind) / (abs(avg_ind) + 1e-8)) * 100.0 if avg_ind != 0 else 0.0
        print(f"  A3C Improvement: {improvement:+.1f}%")

def save_detailed_results(results, filename="evaluation_results.csv"):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Mean_Reward", "Std_Reward", "Min_Reward", "Max_Reward", "Mean_Length"])

        if "A3C_Global" in results:
            r = results["A3C_Global"]
            writer.writerow(["A3C_Global", r["mean_reward"], r["std_reward"],
                             r["min_reward"], r["max_reward"], r["mean_length"]])

        if "Individual_Workers" in results:
            for r in results["Individual_Workers"]:
                wid = r["worker_id"]
                writer.writerow([f"Individual_Worker_{wid}", r["mean_reward"], r["std_reward"],
                                 r["min_reward"], r["max_reward"], r["mean_length"]])

if __name__ == "__main__":
    # print("Starting model evaluation...")
    results = compare_all_models(env_param_list, n_episodes=100, greedy=False)
    print_results(results)
    save_detailed_results(results)
    print("\nDetailed results saved to evaluation_results.csv")
    
    print("Starting model evaluation with action logging...")
    # 적은 수의 에피소드로 액션 로깅
    n_episodes_for_logging = 10  # 액션 로깅용 에피소드 수
    
    # A3C Global 모델 평가 및 액션 로깅
    gpath = f"runs/a3c_{stamp}/models/global_final.pth"
    if os.path.exists(gpath):
        print("Evaluating A3C Global Model with action logging...")
        for env_idx, env_params in enumerate(env_param_list):  # 모든 환경
            print(f"  Environment {env_idx}...")
            evaluate_model_on_env(
                gpath, 
                env_params, 
                n_episodes=n_episodes_for_logging, 
                greedy=False,
                log_actions=True,
                log_prefix=f"a3c_global_env{env_idx}"
            )
    else:
        print(f"❌ A3C Global model not found: {gpath}")
    
    # Individual 모델들 평가 및 액션 로깅
    for worker_id in range(n_workers):  # 모든 워커
        mpath = f"runs/individual_{stamp}/models/individual_worker_{worker_id}_final.pth"
        if os.path.exists(mpath):
            print(f"Evaluating Individual Worker {worker_id} with action logging...")
            for env_idx, env_params in enumerate(env_param_list):  # 모든 환경
                print(f"  Environment {env_idx}...")
                evaluate_model_on_env(
                    mpath,
                    env_params,
                    n_episodes=n_episodes_for_logging,
                    greedy=False,
                    log_actions=True,
                    log_prefix=f"individual_w{worker_id}_env{env_idx}"
                )
        else:
            print(f"❌ Individual Worker {worker_id} model not found: {mpath}")