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
stamp = "20250820_141900"  # 예시 타임스탬프, 필요에 따라 변경

def _deep_copy_obs(obs: Dict[str, Any]) -> Dict[str, Any]:
    """딕셔너리 형태의 관측값을 깊은 복사하여 반환"""
    out = {}
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            out[k] = v.copy()
        else:
            out[k] = copy.deepcopy(v)
    return out

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

@torch.no_grad()
def evaluate_model_on_env(model_path, env_kwargs, n_episodes=100, greedy=True, render=False,
                          input_transform: Callable[[Dict[str, Any]], Dict[str, Any]] = None):
    """
    단일 환경에서 pth 모델 평가.
    input_transform(obs_dict) -> obs_dict' 를 주면 '에이전트 입력'만 교란(퍼뮤테이션/고정 등)해 평가.
    환경 내부 상태는 그대로 유지되어 안정적이다.
    """
    # 모델 로드 (RNN)
    model = RecurrentActorCritic(state_dim, action_dim, hidden_dim).to(device)
    sd = torch.load(model_path, map_location=device)
    try:
        model.load_state_dict(sd, strict=True)
    except Exception as e:
        # 호환 로더로 재시도 (shared.* → feature.* 등)
        load_weights_compat(model, sd, rename_rules=[("shared.", "feature.")], verbose=True)
    model.eval()

    # 환경
    env_fn = make_env(**env_kwargs, reward_params=REWARD_PARAMS)
    env = env_fn()

    episode_rewards, episode_lengths = [], []

    for _ in range(n_episodes):
        if input_transform is not None and hasattr(input_transform, "reset"):
            input_transform.reset()  # ← 에피소드 시작마다 도너 시퀀스 리셋
        
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        hx = model.init_hidden(batch_size=1, device=device)

        while not done and steps < env.max_epoch_size:
            # --- 관찰 dict -> (선택) 입력 교란 -> 텐서 ---
            obs_for_model = _deep_copy_obs(obs)
            if input_transform is not None:
                obs_for_model = input_transform(obs_for_model)

            obs_tensor = torch.as_tensor(
                flatten_dict_values(obs_for_model), dtype=torch.float32, device=device
            ).unsqueeze(0)  # (1, state_dim)

            logits, value, hx = model.step(obs_tensor, hx)
            if greedy:
                action = torch.argmax(logits, dim=-1).item()
            else:
                action = Categorical(logits=logits).sample().item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            total_reward += float(reward)
            steps += 1
            obs = next_obs

            if render:
                env.render()

            if done:
                hx = hx * 0.0

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

    env.close()

    return {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "min_reward": float(np.min(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
        "rewards": episode_rewards,
    }

def collect_feature_buffers(model_path, env_kwargs, keys=None, n_episodes=30, greedy=True,
                            max_per_key=5000, seed=0):
    """
    퍼뮤테이션 임포턴스용으로, 에이전트가 실제로 보게 되는 관찰값 분포를 키별로 수집.
    - 환경 상태는 그대로, '에이전트 입력'만 바꿔 평가하므로 shape/type mismatch 위험이 적다.
    """
    rng = np.random.default_rng(seed)

    # 모델 로드 (평가 전파만)
    model = RecurrentActorCritic(state_dim, action_dim, hidden_dim).to(device)
    sd = torch.load(model_path, map_location=device)
    try:
        model.load_state_dict(sd)
    except Exception as e:
        # 호환 로더로 재시도 (shared.* → feature.* 등 키 리매핑)
        load_weights_compat(
            model,
            sd,  # 또는 model_path (문자열). 둘 다 지원
            rename_rules=[("shared.", "feature.")],
            verbose=True,
        )
    model.eval()

    env_fn = make_env(**env_kwargs, reward_params=REWARD_PARAMS)
    env = env_fn()

    buffers: Dict[str, list] = {}
    episodes = 0
    with torch.no_grad():
        while episodes < n_episodes:
            obs, _ = env.reset()
            done, steps = False, 0
            hx = model.init_hidden(batch_size=1, device=device)

            # 키 초기화
            if keys is None:
                keys = list(obs.keys())

            while not done and steps < env.max_epoch_size:
                # 관찰값들을 버퍼에 저장
                for k in keys:
                    v = obs[k]
                    if k not in buffers:
                        buffers[k] = []
                    # 용량 제한
                    if len(buffers[k]) < max_per_key:
                        buffers[k].append(v.copy() if isinstance(v, np.ndarray) else v)

                # 행동
                x = torch.as_tensor(flatten_dict_values(obs), dtype=torch.float32, device=device).unsqueeze(0)
                logits, _, hx = model.step(x, hx)
                if greedy:
                    a = torch.argmax(logits, dim=-1).item()
                else:
                    a = Categorical(logits=logits).sample().item()

                obs, r, terminated, truncated, _ = env.step(a)
                done = terminated or truncated
                steps += 1

            episodes += 1

    env.close()

    # numpy 배열로 바꿔 두면 추후 샘플링이 빠름
    for k in buffers:
        buffers[k] = np.array(buffers[k], dtype=object)  # 다양한 shape을 담기 위해 object 배열
    return buffers

def collect_feature_buffers_ep(model_path, env_kwargs, keys=None, n_episodes=30, greedy=True, seed=0):
    """
    퍼뮤테이션을 '에피소드 단위'로 하기 위해, 각 key의 '시퀀스(에피소드 전체)'를 모읍니다.
    반환: {key: [seq0, seq1, ...]}, seq는 리스트(스텝 길이만큼)의 값들
    """
    rng = np.random.default_rng(seed)

    # 모델 로드
    model = RecurrentActorCritic(state_dim, action_dim, hidden_dim).to(device)
    sd = torch.load(model_path, map_location=device)
    try:
        model.load_state_dict(sd, strict=True)
    except Exception:
        load_weights_compat(model, sd, rename_rules=[("shared.", "feature.")], verbose=False)
    model.eval()

    env_fn = make_env(**env_kwargs, reward_params=REWARD_PARAMS)
    env = env_fn()

    buffers_ep = {}
    episodes = 0
    with torch.no_grad():
        while episodes < n_episodes:
            obs, _ = env.reset()
            done, steps = False, 0
            hx = model.init_hidden(batch_size=1, device=device)

            # 키 초기화
            if keys is None:
                keys = list(obs.keys())
            seq = {k: [] for k in keys}

            while not done and steps < env.max_epoch_size:
                for k in keys:
                    v = obs[k]
                    seq[k].append(v.copy() if isinstance(v, np.ndarray) else v)

                x = torch.as_tensor(flatten_dict_values(obs), dtype=torch.float32, device=device).unsqueeze(0)
                logits, _, hx = model.step(x, hx)
                a = torch.argmax(logits, dim=-1).item() if greedy else Categorical(logits=logits).sample().item()

                obs, r, term, trunc, _ = env.step(a)
                done = term or trunc
                steps += 1

            for k in keys:
                buffers_ep.setdefault(k, []).append(seq[k])
            episodes += 1

    env.close()
    return buffers_ep


# ====== [추가] 단일 키 퍼뮤테이션 변환기 만들기 ======

def make_permute_transform(buffers: Dict[str, np.ndarray], key: str, seed: int = 0):
    """
    주어진 key만 다른 시점의 값으로 무작위 치환하는 transform(obs)->obs'.
    - obs dict를 받아 같은 타입/shape의 값으로 교체.
    """
    rng = np.random.default_rng(seed)

    if key not in buffers or len(buffers[key]) == 0:
        raise ValueError(f"[permute] buffer for key '{key}' is empty.")

    def transform(obs: Dict[str, Any]) -> Dict[str, Any]:
        out = obs  # 이미 상위에서 deepcopy됨
        cand = buffers[key][rng.integers(len(buffers[key]))]
        # dtype/shape 맞춤
        if isinstance(out[key], np.ndarray):
            cand = np.asarray(cand)
            if cand.shape != out[key].shape:
                # shape이 다르면 가능한 범위에서 잘라내거나 pad하지 말고, 원래 값 유지
                # (서로 다른 환경 구성에서 shape가 다를 수 있으므로 안전장치)
                return out
            out[key] = cand.astype(out[key].dtype, copy=False)
        else:
            # 스칼라/파이썬 타입
            out[key] = type(out[key])(cand)
        return out

    return transform


# ====== [추가] 퍼뮤테이션 임포턴스 (단일 환경) ======

def permutation_importance_on_env(model_path, env_kwargs, keys=None,
                                  collect_episodes=30, eval_episodes=80,
                                  greedy=True, seed=0):
    """
    단일 환경에서 퍼뮤테이션 임포턴스 계산.
    반환: {'baseline': float, 'drops': {key: drop}, 'scores': {key: mean_reward_when_permuted}}
    """
    # 0) baseline
    base = evaluate_model_on_env(model_path, env_kwargs, n_episodes=eval_episodes, greedy=greedy)
    baseline = base["mean_reward"]

    # 1) buffers 수집
    buffers = collect_feature_buffers(model_path, env_kwargs, keys=keys,
                                      n_episodes=collect_episodes, greedy=greedy, seed=seed)

    # 2) 키 목록
    if keys is None:
        keys = list(buffers.keys())

    # 3) 키별 성능 측정
    drops, scores = {}, {}
    for k in keys:
        transform = make_permute_transform(buffers, k, seed=seed+123)
        res = evaluate_model_on_env(model_path, env_kwargs, n_episodes=eval_episodes,
                                    greedy=greedy, input_transform=transform)
        score = res["mean_reward"]
        drops[k] = baseline - score
        scores[k] = score

    return {"baseline": baseline, "drops": drops, "scores": scores}


# ====== [추가] 여러 환경 평균 퍼뮤테이션 임포턴스 ======

def permutation_importance_over_envs(model_path, env_param_list, keys=None,
                                     collect_episodes=20, eval_episodes=50,
                                     greedy=True, max_envs=None, seed=0):
    """
    여러 환경 파라미터에 대해 퍼뮤테이션 임포턴스를 계산하고 평균/표준편차를 리턴.
    """
    if max_envs is None:
        max_envs = len(env_param_list)

    all_keys = None
    baselines, per_key_scores = [], {}

    for idx, env_kwargs in enumerate(env_param_list[:max_envs]):
        out = permutation_importance_on_env(model_path, env_kwargs, keys=keys,
                                            collect_episodes=collect_episodes,
                                            eval_episodes=eval_episodes,
                                            greedy=greedy, seed=seed + idx*7)
        baselines.append(out["baseline"])

        if all_keys is None:
            all_keys = list(out["drops"].keys())

        for k in all_keys:
            per_key_scores.setdefault(k, []).append(out["drops"][k])

    # 평균/표준편차 요약
    mean_drop = {k: float(np.mean(per_key_scores[k])) for k in per_key_scores}
    std_drop  = {k: float(np.std(per_key_scores[k]))  for k in per_key_scores}

    summary = {
        "baseline_mean": float(np.mean(baselines)),
        "baseline_std":  float(np.std(baselines)),
        "mean_drop": mean_drop,
        "std_drop": std_drop,
        "sorted": sorted(mean_drop.items(), key=lambda x: x[1], reverse=True),
    }
    return summary

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

def load_weights_compat(model, raw_state_dict, *, 
                        rename_rules=None, strip_prefix="module.", verbose=True):
    """
    Load a checkpoint into `model` tolerantly:
      - supports {'state_dict': ...} wrappers
      - strips 'module.' (DataParallel) prefix
      - applies simple rename rules, e.g. [('shared.', 'feature.')]
      - loads only keys that exist in the model with matching shapes (strict=False)
    Returns a dict with summary counts.
    """
    # 1) extract actual state_dict
    sd = raw_state_dict
    if isinstance(sd, str):
        sd = torch.load(sd, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]

    # 2) strip prefix
    def strip(k):
        return k[len(strip_prefix):] if strip_prefix and k.startswith(strip_prefix) else k

    # 3) rename rules (default: map 'shared.' → 'feature.')
    if rename_rules is None:
        rename_rules = [("shared.", "feature.")]
    def apply_rules(k):
        k2 = k
        for src, dst in rename_rules:
            if k2.startswith(src):
                k2 = dst + k2[len(src):]
        return k2

    remapped = {}
    for k, v in sd.items():
        k2 = apply_rules(strip(k))
        remapped[k2] = v

    # 4) keep only intersecting keys with matching shapes
    msd = model.state_dict()
    loadable = {k: v for k, v in remapped.items() if (k in msd and msd[k].shape == v.shape)}
    missing = [k for k in msd.keys() if k not in loadable]
    unexpected = [k for k in remapped.keys() if k not in msd]
    shape_mismatch = [k for k in remapped.keys() if (k in msd and msd[k].shape != remapped[k].shape)]

    # 5) load (non-strict)
    model.load_state_dict(loadable, strict=False)

    if verbose:
        print("[load_weights_compat] loaded:", len(loadable),
              "| missing:", len(missing),
              "| unexpected:", len(unexpected),
              "| shape_mismatch:", len(shape_mismatch))
        # 가장 흔한 케이스 안내 (shared→feature)
        if any(k.startswith("shared.") for k in sd.keys()):
            print("  note: remapped 'shared.*' → 'feature.*' for RecurrentActorCritic.")

    return {
        "loaded": list(loadable.keys()),
        "missing": missing,
        "unexpected": unexpected,
        "shape_mismatch": shape_mismatch,
    }

def permutation_importance_groups_on_env(model_path, env_kwargs, groups,
                                         collect_episodes=30, eval_episodes=200,
                                         greedy=True, seed=0):
    """
    그룹 단위(구조 보존) 퍼뮤테이션 임포턴스.
    """
    # baseline
    base = evaluate_model_on_env(model_path, env_kwargs, n_episodes=eval_episodes, greedy=greedy)
    baseline = base["mean_reward"]

    # 에피소드-버퍼 수집
    buffers_ep = collect_feature_buffers_ep(model_path, env_kwargs, keys=list(set(sum([list(g) for g in groups], []))),
                                            n_episodes=collect_episodes, greedy=greedy, seed=seed)

    drops = {}
    for g in groups:
        transform = EpisodeGroupPermuter(buffers_ep, g, seed=seed+7)
        res = evaluate_model_on_env(model_path, env_kwargs, n_episodes=eval_episodes,
                                    greedy=greedy, input_transform=transform)
        drops[g] = baseline - res["mean_reward"]

    return baseline, drops


def occlusion_importance_on_env(model_path, env_kwargs, groups_or_keys,
                                how="mean", collect_episodes=30, eval_episodes=200,
                                greedy=True, seed=0):
    """
    그룹/키를 상수로 고정했을 때 Δreward 측정.
    how: "mean" 또는 "zero"
    """
    # baseline
    base = evaluate_model_on_env(model_path, env_kwargs, n_episodes=eval_episodes, greedy=greedy)
    baseline = base["mean_reward"]

    # 평균 상수 만들기
    buffers_ep = collect_feature_buffers_ep(model_path, env_kwargs, keys=list(set(sum([list(g) if isinstance(g, (list, tuple)) else [g] for g in groups_or_keys], []))),
                                            n_episodes=collect_episodes, greedy=greedy, seed=seed)

    # 에피소드 버퍼를 평탄화해서 평균을 계산
    const_dict = {}
    for g in groups_or_keys:
        keys = g if isinstance(g, (list, tuple)) else (g,)
        for k in keys:
            seqs = buffers_ep[k]
            flat = []
            for s in seqs: flat += s
            v0 = flat[0]
            if isinstance(v0, np.ndarray):
                arr = np.stack([np.asarray(v) for v in flat], axis=0)
                const = (arr.mean(axis=0) if how=="mean" else np.zeros_like(arr.mean(axis=0)))
            else:
                vals = np.array(flat, dtype=np.float32)
                const = (float(vals.mean()) if how=="mean" else 0.0)
            const_dict[k] = const

    transform = KeyConstantOccluder(const_dict)
    res = evaluate_model_on_env(model_path, env_kwargs, n_episodes=eval_episodes,
                                greedy=greedy, input_transform=transform)
    return baseline, baseline - res["mean_reward"]


class EpisodeGroupPermuter:
    """
    지정된 group_keys를 '하나의 도너 에피소드' 시퀀스로 치환.
    evaluate_model_on_env()가 매 에피소드 시작 시 reset()을 호출해야 함.
    """
    def __init__(self, buffers_ep: dict, group_keys, seed=0):
        self.buffers_ep = buffers_ep
        self.group_keys = tuple(group_keys)
        self.rng = np.random.default_rng(seed)
        self.donor_idx = None
        self.t = 0

    def reset(self):
        # 도너 에피소드 하나를 뽑고, 스텝 인덱스 0으로 리셋
        any_key = self.group_keys[0]
        n_eps = len(self.buffers_ep[any_key])
        self.donor_idx = int(self.rng.integers(n_eps))
        self.t = 0

    def __call__(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        out = _deep_copy_obs(obs)
        if self.donor_idx is None:
            self.reset()
        for k in self.group_keys:
            seq = self.buffers_ep[k][self.donor_idx]
            val = seq[min(self.t, len(seq)-1)]
            if isinstance(out[k], np.ndarray):
                out[k] = np.asarray(val, dtype=out[k].dtype)
            else:
                out[k] = type(out[k])(val)
        self.t += 1
        return out

class KeyConstantOccluder:
    """
    특정 key(또는 그룹)를 상수로 고정(평균/0 등). 그룹이면 같은 값으로 모두 고정.
    """
    def __init__(self, const_dict: Dict[str, Any]):
        self.const_dict = const_dict

    def reset(self):  # evaluate에서 호출해도 무해
        pass

    def __call__(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        out = _deep_copy_obs(obs)
        for k, const in self.const_dict.items():
            v = out[k]
            out[k] = (np.asarray(const, dtype=v.dtype)
                      if isinstance(v, np.ndarray) else type(v)(const))
        return out


# ----------------------------
# Policy dataset & bin stats
# ----------------------------
import numpy as np
import torch
from torch.distributions import Categorical

@torch.no_grad()
def collect_policy_dataset(model_path, env_kwargs, n_episodes=200, greedy=True):
    """
    정책이 실제로 내리는 행동/확률과 관찰을 수집.
    rows[i] = { 'obs': <obs_dict>, 'p1': P(a=1), 'a': action(int) }
    """
    # 모델 로드
    model = RecurrentActorCritic(state_dim, action_dim, hidden_dim).to(device)
    sd = torch.load(model_path, map_location=device)
    try:
        model.load_state_dict(sd, strict=True)
    except Exception:
        load_weights_compat(model, sd, rename_rules=[("shared.", "feature.")], verbose=False)
    model.eval()

    # 환경
    env = make_env(**env_kwargs, reward_params=REWARD_PARAMS)()
    rows = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        hx = model.init_hidden(batch_size=1, device=device)
        t = 0
        while not done and t < env.max_epoch_size:
            x = torch.as_tensor(
                flatten_dict_values(obs), dtype=torch.float32, device=device
            ).unsqueeze(0)
            logits, _, hx = model.step(x, hx)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            if greedy:
                a = int(np.argmax(probs))
            else:
                a = int(Categorical(logits=logits).sample().item())
            rows.append({
                "obs": {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in obs.items()},
                "p1": float(probs[1]),
                "a": a,
            })
            obs, r, term, trunc, _ = env.step(a)
            done = term or trunc
            t += 1

    env.close()
    return rows


def _scalarize(value, how="first"):
    """벡터 피처를 binning에 쓰기 위한 스칼라 요약."""
    arr = np.asarray(value).reshape(-1)
    if how == "first": return float(arr[0])
    if how == "mean":  return float(arr.mean())
    if how == "max":   return float(arr.max())
    if how == "min":   return float(arr.min())
    return float(arr[0])


def bin_stat(rows, key, n_bins=10, how="first"):
    """
    p(a=1 | feature-bin) 계산.
    - key: 관찰 딕셔너리의 키
    - how: 벡터일 경우 사용할 요약 방식 ('first'|'mean'|'max'|'min')
    반환: mids(빈 중앙), p_mean(각 bin에서 p1 평균), cnt(샘플 수)
    """
    vals = np.array([_scalarize(r["obs"][key], how=how) for r in rows], dtype=np.float32)
    p1   = np.array([r["p1"] for r in rows], dtype=np.float32)

    # 값이 [0,1] 범위를 벗어나면 자동 min-max 정규화 (안전장치)
    vmin, vmax = np.nanmin(vals), np.nanmax(vals)
    if vmax > 1.0 or vmin < 0.0:
        rng = (vmax - vmin) if (vmax > vmin) else 1.0
        vals = (vals - vmin) / rng

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    mids, p_mean, cnt = [], [], []
    for i in range(n_bins):
        m = (vals >= bins[i]) & (vals < bins[i+1])
        mids.append((bins[i] + bins[i+1]) / 2.0)
        if m.sum() == 0:
            p_mean.append(np.nan); cnt.append(0)
        else:
            p_mean.append(float(p1[m].mean()))
            cnt.append(int(m.sum()))
    return np.array(mids), np.array(p_mean), np.array(cnt)

# ----------------------------
# Gradient-based importance
# ----------------------------

def _key_slices_from_example_obs(obs):
    """
    flatten_dict_values의 '키 순서 그대로' 슬라이스 인덱스 계산.
    (Python 3.7+ dict는 입력 순서 보장)
    """
    key_slices = {}
    idx = 0
    for k, v in obs.items():
        L = int(np.asarray(v).size)
        key_slices[k] = slice(idx, idx + L)
        idx += L
    return key_slices, idx  # 전체 state_dim


def gradient_importance(model_path, env_kwargs, sample_steps=2048, greedy=True):
    """
    로짓 차 (logit1 - logit0)의 입력 민감도 |d(logit diff)/dx| 평균을 키별로 집계.
    값이 클수록 현재 시점의 의사결정에 민감.
    """
    # 모델 로드
    model = RecurrentActorCritic(state_dim, action_dim, hidden_dim).to(device)
    sd = torch.load(model_path, map_location=device)
    try:
        model.load_state_dict(sd, strict=True)
    except Exception:
        load_weights_compat(model, sd, rename_rules=[("shared.", "feature.")], verbose=False)
    model.eval()

    # 환경
    env = make_env(**env_kwargs, reward_params=REWARD_PARAMS)()

    # 샘플 수집 (x, hx) 페어
    xs, hxs = [], []
    obs, _ = env.reset()
    hx = model.init_hidden(batch_size=1, device=device)
    key_slices, flat_dim = _key_slices_from_example_obs(obs)

    with torch.no_grad():
        steps = 0
        done = False
        while steps < sample_steps:
            x = torch.as_tensor(
                flatten_dict_values(obs), dtype=torch.float32, device=device
            ).unsqueeze(0)  # (1, state_dim)
            logits, _, hx = model.step(x, hx)
            a = int(torch.argmax(logits, dim=-1).item()) if greedy else int(Categorical(logits=logits).sample().item())
            xs.append(x.squeeze(0).detach())  # (state_dim,)
            hxs.append(hx.detach())            # (1,1,H)
            obs, r, term, trunc, _ = env.step(a)
            done = term or trunc
            steps += 1
            if done:
                obs, _ = env.reset()
                hx = model.init_hidden(batch_size=1, device=device)
                done = False

    # 샘플별로 개별 backprop (RNN hidden을 고정 상수로 보고 현재 시점 민감도만 계산)
    grad_accum = torch.zeros(flat_dim, dtype=torch.float32, device=device)
    for x, h in zip(xs, hxs):
        x = x.clone().detach().requires_grad_(True)          # (state_dim,)
        logits, _, _ = model.step(x.unsqueeze(0), h)         # (1,2)
        score = (logits[0,1] - logits[0,0])                  # logit_diff
        score.backward()
        grad_accum += x.grad.detach().abs().squeeze(0)

    grad_mean = (grad_accum / max(1, len(xs))).cpu().numpy()  # (state_dim,)

    # 키별로 집계
    per_key = {}
    for k, sl in key_slices.items():
        per_key[k] = float(grad_mean[sl].mean())
    env.close()

    # 큰 순서로 정렬해 리턴
    return dict(sorted(per_key.items(), key=lambda x: x[1], reverse=True))

# ----------------------------
# Sparse surrogate (L1 logistic)
# ----------------------------
import torch.nn as nn
import torch.nn.functional as F

def fit_sparse_surrogate(rows, l1=5e-3, epochs=2000, lr=1e-2, standardize=True):
    """
    정책의 (obs -> action) 매핑을 L1 로지스틱으로 근사하여
    피처 중요도(부호 포함)를 뽑는다.
    반환: {'per_key': {key: weight_avg}, 'acc': float, 'keys': [order]}
    """
    # 피처 순서(딕셔너리 첫 샘플의 키 순서)
    keys = list(rows[0]["obs"].keys())

    # X, y 구성
    feats, labels = [], []
    for r in rows:
        vec = []
        for k in keys:
            v = np.asarray(r["obs"][k]).reshape(-1)
            vec.extend(v.tolist())
        feats.append(vec)
        labels.append(r["a"])
    X = torch.tensor(np.array(feats, dtype=np.float32))
    y = torch.tensor(np.array(labels, dtype=np.float32)).unsqueeze(1)

    # 표준화 (선택)
    if standardize:
        mean = X.mean(dim=0, keepdim=True)
        std = X.std(dim=0, keepdim=True).clamp_min(1e-6)
        Xn = (X - mean) / std
    else:
        Xn = X

    clf = nn.Linear(Xn.shape[1], 1)
    opt = torch.optim.Adam(clf.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        opt.zero_grad()
        logits = clf(Xn)
        loss = bce(logits, y) + l1 * clf.weight.abs().sum()
        loss.backward()
        opt.step()

    with torch.no_grad():
        logits = clf(Xn)
        pred = (torch.sigmoid(logits) > 0.5).float()
        acc = float((pred.eq(y)).float().mean().item())

    w = clf.weight.detach().cpu().numpy()[0]  # (state_dim,)

    # 키별 평균 가중치(벡터 키는 평균/합 중 택1; 여기선 평균)
    per_key = {}
    idx = 0
    for k in keys:
        L = int(np.asarray(rows[0]["obs"][k]).size)
        per_key[k] = float(np.mean(w[idx:idx+L]))
        idx += L

    # 절댓값 큰 순으로 정렬한 dict
    per_key_sorted = dict(sorted(per_key.items(), key=lambda x: abs(x[1]), reverse=True))
    return {"per_key": per_key_sorted, "acc": acc, "keys": keys}

def print_bin_report(rows, keys_binspec):
    """
    keys_binspec: [(key, n_bins, how), ...]
    """
    for key, n_bins, how in keys_binspec:
        mids, p_mean, cnt = bin_stat(rows, key, n_bins=n_bins, how=how)
        print(f"\n[bin] key='{key}' (how={how})")
        for m, p, c in zip(mids, p_mean, cnt):
            if np.isnan(p): continue
            print(f"  bin@{m: .2f}: p(a=1)={p: .3f}  (n={int(c)})")



if __name__ == "__main__":
    print("Starting model evaluation...")
    results = compare_all_models(env_param_list, n_episodes=100, greedy=True)
    print_results(results)
    save_detailed_results(results)
    print("\nDetailed results saved to evaluation_results.csv")

    print("Starting permutation-importance analysis ...")
    # 대표 모델 하나에 대해, env 리스트 평균 임포턴스 계산
    target_model = "models/global_final.pth"  # 필요 시 변경
    if os.path.exists(target_model):
        imp = permutation_importance_over_envs(
            model_path=target_model,
            env_param_list=env_param_list,
            keys=None,                  # None이면 관찰 전 키 전체
            collect_episodes=20,        # 수집(버퍼) 에피소드
            eval_episodes=50,           # 평가 에피소드
            greedy=True,
            max_envs=min(5, len(env_param_list)),  # 시간 단축용
            seed=0,
        )

        print("\n=== Permutation Importance (avg over envs) ===")
        print(f"Baseline mean reward: {imp['baseline_mean']:.2f} ± {imp['baseline_std']:.2f}")
        for k, d in imp["sorted"]:
            print(f"  {k:>20s} : Δreward = {d:.2f} (std {imp['std_drop'][k]:.2f})")
    else:
        print(f"Model not found: {target_model}")

    groups = [
        ("mec_comp_units", "mec_proc_times"),
        ("queue_comp_units", "queue_proc_times"),
        ("ctx_vel", "ctx_comp"),
        ("available_computation_units",),
        ("channel_quality",),
    ]

    # target_model = f"runs/a3c_{stamp}/models/global_final.pth"  # 경로는 환경에 맞게
    target_model = f"runs/individual_{stamp}/models/individual_worker_0_final.pth"  # 경로는 환경에 맞게
    if os.path.exists(target_model):
        print("\n=== Grouped Permutation Importance ===")
        base_list, group_drops = [], {}
        for idx, env_kwargs in enumerate(env_param_list[:min(5, len(env_param_list))]):
            b, d = permutation_importance_groups_on_env(
                target_model, env_kwargs, groups,
                collect_episodes=20, eval_episodes=200, greedy=True, seed=idx*11
            )
            base_list.append(b)
            for g, v in d.items():
                group_drops.setdefault(g, []).append(v)

        print(f"Baseline mean: {np.mean(base_list):.2f} ± {np.std(base_list):.2f}")
        for g in groups:
            vals = group_drops.get(g, [])
            if not vals: continue
            print(f"{str(g):>40s} : Δreward = {np.mean(vals):.2f} (std {np.std(vals):.2f})")

        print("\n=== Occlusion (mean-constant) ===")
        b, d = occlusion_importance_on_env(
            target_model, env_param_list[0], groups, how="mean",
            collect_episodes=20, eval_episodes=200, greedy=True, seed=0
        )
        print(f"Baseline: {b:.2f}, Δreward(mean-const, grouped) = {d:.2f}")

    # 분석 타깃 모델과 환경 하나 고르기
    # target_model = "models/global_final.pth"     # 경로 맞게 조정
    env_kwargs   = env_param_list[0]             # 대표 환경 하나

    if os.path.exists(target_model):
        print("\n=== Policy-factor analysis ===")
        # 1) 데이터 수집
        rows = collect_policy_dataset(target_model, env_kwargs, n_episodes=200, greedy=True)
        print(f"Collected {len(rows)} decisions.")

        # 2) 조건부 확률 p(a=1 | feature-bin)
        keys_binspec = [
            ("channel_quality", 10, "first"),
            ("queue_proc_times", 10, "first"),
            ("queue_comp_units", 10, "first"),
            ("available_computation_units", 10, "first"),
            ("ctx_vel", 10, "first"),
            ("ctx_comp", 10, "first"),
        ]
        print_bin_report(rows, keys_binspec)

        # 3) 그래디언트 민감도 (현재 시점)
        gi = gradient_importance(target_model, env_kwargs, sample_steps=2048, greedy=True)
        print("\n[Gradient importance] top-8 (bigger = more sensitive now)")
        for k, v in list(gi.items())[:8]:
            print(f"  {k:>28s} : {v:.6f}")

        # 4) 희소 대리모델로 규칙/부호
        sur = fit_sparse_surrogate(rows, l1=5e-3, epochs=2000, lr=1e-2, standardize=True)
        print(f"\n[Sparse surrogate] train acc: {sur['acc']*100:.1f}%")
        print("  top-8 weights (sign: + → offload, - → local):")
        i = 0
        for k, w in sur["per_key"].items():
            print(f"    {k:>28s} : {w:+.4f}")
            i += 1
            if i >= 8: break
    else:
        print(f"[skip] model not found: {target_model}")
