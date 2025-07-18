import torch
import numpy as np
import os
import csv
from drl_framework.networks import ActorCritic
from drl_framework.custom_env import make_env
from drl_framework.utils import flatten_dict_values
import drl_framework.params as params
import copy

device = params.device
ENV_PARAMS = params.ENV_PARAMS
REWARD_PARAMS = params.REWARD_PARAMS

temp_env_fn = make_env(**ENV_PARAMS)
temp_env = temp_env_fn()
sample_obs = temp_env.reset()[0]
state_dim = len(flatten_dict_values(sample_obs))
action_dim = temp_env.action_space.n
hidden_dim = params.hidden_dim

env_param_list = []
for i in range(n_eval_workers:=5):
    e = copy.deepcopy(ENV_PARAMS)
    e["max_comp_units"] = np.random.randint(80, 121)
    e["agent_velocities"] = np.random.randint(30, 101)
    env_param_list.append(e)


def evaluate_model(model_path, e, n_episodes=100, render=False):
    """단일 모델 평가"""
    # 모델 로드
    model = ActorCritic(state_dim, action_dim, hidden_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 환경 생성
    env_fn = make_env(**e, reward_params=REWARD_PARAMS)
    env = env_fn()
    
    episode_rewards = []
    episode_lengths = []
    
    with torch.no_grad():
        for episode in range(n_episodes):
            state, _ = env.reset()
            total_reward = 0.0
            steps = 0
            done = False

            while not done and steps < params.ENV_PARAMS['max_epoch_size']:
                # 상태를 텐서로 변환
                state_tensor = torch.FloatTensor(flatten_dict_values(state)).unsqueeze(0).to(device)
                
                # 행동 선택 (greedy policy for evaluation)
                logits, _ = model(state_tensor)
                action = torch.argmax(logits, dim=-1).item()
                
                # 환경에서 스텝 실행
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                total_reward += reward
                steps += 1
                state = next_state
                
                if render:
                    env.render()
            
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
    
    env.close()
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'rewards': episode_rewards
    }

def compare_all_models(n_episodes=100):
    """모든 모델들을 비교 평가"""
    results = {}
    
    # A3C global model 평가
    if os.path.exists("models/global_final.pth"):
        print("Evaluating A3C Global Model...")
        results['A3C_Global'] = evaluate_model(
            "models/global_final.pth", 
            e, 
            n_episodes
        )
    
    # Individual models 평가
    individual_results = []
    for i in range(params.n_workers):
        model_path = f"models/individual_worker_{i}_final.pth"
        if os.path.exists(model_path):
            print(f"Evaluating Individual Worker {i}...")
            result = evaluate_model(model_path, e, n_episodes)
            result['worker_id'] = i
            individual_results.append(result)
    
    results['Individual_Workers'] = individual_results
    
    return results

def print_results(results):
    """결과를 보기 좋게 출력"""
    print("\n" + "="*60)
    print("MODEL EVALUATION RESULTS")
    print("="*60)
    
    # A3C Global 결과
    if 'A3C_Global' in results:
        r = results['A3C_Global']
        print(f"\n📊 A3C Global Model:")
        print(f"  Mean Reward: {r['mean_reward']:.2f} ± {r['std_reward']:.2f}")
        print(f"  Range: [{r['min_reward']:.2f}, {r['max_reward']:.2f}]")
        print(f"  Mean Episode Length: {r['mean_length']:.1f}")
    
    # Individual Workers 결과
    if 'Individual_Workers' in results and results['Individual_Workers']:
        print(f"\n📊 Individual Workers:")
        all_individual_rewards = []
        
        for result in results['Individual_Workers']:
            worker_id = result['worker_id']
            print(f"  Worker {worker_id}: {result['mean_reward']:.2f} ± {result['std_reward']:.2f}")
            all_individual_rewards.extend(result['rewards'])
        
        # Individual workers 전체 통계
        print(f"\n  📈 Individual Workers Overall:")
        print(f"    Mean: {np.mean(all_individual_rewards):.2f}")
        print(f"    Std: {np.std(all_individual_rewards):.2f}")
    
    # 비교 결과
    if 'A3C_Global' in results and 'Individual_Workers' in results:
        a3c_mean = results['A3C_Global']['mean_reward']
        individual_means = [r['mean_reward'] for r in results['Individual_Workers']]
        best_individual = max(individual_means) if individual_means else 0
        avg_individual = np.mean(individual_means) if individual_means else 0
        
        print(f"\n🏆 COMPARISON:")
        print(f"  A3C Global vs Best Individual: {a3c_mean:.2f} vs {best_individual:.2f}")
        print(f"  A3C Global vs Avg Individual: {a3c_mean:.2f} vs {avg_individual:.2f}")
        
        improvement = ((a3c_mean - avg_individual) / abs(avg_individual)) * 100 if avg_individual != 0 else 0
        print(f"  A3C Improvement: {improvement:+.1f}%")

def save_detailed_results(results, filename="evaluation_results.csv"):
    """상세 결과를 CSV로 저장"""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Mean_Reward', 'Std_Reward', 'Min_Reward', 'Max_Reward', 'Mean_Length'])
        
        # A3C Global
        if 'A3C_Global' in results:
            r = results['A3C_Global']
            writer.writerow(['A3C_Global', r['mean_reward'], r['std_reward'], 
                           r['min_reward'], r['max_reward'], r['mean_length']])
        
        # Individual Workers
        if 'Individual_Workers' in results:
            for result in results['Individual_Workers']:
                worker_id = result['worker_id']
                writer.writerow([f'Individual_Worker_{worker_id}', 
                               result['mean_reward'], result['std_reward'],
                               result['min_reward'], result['max_reward'], result['mean_length']])

if __name__ == "__main__":
    # 평가 실행
    print("Starting model evaluation...")
    results = compare_all_models(n_episodes=100)
    
    # 결과 출력
    print_results(results)
    
    # 결과 저장
    save_detailed_results(results)
    print(f"\nDetailed results saved to evaluation_results.csv")