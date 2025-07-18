import torch
import torch.multiprocessing as mp
import os
import csv
from .networks import ActorCritic
from .params import *
from .custom_env import make_env
from .utils import flatten_dict_values

# def make_env(**kwargs):
#     return partial(CustomEnv, **kwargs)

temp_env_fn = make_env(**ENV_PARAMS)
temp_env = temp_env_fn()
sample_obs = temp_env.reset()[0]
state_dim = len(flatten_dict_values(sample_obs))
action_dim = temp_env.action_space.n


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

# def worker(rank, global_model, optimizer, env_fn):
#     import os
#     import csv
#     from drl_framework.utils import flatten_dict_values

#     torch.manual_seed(123 + rank)

#     env = env_fn()
#     local_model = ActorCritic(state_dim, action_dim, hidden_dim).to(device)
#     local_model.load_state_dict(global_model.state_dict())

#     os.makedirs(log_dir, exist_ok=True)
#     log_path = os.path.join(log_dir, f"worker_{rank}_rewards.csv")

#     episode_count = 0

#     while episode_count < target_episode_count:
#         state, _ = env.reset()
#         done = False
#         episode_steps = 0
#         total_reward = 0

#         values, log_probs, rewards, entropies = [], [], [], []

#         while episode_steps < ENV_PARAMS['max_epoch_size']:
#             state_tensor = torch.FloatTensor(flatten_dict_values(state)).unsqueeze(0).to(device)
#             logits, value = local_model(state_tensor)
#             probs = torch.softmax(logits, dim=-1)
#             log_prob = torch.log(probs + 1e-8)
#             entropy = -(log_prob * probs).sum(1, keepdim=True)

#             action = probs.multinomial(num_samples=1).detach()
#             selected_log_prob = log_prob.gather(1, action)

#             next_state, reward, terminated, truncated, _ = env.step(action.item())
#             done = terminated or truncated

#             values.append(value)
#             log_probs.append(selected_log_prob)
#             rewards.append(reward)
#             entropies.append(entropy)

#             state = next_state
#             total_reward += reward
#             episode_steps += 1

#         # 에피소드 종료 후 업데이트
#         R = torch.zeros(1, 1).to(device) if done else local_model(
#             torch.FloatTensor(flatten_dict_values(state)).unsqueeze(0).to(device))[1].detach()
#         values.append(R)

#         policy_loss, value_loss = 0, 0
#         gae = torch.zeros(1, 1).to(device)
#         for i in reversed(range(len(rewards))):
#             R = gamma * R + rewards[i]
#             advantage = R - values[i]
#             value_loss += advantage.pow(2)

#             delta_t = rewards[i] + gamma * values[i + 1] - values[i]
#             gae = gae * gamma + delta_t
#             policy_loss -= log_probs[i] * gae.detach() - entropy_coef * entropies[i]

#         total_loss = policy_loss + value_loss_coef * value_loss

#         optimizer.zero_grad()
#         total_loss.backward()
#         torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_grad_norm)
#         for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
#             global_param._grad = local_param.grad
#         optimizer.step()
#         local_model.load_state_dict(global_model.state_dict())

#         # 로그 및 종료 조건
#         episode_count += 1
#         print(f"[Worker {rank}] episode {episode_count}, reward={total_reward:.2f}, loss={total_loss.item():.4f}")
#         with open(log_path, mode="a", newline="") as f:
#             csv.writer(f).writerow([episode_count, total_reward, total_loss.item()])


def universal_worker(rank, model, optimizer, env_fn, log_path, use_global_model=True, total_episodes=1000):
    torch.manual_seed(123 + rank)
    env = env_fn()
    
    if use_global_model:
        local_model = ActorCritic(state_dim, action_dim, hidden_dim).to(device)
        local_model.load_state_dict(model.state_dict())
        working_model = local_model
    else:
        working_model = model

    for episode in range(1, total_episodes + 1):
        state, _ = env.reset()
        done = False
        total_reward = 0.0
        episode_steps = 0

        # 에피소드 동안 데이터 수집
        states, actions, rewards, values, log_probs, entropies = [], [], [], [], [], []

        while not done and episode_steps < ENV_PARAMS['max_epoch_size']:
            state_tensor = torch.FloatTensor(flatten_dict_values(state)).unsqueeze(0).to(device)
            logits, value = working_model(state_tensor)
            probs = torch.softmax(logits, dim=-1)
            log_prob = torch.log(probs + 1e-8)
            entropy = -(log_prob * probs).sum(1, keepdim=True)

            action = probs.multinomial(num_samples=1).detach()
            selected_log_prob = log_prob.gather(1, action)

            next_state, reward, done, _, _ = env.step(action.item())

            # 데이터 저장
            states.append(state_tensor)
            actions.append(action)
            rewards.append(reward)
            values.append(value)
            log_probs.append(selected_log_prob)
            entropies.append(entropy)

            total_reward += reward
            episode_steps += 1
            state = next_state

        # 에피소드 종료 후 returns 계산
        returns = []
        R = 0.0 if done else working_model(torch.FloatTensor(flatten_dict_values(state)).unsqueeze(0).to(device))[1].item()
        
        # Discounted returns 계산 (역순으로)
        for i in reversed(range(len(rewards))):
            R = rewards[i] + gamma * R
            returns.insert(0, R)
        
        returns = torch.FloatTensor(returns).to(device)
        
        # Advantage 계산
        values_tensor = torch.cat(values).squeeze()
        advantages = returns - values_tensor
        
        # 정규화 (선택적)
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Loss 계산
        policy_loss = 0
        value_loss = 0
        entropy_loss = 0
        
        for i in range(len(log_probs)):
            # Policy loss (REINFORCE with baseline)
            policy_loss -= log_probs[i] * advantages[i].detach()
            
            # Value loss (MSE)
            value_loss += (returns[i] - values[i]).pow(2)
            
            # Entropy loss (exploration 장려)
            entropy_loss -= entropies[i]

        # 총 loss
        total_loss = (policy_loss + 
                     value_loss_coef * value_loss + 
                     entropy_coef * entropy_loss)

        # Gradient 업데이트
        optimizer.zero_grad()
        total_loss.backward()

        if use_global_model:
            # A3C: global model 업데이트
            torch.nn.utils.clip_grad_norm_(working_model.parameters(), max_grad_norm)
            for lp, gp in zip(working_model.parameters(), model.parameters()):
                if gp._grad is None:
                    gp._grad = lp.grad.clone()
                else:
                    gp._grad += lp.grad
            optimizer.step()
            working_model.load_state_dict(model.state_dict())
        else:
            # Individual: 자체 모델 업데이트
            torch.nn.utils.clip_grad_norm_(working_model.parameters(), max_grad_norm)
            optimizer.step()

        # 로깅
        avg_loss = total_loss.item() / len(rewards)  # 스텝당 평균 loss
        
        with open(log_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([episode, total_reward, avg_loss])

        if episode % 100 == 0:
            print(f"[Worker {rank}] Episode {episode}, Reward: {total_reward:.2f}, Avg Loss: {avg_loss:.4f}")


# def train(n_workers=5, env_param_list=None):
#     mp.set_start_method("spawn", force=True)

#     global_model = ActorCritic(state_dim, action_dim, hidden_dim)
#     global_model.share_memory()
#     global_model = global_model.to(device)
#     optimizer = SharedAdam(global_model.parameters(), lr=lr)

#     os.makedirs("models", exist_ok=True)
#     os.makedirs("logs", exist_ok=True)

#     for rank in range(n_workers):
#         log_path = os.path.join("logs", f"worker_{rank}_rewards.csv")
#         with open(log_path, mode="w", newline="") as f:
#             writer = csv.writer(f)
#             writer.writerow(["episode", "reward", "loss"])

#     processes = []
#     for rank in range(n_workers):
#         env_params = env_param_list[rank] if env_param_list else ENV_PARAMS
#         env_fn = make_env(**env_params, reward_params=REWARD_PARAMS)
#         p = mp.Process(target=worker, args=(rank, global_model, optimizer, env_fn))
#         p.start()
#         processes.append(p)

#     for p in processes:
#         p.join()

#     torch.save(global_model.state_dict(), "models/global_final.pth")
#     print("Training complete. Final model saved.")

def train(n_workers, total_episodes, env_param_list=None):
    mp.set_start_method("spawn", force=True)

    global_model = ActorCritic(state_dim, action_dim, hidden_dim)
    global_model.share_memory()
    global_model = global_model.to(device)
    optimizer = SharedAdam(global_model.parameters(), lr=lr)

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    for rank in range(n_workers):
        log_path = os.path.join("logs", f"A3C_worker_{rank}_rewards.csv")
        with open(log_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "reward", "loss"])

    processes = []
    for rank in range(n_workers):
        env_params = env_param_list[rank] if env_param_list else ENV_PARAMS
        env_fn = make_env(**env_params, reward_params=REWARD_PARAMS)
        log_path = os.path.join("logs", f"A3C_worker_{rank}_rewards.csv")
        p = mp.Process(target=universal_worker, args=(rank, global_model, optimizer, env_fn, log_path, True, total_episodes))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    torch.save(global_model.state_dict(), "models/global_final.pth")
    print("Training complete. Final model saved.")


def train_individual(n_workers, total_episodes, env_param_list=None):
    # os.makedirs("models_individual", exist_ok=True)
    # os.makedirs("logs_individual", exist_ok=True)

    for rank in range(n_workers):
        env_params = env_param_list[rank] if env_param_list else ENV_PARAMS
        env_fn = make_env(**env_params, reward_params=REWARD_PARAMS)

        model = ActorCritic(state_dim, action_dim, hidden_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        log_path = os.path.join("logs", f"individual_worker_{rank}_rewards.csv")
        with open(log_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "reward", "loss"])

        print(f"[Worker {rank}] Starting individual training")
        universal_worker(rank, model, optimizer, env_fn, log_path, use_global_model=False, total_episodes=total_episodes)

        torch.save(model.state_dict(), f"models/individual_worker_{rank}_final.pth")
        print(f"[Worker {rank}] Training complete. Model saved.")