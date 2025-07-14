import torch
import torch.nn as nn
import torch.multiprocessing as mp
from .networks import ActorCritic
from .params import *
from .custom_env import CustomEnv, make_env
from .utils import flatten_dict_values
import os
import csv

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

def worker(rank, global_model, optimizer, env_fn):
    import os
    import csv
    from drl_framework.utils import flatten_dict_values

    torch.manual_seed(123 + rank)

    env = env_fn()
    local_model = ActorCritic(state_dim, action_dim, hidden_dim).to(device)
    local_model.load_state_dict(global_model.state_dict())

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"worker_{rank}_rewards.csv")

    episode_count = 0

    while episode_count < target_episode_count:
        state, _ = env.reset()
        done = False
        episode_steps = 0
        total_reward = 0

        values, log_probs, rewards, entropies = [], [], [], []

        while episode_steps < MAX_EPOCH_SIZE:
            state_tensor = torch.FloatTensor(flatten_dict_values(state)).unsqueeze(0).to(device)
            logits, value = local_model(state_tensor)
            probs = torch.softmax(logits, dim=-1)
            log_prob = torch.log(probs + 1e-8)
            entropy = -(log_prob * probs).sum(1, keepdim=True)

            action = probs.multinomial(num_samples=1).detach()
            selected_log_prob = log_prob.gather(1, action)

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            values.append(value)
            log_probs.append(selected_log_prob)
            rewards.append(reward)
            entropies.append(entropy)

            state = next_state
            total_reward += reward
            episode_steps += 1

        # 에피소드 종료 후 업데이트
        R = torch.zeros(1, 1).to(device) if done else local_model(
            torch.FloatTensor(flatten_dict_values(state)).unsqueeze(0).to(device)
        )[1].detach()
        values.append(R)

        policy_loss, value_loss = 0, 0
        gae = torch.zeros(1, 1).to(device)
        for i in reversed(range(len(rewards))):
            R = gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss += advantage.pow(2)

            delta_t = rewards[i] + gamma * values[i + 1] - values[i]
            gae = gae * gamma + delta_t
            policy_loss -= log_probs[i] * gae.detach() - entropy_coef * entropies[i]

        total_loss = policy_loss + value_loss_coef * value_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_grad_norm)
        for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
            global_param._grad = local_param.grad
        optimizer.step()
        local_model.load_state_dict(global_model.state_dict())

        # 로그 및 종료 조건
        episode_count += 1
        print(f"[Worker {rank}] episode {episode_count}, reward={total_reward:.2f}")
        with open(log_path, mode="a", newline="") as f:
            csv.writer(f).writerow([episode_count, total_reward])


def train():
    mp.set_start_method("spawn", force=True)

    global_model = ActorCritic(state_dim, action_dim, hidden_dim)
    global_model.share_memory()
    global_model = global_model.to(device)
    optimizer = SharedAdam(global_model.parameters(), lr=lr)

    processes = []
    for rank in range(n_workers):
        env_fn = make_env(**ENV_PARAMS)
        p = mp.Process(target=worker, args=(rank, global_model, optimizer, env_fn))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
