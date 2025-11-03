import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .params import device

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy = nn.Linear(hidden_dim, action_dim)
        self.value = nn.Linear(hidden_dim, 1)
        self.apply(self._init_weights)
        self.hidden_dim = hidden_dim

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            # Orthogonal init: policy는 작은 gain, value는 보통
            nn.init.orthogonal_(m.weight, gain=1.0)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.shared(x)
        policy_logits = self.policy(x)
        state_value = self.value(x)
        return policy_logits, state_value

    def init_hidden(self, batch_size=1, device='cpu'):
        """Dummy hidden state for compatibility with RNN interface"""
        return torch.zeros(1, batch_size, self.hidden_dim, device=device)

    def step(self, x, hx=None):
        """Single step forward (compatible with RNN interface)
        Args:
            x: (B, state_dim) input
            hx: dummy hidden state (ignored)
        Returns:
            logits: (B, action_dim)
            value: (B, 1)
            hx: dummy hidden state (unchanged)
        """
        logits, value = self.forward(x)
        return logits, value, hx

    def rollout(self, x_seq, hx=None, done_seq=None):
        """Rollout over sequence (compatible with RNN interface)
        Args:
            x_seq: (B, T, state_dim)
            hx: dummy hidden state (ignored)
            done_seq: (B, T) done flags (ignored for feedforward)
        Returns:
            logits_seq: (B, T, action_dim)
            values_seq: (B, T, 1)
            hx: dummy hidden state (unchanged)
        """
        B, T, state_dim = x_seq.shape
        x_flat = x_seq.reshape(B * T, state_dim)  # (B*T, state_dim)
        logits_flat, values_flat = self.forward(x_flat)
        logits_seq = logits_flat.reshape(B, T, -1)  # (B, T, action_dim)
        values_seq = values_flat.reshape(B, T, 1)   # (B, T, 1)
        return logits_seq, values_seq, hx
    
    
class RecurrentActorCritic(nn.Module):
    """
    입력 x: (B, state_dim) 또는 (B, T, state_dim)
    - step() : 온라인/환경 상호작용 1스텝용 (T=1)
    - rollout(): 길이 T 롤아웃 학습용 (done 마스크로 hx 리셋)

    Layer Normalization:
    - Feature extraction: Linear → LayerNorm → ReLU
    - RNN output: GRU → LayerNorm
    - Value loss 폭발 방지 및 학습 안정성 향상
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128, num_layers=1, use_layer_norm=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_layer_norm = use_layer_norm

        # 1) 관측 임베딩 + Layer Normalization
        self.feature = nn.Linear(state_dim, hidden_dim)
        self.ln_feature = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()

        # 2) 순환부 (batch_first=True)
        self.rnn = nn.GRU(hidden_dim, hidden_dim,
                          num_layers=num_layers, batch_first=True)
        self.ln_rnn = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()

        # 3) 헤드
        self.policy = nn.Linear(hidden_dim, action_dim)  # logits
        self.value  = nn.Linear(hidden_dim, 1)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, 1.0)
            nn.init.zeros_(m.bias)
        # LayerNorm은 기본 초기화 유지 (weight=1.0, bias=0.0)

    # --- 유틸 ---
    def init_hidden(self, batch_size, device=None):
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)

    # --- 1-step 경로 (환경 상호작용) ---
    @torch.no_grad()
    def act(self, x, hx, action_mask=None):
        """
        x: (B, state_dim)
        hx: (num_layers, B, hidden_dim)
        return: action (B,), logprob (B,), value (B,1), next_hx
        """
        logits, value, next_hx = self.step(x, hx, action_mask=action_mask)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        logp = dist.log_prob(a)
        return a, logp, value, next_hx

    def step(self, x, hx, action_mask=None):
        """
        학습 시에도 사용 가능. 1스텝 fwd (T=1)
        x: (B, state_dim), hx: (L, B, H)

        Forward pass with Layer Normalization:
        1. Feature: Linear → LayerNorm → ReLU
        2. RNN: GRU → LayerNorm
        3. Heads: Policy and Value
        """
        if x.dim() == 3:  # (B,T,state_dim) 들어오면 마지막 스텝만 쓰는 용도
            assert x.size(1) == 1, "step()에는 T=1만 허용합니다."
            x = x[:, 0, :]

        # Feature extraction with LayerNorm
        z = self.feature(x)              # (B, H)
        z = self.ln_feature(z)           # LayerNorm
        z = F.relu(z)                    # ReLU activation

        # RNN with LayerNorm
        z, next_hx = self.rnn(z.unsqueeze(1), hx)  # (B, 1, H), (L, B, H)
        z = z[:, 0, :]                   # (B, H)
        z = self.ln_rnn(z)               # LayerNorm on RNN output

        # Policy and Value heads
        logits = self.policy(z)
        if action_mask is not None:
            # mask: (B, action_dim) with {0,1}
            logits = logits + (action_mask + 1e-45).log()  # 0→-inf

        value = self.value(z)            # (B, 1)
        return logits, value, next_hx

    # --- T-step 롤아웃 경로 (BPTT) ---
    def rollout(self, x_seq, hx, done_seq=None, action_mask_seq=None):
        """
        x_seq: (B, T, state_dim)
        done_seq: (B, T)  # True면 해당 스텝 이후 hidden을 리셋
        action_mask_seq: (B, T, action_dim) or None
        return:
            logits_seq: (B, T, action_dim)
            values_seq: (B, T, 1)
            next_hx: (L, B, H)
        """
        B, T, _ = x_seq.shape
        logits_list, values_list = [], []
        h = hx

        for t in range(T):
            xt = x_seq[:, t, :]                    # (B, state_dim)
            logits_t, value_t, h = self.step(xt, h,
                                             action_mask=None if action_mask_seq is None
                                             else action_mask_seq[:, t, :])
            logits_list.append(logits_t.unsqueeze(1))  # (B,1,A)
            values_list.append(value_t.unsqueeze(1))   # (B,1,1)

            if done_seq is not None:
                # done=True인 배치의 hidden을 0으로 리셋
                done_t = done_seq[:, t].float().view(1, B, 1)  # (1,B,1)
                h = h * (1.0 - done_t)

        logits_seq = torch.cat(logits_list, dim=1)  # (B,T,A)
        values_seq = torch.cat(values_list, dim=1)  # (B,T,1)
        return logits_seq, values_seq, h
    
    def forward(self, x, hidden=None):
        # x: (batch, seq_len, state_dim)
        rnn_out, hidden = self.rnn(x, hidden)
        logits = self.policy(rnn_out)
        value = self.value(rnn_out)
        return logits, value, hidden