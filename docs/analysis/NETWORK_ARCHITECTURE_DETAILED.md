# Network Architecture Detailed Analysis

**Based on**: `drl_framework/networks.py`
**Date**: 2026-02-06

---

## 1. 전체 구조 개요

프로젝트는 **두 가지 변형**을 제공합니다:

### A. ActorCritic (Feedforward, 비교 베이스라인)
```
Input (state_dim)
  ↓
Linear(state_dim → 128) → ReLU
  ↓
Linear(128 → 128) → ReLU
  ↓         ↘
Policy      Value
(128→A)     (128→1)
```

### B. RecurrentActorCritic (논문의 주요 모델, LayerNorm 포함)
```
Input (state_dim) + Hidden(1, B, 128)
  ↓                      ↓
Embedding Pipeline    Previous h
  ↓                      ↓
  └──────► GRU ◄────────┘
              ↓
         LayerNorm
              ↓
         ┌────┴────┐
      Policy    Value
```

---

## 2. RecurrentActorCritic 상세 구조 (layer-by-layer)

### **입력 계층**
- **관측 벡터**: `x ∈ ℝ^(state_dim)` — 크기는 환경 의존 (flattened dict)
- **Hidden state**: `hx ∈ ℝ^(L=1, B, H=128)` — 이전 시점의 GRU 메모리

### **Layer 1: Feature Embedding**
```python
z = self.feature(x)          # Linear(state_dim → 128)
z = self.ln_feature(z)       # LayerNorm(128)
z = F.relu(z)                # ReLU
```
**출력**: `z ∈ ℝ^(B, 128)`

**LayerNorm 위치**: Linear 직후, activation 직전
- 목적: 입력 분포 안정화 (state normalization 대신)
- 수식: `LayerNorm(x) = γ·(x - μ)/√(σ² + ε) + β`

### **Layer 2: Recurrent Processing**
```python
z_rnn, next_hx = self.rnn(z.unsqueeze(1), hx)  # GRU
#   입력: z (B, 1, 128), hx (1, B, 128)
#   출력: z_rnn (B, 1, 128), next_hx (1, B, 128)

z = z_rnn[:, 0, :]           # (B, 128) — sequence dimension 제거
z = self.ln_rnn(z)           # LayerNorm(128)
```
**GRU 구조**:
- `num_layers=1`, `batch_first=True`
- 내부 업데이트: Reset gate, Update gate, Candidate hidden
- PyTorch GRU 기본 수식:
  ```
  r_t = σ(W_ir·x_t + b_ir + W_hr·h_{t-1} + b_hr)
  z_t = σ(W_iz·x_t + b_iz + W_hz·h_{t-1} + b_hz)
  n_t = tanh(W_in·x_t + b_in + r_t ⊙ (W_hn·h_{t-1} + b_hn))
  h_t = (1 - z_t) ⊙ n_t + z_t ⊙ h_{t-1}
  ```

**LayerNorm 위치**: GRU 출력 직후
- 목적: value loss 폭발 방지, asynchronous update 안정화 (주석 line 79 참고)

### **Layer 3: Output Heads**
```python
logits = self.policy(z)      # Linear(128 → action_dim)
value = self.value(z)        # Linear(128 → 1)
```
- **Policy**: Categorical distribution의 logits (softmax 전)
- **Value**: 상태 가치 추정 V(s)

**출력**:
- `logits ∈ ℝ^(B, A)` where A=3 (LOCAL, OFFLOAD, DISCARD)
- `value ∈ ℝ^(B, 1)`

---

## 3. 차원 흐름표 (Batch=1 기준)

| Layer | Operation | Input Shape | Output Shape | Parameters |
|-------|-----------|-------------|--------------|------------|
| Input | - | `(1, d)` | `(1, d)` | - |
| **Embedding** |
| Linear | W·x + b | `(1, d)` | `(1, 128)` | `(d×128) + 128` |
| LayerNorm | normalize | `(1, 128)` | `(1, 128)` | `γ(128), β(128)` |
| ReLU | max(0,x) | `(1, 128)` | `(1, 128)` | - |
| **Recurrent** |
| Unsqueeze | add T dim | `(1, 128)` | `(1, 1, 128)` | - |
| GRU | recurrent | `(1, 1, 128)` + h | `(1, 1, 128)` + h' | `3×(128×128 + 128×128)` |
| Squeeze | remove T | `(1, 1, 128)` | `(1, 128)` | - |
| LayerNorm | normalize | `(1, 128)` | `(1, 128)` | `γ(128), β(128)` |
| **Heads** |
| Policy | W·z + b | `(1, 128)` | `(1, 3)` | `(128×3) + 3` |
| Value | w·z + b | `(1, 128)` | `(1, 1)` | `(128×1) + 1` |

---

## 4. Hidden State 관리

### **초기화** (episode 시작)
```python
hx = model.init_hidden(batch_size=1, device=device)
# → torch.zeros(1, 1, 128)
```

### **전파** (decision epoch마다)
```python
logits, value, next_hx = model.step(x, hx)
hx = next_hx  # 다음 스텝으로 전달
```

### **리셋** (episode 종료 또는 done=True)
```python
if done:
    hx = hx * 0.0  # 즉시 리셋
```

**Rollout 시 done 처리** (line 183-186):
```python
if done_seq is not None:
    done_t = done_seq[:, t].float().view(1, B, 1)  # (1, B, 1)
    h = h * (1.0 - done_t)  # done=1인 배치는 h=0으로
```

---

## 5. step() vs rollout() 메서드 비교

| | step() | rollout() |
|---|---|---|
| **용도** | 환경 상호작용 (1 epoch) | 학습 시 T-step 재평가 |
| **입력** | `(B, state_dim)` | `(B, T, state_dim)` |
| **출력** | `(B, A), (B, 1), hx'` | `(B, T, A), (B, T, 1), hx'` |
| **Hidden 전파** | 단일 스텝 | T회 반복, done 마스킹 |
| **Gradient** | `@torch.no_grad()` (act용) | 역전파 가능 |

**Rollout 내부 루프** (line 175-186):
```python
for t in range(T):
    xt = x_seq[:, t, :]
    logits_t, value_t, h = self.step(xt, h)  # 매 t마다 step 호출
    # done 처리로 hidden 리셋
```

---

## 6. Weight Initialization

**모든 Linear layer** (line 104-106):
```python
nn.init.orthogonal_(m.weight, gain=1.0)
nn.init.zeros_(m.bias)
```

**LayerNorm**: PyTorch 기본값 유지
- `γ = 1.0` (scale)
- `β = 0.0` (shift)

---

## 7. ActorCritic vs RecurrentActorCritic 비교

| 구성요소 | ActorCritic | RecurrentActorCritic |
|---------|-------------|----------------------|
| Embedding | `Linear → ReLU` (×2) | `Linear → LayerNorm → ReLU` |
| Recurrent | ❌ | `GRU(128→128, L=1)` |
| Post-RNN LN | ❌ | `LayerNorm(128)` |
| Hidden state | Dummy (compatibility) | Real `(L, B, H)` |
| Params | ~33K | ~100K (GRU 추가) |

---

## 8. 그림 작성을 위한 핵심 요소

### **박스 구성** (위→아래 순서):
1. **Input**: `observation ∈ ℝ^d` + `h ∈ ℝ^128` (from previous epoch)
2. **Embedding**: `Linear(d→128)` → `LayerNorm` → `ReLU`
3. **GRU Cell**:
   - 입력: embedded `z` + previous `h`
   - 출력: `h'` (next hidden state)
4. **LayerNorm**: GRU 출력 정규화
5. **Split**:
   - Left: `Linear(128→3)` → `π(·|o;θ)` (policy logits)
   - Right: `Linear(128→1)` → `V(o;θ)` (value)

### **화살표**:
- `observation → Embedding`
- `Embedding output → GRU`
- `previous h → GRU` (점선, recurrent connection)
- `GRU → LayerNorm`
- `LayerNorm → Policy head`
- `LayerNorm → Value head`
- `GRU output h' → next epoch` (점선, feedback)

### **차원 표시**:
각 화살표 옆에 shape 표기:
- `(B, d)`, `(B, 128)`, `(1, B, 128)`, `(B, 3)`, `(B, 1)`

### **색상 제안**:
- Embedding: 파란색
- GRU: 초록색 (recurrent 강조)
- LayerNorm: 회색 (보조 계층)
- Policy head: 주황색
- Value head: 보라색

---

## 9. Code References

- **ActorCritic**: `networks.py:7-68`
- **RecurrentActorCritic**: `networks.py:70-197`
- **Feature embedding**: `networks.py:88-89, 142-144`
- **GRU layer**: `networks.py:92-94, 147-149`
- **Policy/Value heads**: `networks.py:97-98, 152-157`
- **Hidden state init**: `networks.py:110-111`
- **step() method**: `networks.py:127-158`
- **rollout() method**: `networks.py:161-190`

---

## 10. Key Insight

이 구조는 **POMDP 환경에서 partial observability를 hidden state로 보완**하는 전형적인 recurrent actor-critic 구조입니다. LayerNorm의 전략적 배치(embedding 후, GRU 후)는 A3C의 asynchronous gradient update 환경에서 학습 안정성을 높이는 핵심 설계입니다.
