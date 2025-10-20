# Work Summary - Action Masking & Reward Simplification (2025-10-18)

## Overview

이번 세션에서는 A3C 모델의 **action masking 문제**와 **reward 구조 단순화**를 진행했습니다. 주요 발견사항은 action masking이 policy gradient 계산에서 심각한 불일치를 야기하고 있었으며, 이를 제거하고 reward를 단순화했습니다.

---

## 1. Action Masking 문제 발견

### 1.1 문제 상황

**이전 모델 (20251018_023558):**
- 학습 시: Action masking 사용
- 평가 결과:
  - A3C Global (Stochastic): 72.23
  - A3C Global (Greedy): 68.5 (-5%)
  - Individual (Stochastic): 71.22
  - Individual (Greedy): 30.8 (-57%) ❌

**핵심 발견:**
- Greedy policy에서 Individual workers가 완전히 붕괴 (70 → 30)
- Worker 4는 policy collapse (모든 환경에서 49.6 고정)

### 1.2 근본 원인 분석

**Action Masking의 치명적 결함:**

```python
# 샘플링 시 (trainer.py:311-320)
action_mask = env.get_action_mask()
masked_logits[~mask] = float('-inf')
dist = Categorical(logits=masked_logits)  # ✅ Masked 사용

# 학습 시 (trainer.py:380-382)
dist_seq = Categorical(logits=logits_seq)  # ❌ 원본 logits 사용!
logp_seq = dist_seq.log_prob(act_seq_t)
entropy_seq = dist_seq.entropy()
```

**문제점:**
1. **샘플링 분포 ≠ 학습 분포** → Policy gradient 불일치
2. **Entropy 계산 오류:** Invalid action이 entropy를 부풀림
3. **Log probability 불일치:** 샘플링한 action의 확률이 잘못 계산됨

### 1.3 해결 방안 1차 시도 (실패)

**Soft Action Masking 시도:**
- Masked logits를 학습 시에도 사용하도록 수정
- Action mask를 rollout 시 저장
- 학습 시 masked distribution으로 entropy/log_prob 계산

**문제:**
- 구현 복잡도 증가
- 평가 시 masking 여부에 따라 성능 차이 발생

---

## 2. Action Masking 완전 제거

### 2.1 수정 내용

**[trainer.py:310-314](drl_framework/trainer.py#L310-314):**
```python
# 이전
action_mask = env.get_action_mask()
masked_logits = logits.clone()
masked_logits[~mask] = float('-inf')
dist = Categorical(logits=masked_logits)

# 수정 후
dist = Categorical(logits=logits)  # 원본 logits 직접 사용
```

**[trainer.py:375-377](drl_framework/trainer.py#L375-377):**
```python
# 이전
masked_logits_seq = logits_seq.clone()
masked_logits_seq[~mask_seq_t] = float('-inf')
dist_seq = Categorical(logits=masked_logits_seq)

# 수정 후
dist_seq = Categorical(logits=logits_seq)  # 원본 사용
```

**동일 수정:**
- A3C universal_worker
- Individual_worker

### 2.2 기대 효과

1. ✅ **샘플링-학습 일관성:** 동일한 distribution 사용
2. ✅ **Entropy 정확성:** 모든 action 포함된 실제 entropy
3. ✅ **학습-평가 일관성:** Masking 없이 일관된 정책
4. ✅ **Agent 학습:** State로부터 valid/invalid 스스로 학습

---

## 3. Reward 구조 단순화

### 3.1 이전 Reward 구조 (20251018_023558)

```python
# 즉시 패널티
LOCAL 실패: -2 * FAILURE_PENALTY = -10.0
OFFLOAD 실패: -2 * FAILURE_PENALTY = -10.0
DISCARD: -FAILURE_PENALTY = -5.0

# 완료 보상
MEC 완료: +done_comp
Cloud 완료: +done_comp - BETA*consumed_time

# 부분 보상 (에피소드 종료 시)
미완료 작업: +comp * progress * 0.5
```

### 3.2 수정된 Reward 구조 (20251018_185651)

**[custom_env.py:214-260](drl_framework/custom_env.py#L214-260):**
```python
# 즉시 패널티/보너스 완전 제거
self.reward = 0

if action == LOCAL:
    # 패널티 없음
if action == OFFLOAD:
    # 패널티 없음
elif action == DISCARD:
    pass  # 아무것도 하지 않음
```

**[custom_env.py:277-294](drl_framework/custom_env.py#L277-294):**
```python
# 완료된 comp_units만 보상
if zeroed_mec.any():
    done_comp = self.mec_comp_units[zeroed_mec].sum()
    self.reward += done_comp

if zeroed_cloud.any():
    done_comp = self.cloud_comp_units[zeroed_cloud].sum()
    self.reward += done_comp  # latency 비용도 제거
```

**[custom_env.py:305-306](drl_framework/custom_env.py#L305-306):**
```python
# 부분 보상 제거
# 에피소드 종료 시 미완료 작업은 보상 없음
```

### 3.3 단순화 효과

**장점:**
- ✅ **명확한 목표:** 완료된 comp_units 최대화
- ✅ **단순한 학습 신호:** 즉시 패널티 없이 완료만 보상
- ✅ **Credit assignment 단순화**

**단점:**
- ❌ **DISCARD 억제 부족:** 패널티 없어서 DISCARD만 선택 가능
- ❌ **탐색 부족:** 실패에 대한 피드백 없음

---

## 4. 학습 결과 (20251018_185651)

### 4.1 학습 환경 설정

```python
ENV_PARAMS = {
    'max_comp_units': 200,
    'max_comp_units_for_cloud': 1000,
    'max_epoch_size': 100,
    'max_queue_size': 20,
}

# Worker별 velocity: 5 + (25-5) * i / (n_workers-1)
Worker 0: 5 km/h
Worker 1: 10 km/h
Worker 2: 15 km/h
Worker 3: 20 km/h
Worker 4: 25 km/h

REWARD_PARAMS = {
    'REWARD_SCALE': 0.05,  # 스케일링만 유지
    # 모든 패널티 제거
}

# 학습 파라미터
entropy_coef = 0.1
value_loss_coef = 0.25
lr = 5e-5
n_workers = 5
target_episode_count = 5000
```

### 4.2 학습 성능

**A3C Global (Episode 1000):**
- Reward: 63.53
- Entropy: 0.85 ✅ (collapse 없음)
- Value Loss: 43.90
- Policy Loss: -0.082

**Individual Workers (Episode 1000):**
| Worker | Reward | Entropy | Value Loss | 상태 |
|--------|--------|---------|------------|------|
| W0 | 61.75 | 1.032 | 77.60 | 정상 |
| W1 | 53.35 | 1.058 | 457.81 | 불안정 ⚠️ |
| W2 | 46.20 | 1.082 | 688.70 | 불안정 ❌ |
| W3 | 87.50 | 1.008 | 675.35 | 불안정 ❌ |
| W4 | 39.45 | 1.050 | 546.60 | 불안정 ❌ |

**문제:** Individual workers의 value loss가 매우 불안정

### 4.3 일반화 성능 (Greedy Policy)

**Seen 환경:**
- A3C Global: 63.92
- Individual Avg: 34.45
- **A3C 우위: +85%**

**Intra 환경 (보간):**
- A3C Global: 65.41
- Individual Avg: 30.16
- **A3C 우위: +117%**

**Extra 환경 (외삽):**
- A3C Global: 54.22
- Individual Avg: 20.35
- **A3C 우위: +166%**

### 4.4 Worker 3 Policy Collapse

**증상:**
- 모든 환경에서 49.6~49.7 고정
- comp_units, velocity 무관하게 동일
- std = 0.3 (거의 변동 없음)

**진단:**
- DISCARD-only policy
- 어떤 action도 학습하지 못함
- 완전한 학습 실패

**원인 분석:**
- Reward 단순화로 DISCARD 억제 부족
- DISCARD해도 패널티 없음
- "아무것도 안하는" 정책이 안전함

---

## 5. 20251018_023558 vs 20251018_185651 비교

### 5.1 환경 설정 차이

| 항목 | 023558 (Baseline) | 185651 (New) |
|------|------------------|--------------|
| **Action Masking** | O | X |
| **Velocity 범위** | 5-20 km/h | 5-25 km/h |
| **LOCAL 실패 패널티** | -10.0 | 0 |
| **OFFLOAD 실패 패널티** | -10.0 | 0 |
| **DISCARD 패널티** | -5.0 | 0 |
| **Cloud latency 비용** | -BETA*time | 0 |
| **부분 보상** | +comp*progress*0.5 | 0 |

### 5.2 성능 비교

**A3C Global:**
| 환경 | 023558 | 185651 | 변화 |
|------|--------|--------|------|
| Seen | 68.5 | 63.9 | -6.7% |
| Intra | 67.5 | 65.4 | -3.1% |
| Extra | 56.6 | 54.2 | -4.2% |
| Entropy | 1.061 | 0.85 | -19% |

**Individual Workers:**
| 환경 | 023558 | 185651 | 변화 |
|------|--------|--------|------|
| Seen | 30.8 | 34.5 | +12% ✅ |
| W4 collapse | 49.6 | 49.7 | 동일 ❌ |

### 5.3 핵심 통찰

**A3C Global:**
- Action masking 제거에도 강건함 유지
- 성능 약간 감소 (-4~7%)
- 여전히 일반화 능력 우수

**Individual Workers:**
- Worker 0-2, 4: 약간 개선 (30 → 34)
- Worker 3: 여전히 collapse (49.7)
- **근본적 문제 미해결**

---

## 6. 논문 Contribution 분석

### 6.1 현재 상태 평가

**주장:** "Individual training의 단점이 많으니 A3C를 제안한다"

**Reviewer 예상 반응:**

**Reviewer 1:**
> "A3C가 Individual보다 우수한 것은 당연합니다. Individual은 단일 환경만 학습하므로 일반화가 안 됩니다. 새로운 contribution이 무엇인가요?"

**Reviewer 2:**
> "Individual workers의 성능이 너무 낮습니다 (34.5). Worker 3의 policy collapse는 학습 실패입니다. 비교가 공정하지 않습니다."

**Reviewer 3:**
> "UAV task offloading에 A3C를 적용한 것은 응용 contribution입니다. 하지만 baseline 비교가 약합니다. DQN, PPO, SAC와 비교해야 합니다."

### 6.2 Contribution 강도 평가

**현재 상태:**
- ❌ **Top-tier (JSAC, TWC, INFOCOM):** 부족
- ⚠️ **Mid-tier (IoT, VTC, GLOBECOM):** 경계선
- ✅ **Application-focused:** 충분 가능

**부족한 점:**
1. Individual 성능이 너무 낮음 (학습 실패)
2. Baseline 비교 부족 (DQN, PPO 없음)
3. Policy collapse 문제 미해결

### 6.3 개선 방향

**Option 1: Individual 학습 개선**
```python
# DISCARD penalty 추가
elif action == DISCARD:
    self.reward -= 1.0  # 작은 패널티

# 목표: Seen 60~70 달성
# 공정한 비교 확보
```

**Option 2: 추가 Baseline**
- DQN (individual)
- PPO (centralized)
- Independent Q-learning

**Option 3: Contribution 재정의**
- Main: Multi-UAV task offloading에 A3C 적용
- Sub: 일반화 능력 검증
- Sub: Greedy vs Stochastic 분석

---

## 7. Multi-Seed 실험 계획

### 7.1 main_train_multi_seed.py 분석

**기능:**
- 5개 다른 seed로 반복 학습
- 통계적 유의성 검증 (paired t-test)
- 시각화 (box plot, bar chart, difference plot)

**수집 데이터:**
```python
{
    'seed': seed,
    'a3c_final_reward': a3c_final_reward,
    'individual_final_reward': ind_final_reward,
    'difference': a3c - individual
}
```

**통계 분석:**
- 평균, 표준편차, Min/Max
- Paired t-test → p-value
- 3가지 시각화

### 7.2 개선 필요 사항

**1. Timeout 증가:**
```python
timeout=7200  # 1 hour → 2 hours
```

**2. 평가 지표 개선:**
```python
# 마지막 100 episodes 평균 (마지막 1개 아님)
a3c_final_reward = a3c_summary['reward'].iloc[-100:].mean()
```

**3. 일반화 성능 추가:**
```python
# test_generalization_v2.py 자동 실행
# Seen/Intra/Extra 성능 수집
```

### 7.3 예상 결과

**시나리오 1: A3C 일관되게 우수 (이상적)**
```
A3C: 65 ± 3
Individual: 60 ± 5
p-value < 0.01
→ ✅ Strong contribution
```

**시나리오 2: Individual collapse 빈번 (현재)**
```
A3C: 65 ± 3
Individual: 45 ± 15
p-value < 0.05
→ ⚠️ Weak contribution (학습 실패)
```

**예상 실행 시간:**
```
5 seeds × 10,000 episodes × 1 sec
≈ 14 hours
```

---

## 8. 다음 단계 권장사항

### 8.1 우선순위 1: Individual 학습 개선

**문제:**
- Worker 3 policy collapse (49.7)
- Worker 1-2, 4 value loss 폭발 (457~690)
- Seen 환경에서도 34.5 (낮음)

**해결책:**
```python
# custom_env.py
elif action == DISCARD:
    self.reward -= 1.0  # 작은 패널티 추가
```

**목표:**
- Individual Seen: 60~70
- Policy collapse 방지
- 공정한 비교

### 8.2 우선순위 2: Multi-Seed 실험

**수정 후 실행:**
```bash
# 1. DISCARD penalty 추가
# 2. Timeout 2시간으로 증가
# 3. 평가 지표 개선 (마지막 100 episodes)
python main_train_multi_seed.py
```

**기대 효과:**
- 통계적 유의성 확보
- 재현성 검증
- 논문 신뢰도 증가

### 8.3 우선순위 3: 추가 Baseline

**DQN (individual) 추가:**
- 공정한 비교 대상
- 협력 학습 vs 독립 학습 비교

**예상 결과:**
- A3C > DQN (일반화)
- A3C ≈ DQN (Seen)

### 8.4 장기 계획

**새로운 방법론 제안:**
1. Attention-based A3C
2. Meta-learning + A3C
3. Hybrid approach (A3C + Domain knowledge)

**Target:** Top-tier (JSAC, TWC, INFOCOM)

---

## 9. 핵심 코드 변경 요약

### 9.1 trainer.py

**변경 파일:** [drl_framework/trainer.py](drl_framework/trainer.py)

**주요 변경:**
```python
# Line 310-314: Action masking 제거 (샘플링)
dist = Categorical(logits=logits)  # 원본 사용

# Line 375-377: Action masking 제거 (학습)
dist_seq = Categorical(logits=logits_seq)  # 원본 사용

# Line 295, 703: mask_seq 제거
obs_seq, act_seq, rew_seq, done_seq = [], [], [], []  # mask_seq 제거
```

### 9.2 custom_env.py

**변경 파일:** [drl_framework/custom_env.py](drl_framework/custom_env.py)

**주요 변경:**
```python
# Line 214-262: 모든 즉시 패널티 제거
self.reward = 0
if action == LOCAL:
    # 패널티 없음
if action == DISCARD:
    pass  # 패널티 없음

# Line 277-294: 완료된 comp_units만 보상
self.reward += done_comp  # latency 비용 제거

# Line 305-306: 부분 보상 제거
# 미완료 작업 보상 없음
```

### 9.3 test_generalization_v2.py

**변경 파일:** [test_generalization_v2.py](test_generalization_v2.py)

**주요 변경:**
```python
# Line 19: Timestamp 업데이트
TIMESTAMP = "20251018_185651"

# Line 93: Velocity 범위 수정 (5-20 → 5-25)
SEEN_VELOCITIES = [5, 10, 15, 20, 25]

# Line 96-102: Seen 환경 수정
SEEN_ENVS = [
    (200, 5), (200, 10), (200, 15), (200, 20), (200, 25)
]
```

### 9.4 main_evaluation_fixed.py

**변경 파일:** [main_evaluation_fixed.py](main_evaluation_fixed.py)

**주요 변경:**
```python
# Line 37-38: 학습 환경과 동일하게 설정
e["agent_velocities"] = int(5 + (25 - 5) * i / (n_workers - 1))
```

---

## 10. 실험 결과 파일

### 10.1 학습 결과

**A3C Global:**
- `runs/a3c_20251018_185651/`
- `summary_global.csv`
- 그래프: curves_*.png

**Individual Workers:**
- `runs/individual_20251018_185651/`
- `summary_Individual_*.csv`
- 그래프: curves_*.png

### 10.2 평가 결과

**일반화 성능:**
- `generalization_results_v2_20251018_185651.csv`
- `generalization_test_v2_20251018_185651.png`

---

## 11. 결론

### 11.1 성과

✅ **Action masking 문제 발견 및 해결**
- Policy gradient 불일치 원인 파악
- 완전 제거로 학습-평가 일관성 확보

✅ **Reward 구조 단순화**
- 명확한 목표: 완료된 comp_units 최대화
- 복잡한 패널티 제거

✅ **A3C의 강건함 확인**
- Action masking 없이도 우수한 성능
- 일반화 능력 유지 (Extra: 54.2)

### 11.2 남은 문제

❌ **Individual workers 학습 실패**
- Worker 3 policy collapse (49.7)
- Worker 1-2, 4 value loss 폭발
- Seen 환경에서도 34.5 (낮음)

❌ **DISCARD 억제 부족**
- 패널티 없어서 DISCARD-only policy 학습

❌ **논문 Contribution 약함**
- Individual이 너무 약함 (비교 불공정)
- Baseline 비교 부족

### 11.3 다음 액션

1. **DISCARD penalty 추가** (1.0)
2. **재학습 및 평가**
3. **Multi-seed 실험** (통계적 유의성)
4. **추가 Baseline** (DQN)
5. **논문 작성**

---

## 참고 자료

- 이전 work_summary: `work_summary.md`
- 관련 commit: `b374708` (20251018_023558 학습 시점)
- 논문 draft: `latex/` (있다면)

---

**작성일:** 2025-10-18
**작성자:** Claude Code
**다음 세션 시작 전 확인사항:**
1. DISCARD penalty 추가 여부
2. Multi-seed 실험 실행 여부
3. 논문 작성 시작 여부
