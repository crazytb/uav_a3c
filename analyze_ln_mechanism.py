"""
네트워크 입출력 구조를 분석하여 Layer Normalization이 왜 효과적인지 파악

핵심 분석:
1. 입력 데이터의 특성
2. 네트워크의 정보 흐름
3. Value Loss 폭발 원인
4. Layer Normalization의 작동 메커니즘
"""

print("=" * 80)
print("Layer Normalization 효과 분석: 네트워크 구조 기반")
print("=" * 80)

# ===== 1. 입력 데이터 분석 =====
print("\n" + "=" * 80)
print("1. 입력 데이터 특성 분석")
print("=" * 80)

observation_components = {
    "available_computation_units": {"shape": (1,), "range": "[0, 1]", "type": "continuous"},
    "remain_epochs": {"shape": (1,), "range": "[0, 1]", "type": "continuous"},
    "mec_comp_units": {"shape": (20,), "range": "[0, 1]", "type": "continuous"},
    "mec_proc_times": {"shape": (20,), "range": "[0, 1]", "type": "continuous"},
    "queue_comp_units": {"shape": (1,), "range": "[0, 1]", "type": "continuous"},
    "queue_proc_times": {"shape": (1,), "range": "[0, 1]", "type": "continuous"},
    "local_success": {"shape": (1,), "range": "{0, 1}", "type": "discrete"},
    "offload_success": {"shape": (1,), "range": "{0, 1}", "type": "discrete"},
    "ctx_vel": {"shape": (1,), "range": "[0, 1]", "type": "continuous"},
    "ctx_comp": {"shape": (1,), "range": "[0, 1]", "type": "continuous"},
}

total_state_dim = sum(c["shape"][0] for c in observation_components.values())

print(f"\n📊 State Space 구성:")
print(f"  총 차원: {total_state_dim}차원")
print(f"\n  구성 요소:")
for name, info in observation_components.items():
    print(f"    - {name:30s}: {str(info['shape']):8s} {info['range']:12s} ({info['type']})")

print(f"\n🔍 입력 데이터의 문제점:")
print(f"  1. 혼합된 스케일: ")
print(f"     - 큐 정보 (20차원): 동적으로 변화, 대부분 0 또는 작은 값")
print(f"     - 컨텍스트 정보 (2차원): 고정값 (velocity, comp_units)")
print(f"     - 이산 플래그 (2차원): {{0, 1}}")
print(f"  2. 희소성 (Sparsity):")
print(f"     - mec_comp_units, mec_proc_times: 대부분 0 (빈 큐)")
print(f"     - 에피소드 초반 vs 후반에 패턴 크게 변화")
print(f"  3. 시간적 변동성:")
print(f"     - available_computation_units: 매 스텝 크게 변화 (0~200)")
print(f"     - remain_epochs: 단조 감소 (100→0)")

# ===== 2. 네트워크 정보 흐름 분석 =====
print("\n" + "=" * 80)
print("2. 네트워크 정보 흐름 (RecurrentActorCritic)")
print("=" * 80)

print(f"""
[입력] State (50차원)
   ↓
[Feature Extraction] Linear(50 → 128)
   ↓ z = W·x + b
   ↓ **문제점: W의 각 행이 50개 입력의 선형 결합**
   ↓            **입력 스케일 불균형 → 출력 스케일 폭발 가능**
   ↓
[LayerNorm] (있으면) → z_normalized
   ↓ **효과: 128차원 각각을 평균 0, 분산 1로 정규화**
   ↓        **입력 불균형의 영향을 제거**
   ↓
[ReLU] → z_activated
   ↓
[GRU] GRU(128 → 128) + hidden state
   ↓ h_t = GRU(z, h_{t-1})
   ↓ **문제점: h_{t-1}이 커지면 h_t도 폭발**
   ↓            **gradient 폭발/소실 가능**
   ↓
[LayerNorm] (있으면) → h_normalized
   ↓ **효과: RNN 출력을 안정화**
   ↓        **Value head 입력의 스케일 제어**
   ↓
[Value Head] Linear(128 → 1)
   ↓ value = W_v·h + b_v
   ↓ **문제점: h의 스케일이 크면 value 폭발!**
   ↓
[MSE Loss] (value - target)²
   ↓ **문제점: value가 크면 Loss 폭발!**
   ↓            **gradient 폭발 → 학습 불안정**
""")

# ===== 3. Value Loss 폭발 메커니즘 =====
print("\n" + "=" * 80)
print("3. Value Loss 폭발 메커니즘 (Layer Norm 없을 때)")
print("=" * 80)

print(f"""
🔥 Value Loss 폭발의 연쇄 반응:

Step 1: 입력 불균형
  - 큐가 꽉 찼을 때: mec_comp_units = [0.8, 0.9, ..., 0.7] (큰 값들)
  - 큐가 비었을 때: mec_comp_units = [0, 0, ..., 0] (모두 0)
  ↓

Step 2: Feature Extraction 출력 폭발
  - z = W·x + b
  - 입력이 크면 → z의 일부 차원이 매우 큰 값
  - 예: z[i] ∈ [-50, 50] (정규화 없으면)
  ↓

Step 3: ReLU 후에도 큰 값 유지
  - ReLU(z) → 음수는 0이 되지만 양수는 그대로
  - 큰 값들이 GRU로 전달
  ↓

Step 4: GRU Hidden State 누적
  - h_t = GRU(큰 입력, h_{t-1})
  - 시퀀스가 길어질수록 h_t가 점점 커짐
  - 특히 Individual: 워커마다 독립 → 안정화 메커니즘 없음
  ↓

Step 5: Value 예측 폭발
  - value = W_v·(큰 h) + b_v
  - h가 크면 value도 매우 큰 값 (예: ±500)
  ↓

Step 6: MSE Loss 폭발
  - loss = (value - target)²
  - value=500, target=60 → loss = 193,600!
  - 실험 결과: Individual without LN의 value loss 최대 1198.3
  ↓

Step 7: Gradient 폭발
  - ∂loss/∂W = 2·(value - target)·h
  - loss가 크면 gradient도 매우 큼
  - Gradient clipping으로 완화하지만 근본 해결 안 됨
""")

# ===== 4. Layer Normalization의 작동 메커니즘 =====
print("\n" + "=" * 80)
print("4. Layer Normalization의 작동 메커니즘")
print("=" * 80)

print(f"""
✅ Layer Normalization이 문제를 해결하는 방법:

[위치 1] Feature Extraction 후:
-----------------------------------
  z = Linear(x)  # (B, 128)
  ↓
  z_norm = LayerNorm(z)
  ↓
  각 샘플에 대해:
    mean_z = mean(z, dim=-1)  # 128차원의 평균
    std_z = std(z, dim=-1)    # 128차원의 표준편차
    z_norm = (z - mean_z) / (std_z + ε)
  ↓
  **효과:**
    - 입력이 아무리 불균형해도 z_norm은 평균 0, 분산 1
    - ReLU 입력이 안정적인 범위 유지
    - GRU에 전달되는 값의 스케일 제어

  **실험 결과:**
    - Feature 출력: mean ≈ 0, std ≈ 1 (정규화됨)
    - Without LN: mean ≈ random, std >> 1 (불안정)

[위치 2] GRU 출력 후:
-----------------------------------
  h = GRU(z, h_prev)  # (B, 128)
  ↓
  h_norm = LayerNorm(h)
  ↓
  **효과:**
    - RNN hidden state의 누적 효과 제한
    - Value head 입력이 항상 안정적인 범위
    - value = W_v·h_norm + b_v → value도 안정적

  **실험 결과:**
    - RNN 출력: mean ≈ 0, std ≈ 1
    - Value Loss 평균:
      * A3C with LN: 30.4
      * A3C without LN: 78.9 (2.6배 차이)
      * Individual with LN: 34.2
      * Individual without LN: 386.4 (11.3배 차이!)
""")

# ===== 5. A3C vs Individual 차이 분석 =====
print("\n" + "=" * 80)
print("5. 왜 Individual이 Layer Norm을 더 필요로 하는가?")
print("=" * 80)

print(f"""
🔄 A3C의 자연스러운 안정화 메커니즘:
-----------------------------------
  1. Gradient Averaging:
     - 5개 워커의 gradient를 평균
     - 극단적인 gradient가 상쇄됨
     - 예: Worker A의 큰 positive gradient + Worker B의 큰 negative gradient → 평균은 작음

  2. Global Model Sharing:
     - 모든 워커가 동일한 파라미터 사용
     - 한 워커의 폭발이 전체에 영향 주기 전에 다른 워커들이 보정

  3. 암묵적 Regularization:
     - 여러 환경에서 동시 학습
     - 과적합 방지

  **결과:**
    - Value Loss without LN: 78.9 (나쁘지만 학습 가능)
    - Explosion 비율: 34% (감당 가능)

❌ Individual의 문제점:
-----------------------------------
  1. 독립 학습:
     - 각 워커가 완전히 독립적으로 학습
     - Gradient averaging 효과 없음
     - 한 번 폭발하면 복구 어려움

  2. 환경 특수화:
     - 각 워커가 특정 velocity만 경험
     - 편향된 학습 → 극단적인 파라미터

  3. 안정화 메커니즘 부재:
     - Layer Norm 없으면 폭발을 막을 방법이 없음!

  **결과:**
    - Value Loss without LN: 386.4 (A3C의 4.9배!)
    - Explosion 비율: 96.6% (거의 모든 에피소드!)
    - Layer Norm 적용 시: 34.2로 급감 (91% 개선)
""")

# ===== 6. 정량적 분석 =====
print("\n" + "=" * 80)
print("6. 정량적 효과 분석")
print("=" * 80)

import pandas as pd

data = {
    "Model": ["A3C w/ LN", "A3C w/o LN", "Individual w/ LN", "Individual w/o LN"],
    "Value Loss": [30.4, 78.9, 34.2, 386.4],
    "Max Loss": [538.4, 850.2, 959.8, 1198.3],
    "Explosion %": [27.3, 34.0, 29.6, 96.6],
    "Reward": [62.72, 57.23, 57.05, 56.06]
}

df = pd.DataFrame(data)
print("\n" + df.to_string(index=False))

print(f"""

📊 Layer Norm의 효과 크기:

  A3C:
    - Value Loss 감소: 61.5%
    - Explosion 감소: 19.9%
    - Reward 향상: 9.6%
    → 안정화 + 성능 향상

  Individual:
    - Value Loss 감소: 91.1% ⭐⭐⭐
    - Explosion 감소: 69.4% ⭐⭐⭐
    - Reward 향상: 1.8%
    → 필수적인 안정화 (성능은 부차적)
""")

# ===== 7. 결론 =====
print("\n" + "=" * 80)
print("7. 종합 결론")
print("=" * 80)

print(f"""
🎯 Layer Normalization이 효과적인 이유 (네트워크 구조 기반):

1. **입력 불균형 문제 해결**
   - 50차원 state의 스케일이 제각각
   - Feature extraction 출력을 정규화 → ReLU 입력 안정화

2. **RNN Hidden State 폭발 방지**
   - GRU는 시퀀스 길이만큼 정보를 누적
   - LayerNorm이 각 스텝의 출력을 제한 → 누적 폭발 방지

3. **Value Head 입력 안정화**
   - Value = Linear(RNN output)
   - RNN 출력이 안정적 → Value 예측 안정적 → MSE Loss 안정적

4. **Gradient Flow 개선**
   - Loss가 작으면 gradient도 작고 안정적
   - Backpropagation through time (BPTT)에서 특히 중요

5. **모델 독립성에 따른 차별적 효과**
   - A3C: Gradient averaging으로 어느 정도 안정 → LN은 성능 향상
   - Individual: 안정화 메커니즘 전무 → LN은 필수 (91% Loss 감소!)

💡 **핵심 인사이트:**
   Layer Normalization은 단순히 "좋은 테크닉"이 아니라,
   RNN 기반 RL에서 **Value Loss 폭발을 막는 필수 안전장치**입니다.
   특히 Independent Learning에서는 **생존 필수 조건**입니다.
""")

print("\n" + "=" * 80)
print("분석 완료!")
print("=" * 80)
