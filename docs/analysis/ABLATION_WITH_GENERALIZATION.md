# Ablation Study with Generalization Testing

## 핵심 아이디어

**Training 성능 대신 Generalization 성능으로 ablation study 수행**

이전 연구의 핵심 발견:
- A3C와 Individual의 **학습 성능은 비슷** (72.73 vs 46.62)
- 하지만 **일반화 성능과 안정성에서 A3C가 압도적 우세**
- A3C: 완벽한 재현성 (variance = 0.00)
- Individual: 극단적 불안정성 (43 point swing)

---

## Ablation Study 새로운 접근법

### 기존 방식의 문제점:
1. **Training reward만 비교** → 학습 시간 오래 걸림 (2000-5000 episodes)
2. **수렴 여부 불확실** → 짧은 학습으로는 의미 없음
3. **핵심 기여(일반화) 놓침** → Training 성능은 비슷할 수 있음

### 새로운 방식의 장점:
1. **짧은 학습으로도 테스트 가능** (500 episodes면 충분)
2. **일반화 성능에 집중** → 논문의 핵심 기여 강조
3. **다양한 환경 테스트** → 실용성 증명

---

## 실험 프로토콜

### Phase 1: 짧은 학습 (500 episodes)
각 ablation configuration으로:
- A3C Global 모델 학습
- Individual Workers 학습

### Phase 2: Generalization Testing
학습된 모델을 **다양한 환경**에서 테스트:

#### Test Environment 1: 다양한 속도
- 5, 10, 15, 20, 25, 30, 50, 75, 100 km/h
- **목적**: 채널 동역학 변화에 대한 강건성

#### Test Environment 2: 다양한 클라우드 자원
- 500, 750, 1000, 1500, 2000 units
- **목적**: 자원 제약 변화에 대한 적응성

#### Test Environment 3: 다양한 Worker 수
- 3, 5, 7, 10 workers
- **목적**: 경쟁 강도 변화에 대한 확장성

---

## 평가 메트릭

### 1. Generalization Score
- 평균 테스트 성능 (across all test environments)
- **높을수록 좋음**

### 2. Generalization Robustness
- 테스트 성능의 표준편차
- **낮을수록 좋음** (안정적)

### 3. Worst-Case Performance
- 가장 어려운 환경에서의 성능
- **높을수록 좋음** (강건함)

### 4. Performance Drop
- Training vs Test 성능 차이
- **낮을수록 좋음** (일반화 잘됨)

---

## 구현 방안

### 1. 수정된 Ablation Runner

```python
# run_ablation_generalization.py

for ablation_config in ABLATIONS:
    # Step 1: 짧은 학습 (500 episodes)
    train_models(ablation_config, episodes=500)

    # Step 2: Generalization Testing
    for test_env in TEST_ENVIRONMENTS:
        results = test_generalization_v2.py(
            model=trained_model,
            test_env=test_env,
            episodes=100  # 평가용
        )

    # Step 3: 결과 집계
    aggregate_generalization_scores(results)
```

### 2. Test Environment Configurations

```python
TEST_CONFIGS = {
    'velocity_sweep': {
        'velocities': [5, 10, 15, 20, 25, 30, 50, 75, 100],
        'other_params': 'default'
    },
    'cloud_resource_sweep': {
        'max_cloud_units': [500, 750, 1000, 1500, 2000],
        'other_params': 'default'
    },
    'worker_count_sweep': {
        'n_workers': [3, 5, 7, 10],
        'other_params': 'default'
    },
    'mixed_challenge': {
        'velocity': 100,  # 어려운 조합
        'max_cloud_units': 500,
        'n_workers': 10
    }
}
```

---

## 예상 결과 (가설)

### Baseline (A3C with LN + RNN)
- Generalization Score: **높음**
- Robustness: **매우 안정적** (낮은 std)
- Worst-Case: **준수**

### Ablation: No RNN
- Generalization Score: **중간**
- Robustness: **불안정** (높은 std)
- Worst-Case: **나쁨**
- **이유**: 시퀀스 정보 없어서 채널 변화 대응 어려움

### Ablation: No Layer Norm
- Generalization Score: **낮음**
- Robustness: **불안정**
- Worst-Case: **매우 나쁨**
- **이유**: 다양한 입력 스케일에 대한 대응력 부족

### Ablation: Few Workers (3)
- Generalization Score: **높음** (자원 충분)
- Worker Count Test: **10 workers로 테스트 시 성능 급락**
- **이유**: 경쟁 강도 변화에 취약

---

## 실험 시간 예산

### 기존 방식 (Training reward 중심):
- 각 ablation: 2-4시간 (2000 episodes)
- 5 seeds × 21 ablations = **210-420시간**

### 새로운 방식 (Generalization 중심):
- 학습: 20-40분 (500 episodes)
- 테스트: 10-20분 (9개 환경 × 100 episodes)
- 총 **30-60분 per ablation**
- 21 ablations = **10-21시간** ✅

**시간 절약: 10-20배!**

---

## 논문 기여 강화

### 기존 ablation study:
"RNN을 제거하면 학습 성능이 X% 감소한다"
- **약한 기여**: 단순 성능 비교

### Generalization ablation study:
"RNN이 없으면 다양한 속도 환경(5-100 km/h)에서 성능이 평균 40% 감소하며,
특히 고속 환경(>75 km/h)에서는 80% 성능 저하를 보인다.
이는 RNN의 시퀀스 학습 능력이 빠르게 변하는 채널 상태를 추적하는 데
필수적임을 증명한다."
- **강한 기여**: 실용적 통찰 + 메커니즘 설명

---

## 시각화

### Figure 1: Generalization Heatmap
```
              | Vel=5 | Vel=15 | Vel=50 | Vel=100 | Avg | Std
-----------------------------------------------------------------
Baseline      |  72.3 |  73.1  |  69.5  |  65.2   | 70.0| 3.2
No RNN        |  68.1 |  62.3  |  45.2  |  28.1   | 50.9|17.8
No Layer Norm |  65.4 |  58.7  |  42.3  |  25.6   | 48.0|18.5
Small Hidden  |  70.2 |  68.5  |  63.1  |  58.7   | 65.1| 5.1
```
- Color: Green (good) → Red (bad)
- 한눈에 robustness 비교

### Figure 2: Worst-Case Performance
- X축: Ablation configurations
- Y축: Worst-case reward
- Baseline이 압도적으로 높음을 보여줌

### Figure 3: Performance Drop
- X축: Ablation configurations
- Y축: (Training Reward - Average Test Reward)
- Baseline이 가장 낮음 (일반화 잘됨)

---

## 구현 우선순위

### Minimum Viable Product (1일):
1. Baseline + 3개 high priority ablations (No RNN, No LN, Few Workers)
2. 1개 Test sweep (velocity)
3. 기본 집계 및 시각화

### Standard Version (2-3일):
1. Phase 1 전체 (4 ablations)
2. 3개 Test sweeps (velocity, cloud, workers)
3. 전체 메트릭 + 통계 검정

### Full Version (5-7일):
1. 모든 21 ablations
2. 모든 Test sweeps
3. Comprehensive 분석 + 논문용 figures

---

## 다음 단계

1. **`test_generalization_v2.py` 수정**
   - Command-line으로 test environment 지정 가능하게

2. **Generalization test runner 작성**
   - 각 학습된 모델을 여러 환경에서 자동 테스트

3. **결과 집계 스크립트**
   - Heatmap, robustness 차트 자동 생성

4. **MVP 실행**
   - Baseline만 우선 테스트하여 프레임워크 검증

---

## 예상 실행 명령

```bash
# Step 1: 짧은 학습 (이미 완료된 baseline 사용 가능!)
# ablation_results/baseline_20251029_131034/

# Step 2: Generalization testing
python run_generalization_ablation.py \
    --model-dir ablation_results/baseline_20251029_131034/seed_42 \
    --test-config velocity_sweep \
    --output generalization_baseline_seed42.csv

# Step 3: 분석
python analyze_generalization_ablation.py \
    --results-dir ablation_results/ \
    --create-figures
```

---

*Last Updated: 2025-10-29*
