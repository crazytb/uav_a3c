# Ablation Study 실행 가이드

## 개요
이 문서는 체계적인 ablation study 실험을 수행하기 위한 실행 가이드입니다.

---

## 파일 구조

### 1. `ablation_configs.py`
- 모든 ablation 실험의 설정을 정의
- Baseline과 21개의 ablation 설정 포함
- Phase와 priority로 구조화

### 2. `run_ablation_study.py`
- Ablation study 자동 실행 스크립트
- 다중 seed 실험 지원
- 결과 자동 수집 및 분석

---

## 실행 방법

### 기본 실행 명령어

#### 1. 모든 ablation 실행 (시간이 매우 오래 걸림!)
```bash
python run_ablation_study.py --n-seeds 3
```

#### 2. 특정 Phase만 실행 (추천)
```bash
# Phase 1: Network Architecture (가장 중요!)
python run_ablation_study.py --phase 1 --n-seeds 5

# Phase 2: Hyperparameters
python run_ablation_study.py --phase 2 --n-seeds 3

# Phase 3: Environment Configuration
python run_ablation_study.py --phase 3 --n-seeds 3

# Phase 4: Reward Design
python run_ablation_study.py --phase 4 --n-seeds 3
```

#### 3. 특정 Priority만 실행
```bash
# High priority만 (가장 중요한 것들)
python run_ablation_study.py --priority high --n-seeds 5

# Medium priority
python run_ablation_study.py --priority medium --n-seeds 3

# Low priority
python run_ablation_study.py --priority low --n-seeds 3
```

#### 4. 특정 Ablation만 실행
```bash
# 개별 ablation 테스트
python run_ablation_study.py --ablations ablation_1_no_rnn ablation_2_no_layer_norm --n-seeds 5

# Baseline만 실행 (테스트용)
python run_ablation_study.py --ablations baseline --n-seeds 3
```

---

## 추천 실행 순서

### Step 1: Baseline 확립 (먼저 수행)
```bash
# Baseline을 5개 seed로 충분히 실행하여 신뢰구간 확보
python run_ablation_study.py --ablations baseline --n-seeds 5
```
**예상 시간**: 약 5-10시간 (5 seeds × 2 hours/seed)

### Step 2: High Priority Ablations
```bash
# 가장 중요한 컴포넌트들 (RNN, LN, Worker 수)
python run_ablation_study.py --priority high --n-seeds 5
```
**포함 ablations**:
- ablation_1_no_rnn (RNN 제거)
- ablation_2_no_layer_norm (Layer Norm 제거)
- ablation_15_few_workers (3 workers)
- ablation_16_many_workers (10 workers)

**예상 시간**: 약 20-40시간

### Step 3: Phase 1 완성 (Architecture)
```bash
# Hidden dimension 실험 추가
python run_ablation_study.py --ablations ablation_3_small_hidden ablation_4_large_hidden --n-seeds 3
```
**예상 시간**: 약 6-12시간

### Step 4: 시간 여유가 있다면 Phase 2-3
```bash
# Phase 2: Hyperparameters
python run_ablation_study.py --phase 2 --n-seeds 3

# Phase 3: Environment
python run_ablation_study.py --phase 3 --n-seeds 3
```

---

## 결과 파일 구조

```
ablation_results/
└── study_20251029_120000/           # Timestamp
    ├── raw_results.csv              # 모든 실험의 raw data
    ├── summary_results.csv          # 집계된 결과 (mean, std)
    ├── baseline_comparison.csv      # Baseline 대비 성능 변화
    ├── baseline/
    │   ├── config.json
    │   ├── seed_42/
    │   │   ├── a3c/
    │   │   └── individual/
    │   ├── seed_123/
    │   └── seed_456/
    ├── ablation_1_no_rnn/
    │   ├── config.json
    │   ├── seed_42/
    │   ├── seed_123/
    │   └── seed_456/
    └── ...
```

---

## 결과 분석

### 1. Summary 확인
```python
import pandas as pd

# Summary 로드
summary = pd.read_csv('ablation_results/study_TIMESTAMP/summary_results.csv', index_col=0)

# A3C 성능 기준 정렬
summary_sorted = summary.sort_values('a3c_final_reward_mean', ascending=False)
print(summary_sorted[['a3c_final_reward_mean', 'a3c_final_reward_std', 'a3c_advantage_mean']])
```

### 2. Baseline 대비 비교
```python
# Baseline comparison 로드
comparison = pd.read_csv('ablation_results/study_TIMESTAMP/baseline_comparison.csv')

# 가장 큰 성능 향상
print("Top improvements:")
print(comparison.sort_values('a3c_delta_pct', ascending=False).head(5))

# 가장 큰 성능 저하
print("\nTop degradations:")
print(comparison.sort_values('a3c_delta_pct', ascending=True).head(5))
```

### 3. Statistical Significance Test
```python
from scipy import stats

# Raw results 로드
raw = pd.read_csv('ablation_results/study_TIMESTAMP/raw_results.csv')

# Baseline vs Ablation 비교
baseline_rewards = raw[raw['config_name'] == 'baseline']['a3c_final_reward']
ablation_rewards = raw[raw['config_name'] == 'ablation_1_no_rnn']['a3c_final_reward']

t_stat, p_value = stats.ttest_ind(baseline_rewards, ablation_rewards)
print(f"t-statistic: {t_stat:.3f}, p-value: {p_value:.4f}")

if p_value < 0.05:
    print("Statistically significant difference!")
```

---

## 시간 예산 계획

### 최소 계획 (논문 작성 가능 수준)
1. **Baseline** (5 seeds): 10시간
2. **Phase 1 High Priority** (2 ablations × 5 seeds): 20시간
3. **Phase 1 Medium Priority** (2 ablations × 3 seeds): 12시간

**총 예상 시간**: 약 42시간 (2일)
**논문 기여**: Phase 1 완성으로 네트워크 아키텍처 분석 가능

### 표준 계획 (충분한 분석)
1. **Baseline** (5 seeds): 10시간
2. **Phase 1 전체** (4 ablations × 5 seeds): 40시간
3. **Phase 2 High Priority** (2 ablations × 3 seeds): 12시간
4. **Phase 3 High Priority** (2 ablations × 5 seeds): 20시간

**총 예상 시간**: 약 82시간 (3-4일)
**논문 기여**: 네트워크 + 환경 민감도 분석

### 완전 계획 (모든 ablation)
- **All Phases** (21 ablations × 3 seeds): 약 126시간 (5-6일)
- **논문 기여**: Comprehensive ablation study

---

## 주의사항

### 1. 리소스 관리
- 각 실험은 2시간 timeout 설정
- GPU/CPU 사용률 모니터링 필요
- 디스크 공간 충분히 확보 (각 실험당 ~100MB)

### 2. 중간 저장
- 각 실험이 끝날 때마다 결과가 자동 저장됨
- 중단되어도 완료된 실험은 보존됨
- 재시작 시 이미 완료된 실험은 스킵 (수동 체크 필요)

### 3. Params 복원
- 실험 종료 시 자동으로 원본 `params.py` 복원
- 수동 중단 시에는 `params.py.backup`에서 복원:
```bash
cp drl_framework/params.py.backup drl_framework/params.py
```

### 4. 병렬 실행 주의
- 여러 ablation study를 동시에 실행하면 안됨 (params.py 충돌)
- 순차적으로 실행해야 함

---

## 논문 작성 팁

### Table 1: Ablation Study Results (Phase 1)
| Configuration | A3C Reward (Mean±Std) | Individual Reward | Δ vs Baseline | p-value |
|---------------|----------------------|-------------------|---------------|---------|
| Baseline      | 72.73 ± 0.00        | 46.62 ± 8.21      | -             | -       |
| No RNN        | XX.XX ± X.XX        | XX.XX ± X.XX      | -X.XX%        | 0.XXX   |
| No Layer Norm | XX.XX ± X.XX        | XX.XX ± X.XX      | -X.XX%        | 0.XXX   |
| Small Hidden  | XX.XX ± X.XX        | XX.XX ± X.XX      | -X.XX%        | 0.XXX   |
| Large Hidden  | XX.XX ± X.XX        | XX.XX ± X.XX      | +X.XX%        | 0.XXX   |

### Figure: Learning Curves
- Baseline vs Top 3 ablations
- 학습 곡선 비교 플롯
- Confidence interval (mean ± std)

### Figure: Performance Heatmap
- X축: Ablation configuration
- Y축: Metrics (Reward, Value Loss, Policy Loss)
- Color: Performance (녹색=좋음, 빨강=나쁨)

---

## 문제 해결

### 실험이 실패하는 경우
1. `params.py` 복원:
   ```bash
   cp drl_framework/params.py.backup drl_framework/params.py
   ```

2. 특정 ablation만 재실행:
   ```bash
   python run_ablation_study.py --ablations ablation_1_no_rnn --n-seeds 3
   ```

### Timeout 발생
- `run_ablation_study.py`의 `timeout=7200` 값을 증가
- 또는 `target_episode_count`를 줄이기 (params.py)

### 메모리 부족
- `n_workers` 줄이기
- `hidden_dim` 줄이기
- Batch size 조정

---

## 빠른 테스트 (개발용)

실험 스크립트가 제대로 작동하는지 빠르게 테스트:

```bash
# 짧은 episode로 baseline만 테스트
# params.py에서 target_episode_count를 100으로 임시 변경
python run_ablation_study.py --ablations baseline --n-seeds 1
```

**예상 시간**: 약 5-10분

---

## 다음 단계

1. **Step 1**: 위의 "추천 실행 순서"를 따라 실험 수행
2. **Step 2**: 결과 분석 및 시각화 스크립트 작성
3. **Step 3**: 논문 작성용 표와 그래프 생성
4. **Step 4**: Statistical significance test 수행

---

*Last Updated: 2025-10-29*
