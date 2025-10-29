# Ablation Study 실험 계획

## 개요
UAV A3C 프로젝트의 핵심 컴포넌트별 기여도를 체계적으로 분석하기 위한 ablation study 실험 방안

---

## 1. 네트워크 아키텍처 Ablation

### 1.1 RNN vs Feedforward 비교
- **Baseline**: RecurrentActorCritic (GRU 포함)
- **Ablation-1**: ActorCritic (feedforward만 사용)
- **목적**: 순차적 의사결정에서 RNN의 효과 검증
- **수정 위치**: `main_train.py`의 네트워크 선택 로직

### 1.2 Layer Normalization 효과
- **Baseline**: `use_layer_norm=True`
- **Ablation-2**: `use_layer_norm=False`
- **목적**: Layer Normalization의 학습 안정성 기여도 검증
- **수정 위치**: `drl_framework/params.py:53`
- **현재 설정**: False

### 1.3 Hidden Dimension 크기
- **Baseline**: `hidden_dim=128`
- **Ablation-3**: `hidden_dim=64`
- **Ablation-4**: `hidden_dim=256`
- **목적**: 모델 용량과 성능 간의 관계 분석
- **수정 위치**: `drl_framework/params.py:50`

---

## 2. 학습 하이퍼파라미터 Ablation

### 2.1 Entropy Coefficient (탐색-활용 균형)
- **Baseline**: `entropy_coef=0.05`
- **Ablation-5**: `entropy_coef=0.01` (낮은 탐색)
- **Ablation-6**: `entropy_coef=0.1` (높은 탐색)
- **목적**: 탐색-활용 균형이 학습에 미치는 영향
- **수정 위치**: `drl_framework/params.py:46`

### 2.2 Value Loss Coefficient
- **Baseline**: `value_loss_coef=0.25`
- **Ablation-7**: `value_loss_coef=0.5`
- **Ablation-8**: `value_loss_coef=1.0`
- **목적**: Value 학습 가중치의 영향
- **수정 위치**: `drl_framework/params.py:47`

### 2.3 Learning Rate
- **Baseline**: `lr=1e-4`
- **Ablation-9**: `lr=5e-5`
- **Ablation-10**: `lr=5e-4`
- **목적**: 학습률이 수렴 속도와 안정성에 미치는 영향
- **수정 위치**: `drl_framework/params.py:48`

---

## 3. 환경 설정 Ablation

### 3.1 클라우드 자원 용량
- **Baseline**: `max_comp_units_for_cloud=1000`
- **Ablation-11**: `max_comp_units_for_cloud=500` (제한적 자원)
- **Ablation-12**: `max_comp_units_for_cloud=2000` (풍부한 자원)
- **목적**: 클라우드 자원 가용성이 정책 학습에 미치는 영향
- **수정 위치**: `drl_framework/params.py:27`

### 3.2 UAV 속도
- **Baseline**: `agent_velocities=50`
- **Ablation-13**: `agent_velocities=30` (저속, 안정적 채널)
- **Ablation-14**: `agent_velocities=100` (고속, 불안정 채널)
- **목적**: 채널 품질 변화 속도가 오프로딩 결정에 미치는 영향
- **수정 위치**: `drl_framework/params.py:31`

### 3.3 Worker 수 (병렬성)
- **Baseline**: `n_workers=5`
- **Ablation-15**: `n_workers=3`
- **Ablation-16**: `n_workers=10`
- **목적**: A3C 병렬 학습의 효율성 검증
- **수정 위치**: `drl_framework/params.py:19`

---

## 4. 보상 함수 Ablation

### 4.1 보상 스케일링
- **Baseline**: `REWARD_SCALE=0.05`
- **Ablation-17**: `REWARD_SCALE=0.01`
- **Ablation-18**: `REWARD_SCALE=0.1`
- **목적**: 보상 크기가 학습 안정성에 미치는 영향
- **수정 위치**: `drl_framework/params.py:39`

### 4.2 보상 구성 요소
- **Baseline**: 완료된 computation units만 보상
- **Ablation-19**: 지연 시간 패널티 추가
- **Ablation-20**: 에너지 소비 패널티 추가
- **목적**: 보상 신호의 복잡도와 학습 효율성 분석
- **수정 위치**: `drl_framework/custom_env.py:step()` 함수

---

## 5. Action Masking 효과

### 5.1 Action Masking 유무
- **Baseline**: Action masking 사용
- **Ablation-21**: Action masking 제거 (무효 액션에 큰 패널티)
- **목적**: 유효 액션 제약이 학습 속도와 최종 성능에 미치는 영향
- **수정 위치**: `drl_framework/custom_env.py:123-129`, `drl_framework/trainer.py`

---

## 실험 실행 우선순위

### Phase 1: 네트워크 아키텍처 (가장 중요)
1. Ablation-1: RNN vs Feedforward
2. Ablation-2: Layer Normalization on/off
3. Ablation-3,4: Hidden dimension 변화

### Phase 2: 학습 안정성
4. Ablation-5,6: Entropy coefficient
5. Ablation-7,8: Value loss coefficient
6. Ablation-9,10: Learning rate

### Phase 3: 환경 민감도
7. Ablation-11,12: 클라우드 자원
8. Ablation-13,14: UAV 속도
9. Ablation-15,16: Worker 수

### Phase 4: 보상 설계 (선택적)
10. Ablation-17,18: 보상 스케일링
11. Ablation-19,20: 보상 구성 요소
12. Ablation-21: Action masking

---

## 평가 메트릭

각 ablation 실험은 다음 메트릭으로 평가:

### 1. 학습 효율성
- 에피소드당 평균 보상
- 수렴 속도 (target reward 달성까지의 에피소드 수)
- 학습 안정성 (보상의 분산)
- 학습 곡선의 smoothness

### 2. 최종 성능
- 평가 시 평균 보상 (100 에피소드)
- 로컬/오프로드/폐기 액션 분포
- 작업 완료율
- 평균 대기 시간

### 3. 계산 효율성
- 에피소드당 학습 시간
- 총 학습 시간
- 메모리 사용량

---

## 구현 가이드

### 1. 설정 파일 생성
```python
# ablation_configs.py
ABLATION_CONFIGS = {
    'baseline': {
        'use_recurrent': True,
        'use_layer_norm': True,
        'hidden_dim': 128,
        'entropy_coef': 0.05,
        'value_loss_coef': 0.25,
        'lr': 1e-4,
        # ... 기타 파라미터
    },
    'ablation_1_no_rnn': {
        'use_recurrent': False,
        # 나머지는 baseline과 동일
    },
    'ablation_2_no_layer_norm': {
        'use_layer_norm': False,
    },
    # ... 각 ablation별 설정
}
```

### 2. 배치 실험 스크립트
```python
# run_ablation_experiments.py
for name, config in ABLATION_CONFIGS.items():
    print(f"Running {name}...")
    # params.py 수정 또는 동적으로 파라미터 오버라이드
    # 학습 실행
    # 결과 저장
```

### 3. 결과 분석 및 시각화
- 학습 곡선 비교 플롯
- 성능 메트릭 테이블
- Statistical significance test (t-test, Wilcoxon)

---

## 실험 설정 권장사항

### 학습 설정
- `target_episode_count`: 5000 (충분한 학습)
- 각 ablation당 3-5회 반복 실험 (random seed 변경)
- 평가: 100 에피소드, greedy policy

### 시간 예산
- Phase 1: 가장 중요, 우선 실행
- Phase 2-3: 리소스 허용 시
- Phase 4: 논문 페이지 여유 있을 경우

### 논문 작성 팁
- 각 phase별 섹션 구성
- Baseline 대비 성능 변화율(%) 표시
- 대표적인 학습 곡선 플롯 포함
- Ablation table로 요약 제공

---

## 예상 결과

### 중요도가 높을 것으로 예상되는 컴포넌트
1. **RNN**: 순차적 환경에서 큰 영향 예상
2. **Action Masking**: 무효 액션 방지로 학습 효율성 향상
3. **Entropy Coefficient**: 탐색-활용 균형이 성능에 중요

### 중요도가 낮을 것으로 예상되는 컴포넌트
1. **Layer Normalization**: 작은 네트워크에서는 영향 제한적
2. **보상 스케일링**: 일정 범위 내에서는 비슷한 성능

---

## 참고사항

- 각 실험 결과는 `runs/ablation_{name}_{timestamp}/` 디렉토리에 저장
- 실험 로그는 CSV 형태로 저장하여 후처리 용이하게
- Git branch를 생성하여 실험별로 관리 권장
- 베이스라인 모델은 여러 번 학습하여 신뢰구간 확보
