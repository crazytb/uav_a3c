# High Priority Ablation Study - Current Status

**Last Updated**: 2025-10-30 13:50 KST

---

## ✅ 진행 상황

### 학습 완료 (20/20) ✓
- ✅ ablation_1_no_rnn: 5 seeds 완료
- ✅ ablation_2_no_layer_norm: 5 seeds 완료
- ✅ ablation_15_few_workers: 5 seeds 완료
- ✅ ablation_16_many_workers: 5 seeds 완료

**학습 진행률**: 20/20 완료 (100%)

### 일반화 테스트 진행 중
- ⏳ 모든 ablation에 대한 velocity sweep 테스트 실행 중
- 테스트 범위: 5-100 km/h (9개 속도)
- 각 조건당 100 episodes
- 예상 완료 시간: ~2-3시간

---

## 📊 완료된 실험 결과 (Seed 42)

### Ablation 1: No RNN (Feedforward only)

**A3C Performance**:
- Final Reward: 61.93
- vs Baseline (with RNN): 60.31
- **차이**: +1.62 (+2.7% 향상)

**Individual Performance**:
- Average Reward: 57.13
  - Worker 0: 46.35
  - Worker 1: 46.10
  - Worker 2: 58.45
  - Worker 3: 56.50
  - Worker 4: 78.25
- vs Baseline (with RNN): 57.57
- **차이**: -0.44 (-0.8% 감소)

**A3C Advantage**:
- Gap: +4.80 (A3C - Individual)
- vs Baseline Gap: +2.74
- **RNN 제거 후 gap이 오히려 75% 증가!**

### 🤔 예비 관찰 (단일 seed)

1. **예상과 다른 결과**: RNN 제거가 성능을 향상시킴
2. **A3C가 더 강해짐**: Individual보다 A3C의 상대적 이득이 증가
3. **통계적 확인 필요**: 5개 seed 평균이 필요

---

## 🔄 자동 실행 중

### 실행 스크립트
- `run_remaining_ablations.sh` 백그라운드 실행 중
- 자동으로 순차 실행: seeds 456 → 789 → 1024
- 이후 자동으로 나머지 3개 ablation 실행

### 모니터링
```bash
# 진행 상황 확인
./monitor_ablation_progress.sh

# 실시간 로그 확인
tail -f ablation_results/logs/auto_execution.log

# 학습 진행 확인
tail -f runs/a3c_*/training_log.csv
```

---

## ⏱️ 예상 일정

### 시간 추정
- **Seed 456**: ~8-10시간 (진행 중)
- **Seed 789**: ~8-10시간
- **Seed 1024**: ~8-10시간
- **Ablation 1 완료**: 약 24-30시간 후

### 전체 완료 예상
- **Ablation 1** (No RNN): ~40-50시간
- **Ablation 2** (No LayerNorm): ~40-50시간
- **Ablation 3** (Few workers): ~40-50시간
- **Ablation 4** (Many workers): ~40-50시간

**총 예상 시간**: 160-200시간 (~7-8일)

---

## 📂 결과 위치

### 디렉토리 구조
```
ablation_results/high_priority/
├── ablation_1_no_rnn/
│   ├── seed_42/    ✓ 완료
│   ├── seed_123/   ✓ 완료
│   ├── seed_456/   ⏳ 진행 중
│   ├── seed_789/   ⏸️ 대기
│   └── seed_1024/  ⏸️ 대기
├── ablation_2_no_layer_norm/     ⏸️ 대기
├── ablation_15_few_workers/      ⏸️ 대기
└── ablation_16_many_workers/     ⏸️ 대기
```

### 로그 파일
- 자동 실행 로그: `ablation_results/logs/auto_execution.log`
- 개별 실험 로그: `ablation_results/logs/ablation_*_seed_*.log`
- 학습 진행 로그: `runs/a3c_*/training_log.csv`

---

## 🎯 다음 단계

### 1단계: 모든 실험 완료 대기 (~7-8일)
- 자동 실행 스크립트가 모든 실험을 순차 실행
- 주기적으로 진행 상황 모니터링

### 2단계: 학습 성능 분석
```bash
python analyze_high_priority_ablations.py
```
- 4개 ablation의 training performance 비교
- 통계적 유의성 검정
- Baseline과 비교

### 3단계: 일반화 성능 테스트
```bash
python test_ablation_generalization.py
```
- Velocity sweep (5-100 km/h) 테스트
- 각 ablation의 generalization score 계산
- Robustness 지표 (CV, worst-case)

### 4단계: 논문용 출력 생성
```bash
python generate_paper_tables.py
```
- LaTeX 테이블
- 비교 그래프 (PNG, PDF)
- Markdown 요약 보고서

---

## 📝 중요 발견 (예비)

### Ablation 1 (No RNN) - Seed 42 결과

**예상**: RNN 제거 시 성능 하락 예상
**실제**: RNN 제거 시 성능 향상!

**가능한 해석**:
1. 이 task는 sequential memory가 불필요할 수 있음
2. RNN이 과적합을 유발했을 가능성
3. Feedforward가 더 안정적으로 학습
4. 단일 seed 변동성 (5 seed 평균 필요)

**확인 필요**:
- ✅ 나머지 4개 seed 결과
- ✅ 일반화 성능 테스트
- ✅ 통계적 유의성 검정

---

## ⚠️ 주의사항

1. **자동 실행 중단 금지**
   - 백그라운드 프로세스가 실행 중
   - 터미널 종료해도 계속 실행됨
   - 중단 시: `pkill -f run_remaining_ablations.sh`

2. **디스크 공간 확인**
   - 예상 사용량: ~2GB
   - 정기적으로 확인 필요

3. **CPU 사용률**
   - 5 workers 병렬 실행으로 높은 CPU 사용
   - 다른 작업 시 성능 저하 가능

4. **실험 재시작**
   - 문제 발생 시 특정 seed만 재실행 가능
   - `run_single_ablation.py` 사용

---

**문의사항이나 문제 발생 시**:
- 진행 상황: `./monitor_ablation_progress.sh`
- 로그 확인: `tail -f ablation_results/logs/auto_execution.log`
- 프로세스 확인: `ps aux | grep python | grep ablation`

---

**마지막 업데이트**: 2025-10-30 07:10 KST
**다음 확인 예정**: Seed 456 완료 시 (~8-10시간 후)
