# High Priority Ablation Execution Guide

## ✅ 실행 시작됨!

**시작 시간**: 2025-10-30 00:38
**현재 상태**: ablation_1_no_rnn (seed 42) 학습 중

---

## 🎯 실행 계획

### 총 실험 수: 20개
- 4개 ablation × 5 seeds = 20 experiments

### Ablation 목록 (실행 순서)

1. **ablation_1_no_rnn** (No RNN - Feedforward only)
   - Seeds: 42, 123, 456, 789, 1024
   - 예상 시간: ~40-50시간
   - 상태: ⏳ seed 42 진행 중

2. **ablation_2_no_layer_norm** (No Layer Normalization)
   - Seeds: 42, 123, 456, 789, 1024
   - 예상 시간: ~40-50시간
   - 상태: ⏸️ 대기 중

3. **ablation_15_few_workers** (3 workers instead of 5)
   - Seeds: 42, 123, 456, 789, 1024
   - 예상 시간: ~40-50시간
   - 상태: ⏸️ 대기 중

4. **ablation_16_many_workers** (10 workers instead of 5)
   - Seeds: 42, 123, 456, 789, 1024
   - 예상 시간: ~40-50시간
   - 상태: ⏸️ 대기 중

**총 예상 시간**: 160-200시간 (~7-8일)

---

## 📊 진행 상황 모니터링

### 1. 빠른 상태 확인
```bash
./monitor_ablation_progress.sh
```

### 2. 실시간 학습 로그 확인
```bash
# 최신 A3C 학습 진행 상황
tail -f runs/a3c_*/training_log.csv

# 최신 Individual 학습 진행 상황
tail -f runs/individual_*/worker_*.csv
```

### 3. 백그라운드 프로세스 확인
```bash
ps aux | grep "run_single_ablation.py\|main_train.py" | grep -v grep
```

---

## 🔄 실행 방법

### 현재 실행 중인 작업
```bash
# ablation_1_no_rnn, seed 42가 백그라운드에서 실행 중
# PID 확인:
ps aux | grep run_single_ablation.py | grep -v grep
```

### 다음 실험 자동 실행 (한번에 전체)

**방법 1: 전체 자동 실행 스크립트**
```bash
./run_high_priority_ablations.sh
```

**방법 2: 개별 실행**
```bash
# 각 ablation과 seed를 개별적으로 실행
/Users/crazytb/miniconda/envs/torch-cert/bin/python run_single_ablation.py \
    --ablation ablation_1_no_rnn \
    --seed 123 \
    --output-dir ablation_results/high_priority
```

---

## 📂 결과 구조

```
ablation_results/high_priority/
├── ablation_1_no_rnn/
│   ├── seed_42/          ← 현재 진행 중
│   │   ├── a3c/
│   │   │   ├── models/global_final.pth
│   │   │   └── training_log.csv
│   │   ├── individual/
│   │   │   ├── models/individual_worker_*_final.pth
│   │   │   └── worker_*.csv
│   │   └── config.txt
│   ├── seed_123/         ← 대기
│   ├── seed_456/         ← 대기
│   ├── seed_789/         ← 대기
│   └── seed_1024/        ← 대기
├── ablation_2_no_layer_norm/  ← 대기
├── ablation_15_few_workers/   ← 대기
└── ablation_16_many_workers/  ← 대기
```

---

## ⚠️ 중요 사항

### 1. 백그라운드 실행
- 현재 실험은 백그라운드에서 실행 중입니다
- 터미널을 종료해도 계속 실행됩니다
- 중단하려면: `pkill -f run_single_ablation.py`

### 2. 디스크 공간
- 각 seed는 약 50-100MB의 공간을 사용합니다
- 총 예상 사용량: ~1-2GB

### 3. CPU 사용량
- 5 workers가 병렬로 실행되므로 CPU 사용률이 높습니다
- 다른 작업을 동시에 수행할 경우 성능이 저하될 수 있습니다

### 4. 실험 중단 시
- 중단된 실험은 자동으로 재개되지 않습니다
- 특정 seed부터 다시 시작하려면 해당 seed만 개별적으로 실행하세요

---

## 🔍 문제 해결

### 프로세스가 멈춘 것 같을 때
```bash
# 프로세스 확인
ps aux | grep python | grep -E "run_single_ablation|main_train"

# 최신 runs 디렉토리 확인
ls -lht runs/ | head -5

# 학습 로그 마지막 줄 확인
tail runs/a3c_*/training_log.csv
```

### 재시작이 필요한 경우
```bash
# 1. 현재 프로세스 종료
pkill -f run_single_ablation.py
pkill -f main_train.py

# 2. 특정 ablation/seed 재실행
/Users/crazytb/miniconda/envs/torch-cert/bin/python run_single_ablation.py \
    --ablation ablation_1_no_rnn \
    --seed 42 \
    --output-dir ablation_results/high_priority
```

---

## 📈 완료 후 분석

### 1. 학습 성능 분석
```bash
/Users/crazytb/miniconda/envs/torch-cert/bin/python analyze_high_priority_ablations.py
```

### 2. 일반화 성능 테스트
```bash
/Users/crazytb/miniconda/envs/torch-cert/bin/python test_ablation_generalization.py
```

### 3. 논문용 테이블/그래프 생성
```bash
/Users/crazytb/miniconda/envs/torch-cert/bin/python generate_paper_tables.py
```

---

## 📝 체크리스트

### 진행 중
- [x] 디렉토리 구조 생성
- [x] 실행 스크립트 준비
- [x] ablation_1_no_rnn, seed 42 시작
- [ ] ablation_1_no_rnn, 나머지 seeds
- [ ] ablation_2_no_layer_norm, 모든 seeds
- [ ] ablation_15_few_workers, 모든 seeds
- [ ] ablation_16_many_workers, 모든 seeds

### 완료 후
- [ ] 학습 성능 분석
- [ ] 일반화 성능 테스트
- [ ] 논문용 테이블/그래프 생성
- [ ] ABLATION_SUMMARY_REPORT.md 작성

---

## 📞 도움말

**진행 상황 확인**: `./monitor_ablation_progress.sh`
**로그 파일**: `ablation_results/logs/`
**결과 디렉토리**: `ablation_results/high_priority/`

---

**마지막 업데이트**: 2025-10-30 00:40
