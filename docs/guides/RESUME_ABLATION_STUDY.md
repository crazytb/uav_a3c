# Ablation Study Resume Guide

**Last Updated:** October 29, 2025
**Status:** Baseline experiments completed, ablation variants pending

---

## Current Progress

### ✅ Completed Tasks

1. **Baseline Training (2000 episodes)**
   - 5 seeds: 42, 123, 456, 789, 1024
   - Both A3C and Individual models trained
   - Results: `ablation_results/baseline_20251029_165119/`
   - Training performance: A3C 60.31 vs Individual 57.57 (+4.76%)

2. **Generalization Testing**
   - Velocity sweep: 5-100 km/h (9 velocities)
   - 100 episodes per velocity
   - Results: `generalization_results_2000ep/`
   - **Key Finding: A3C 49.57 vs Individual 38.22 (+29.7%)**

3. **Analysis Documentation**
   - Comprehensive summary: `docs/analysis/BASELINE_EXPERIMENT_SUMMARY.md`
   - Ablation plan: `docs/analysis/ABLATION_STUDY_PLAN.md`
   - Generalization methodology: `docs/analysis/ABLATION_WITH_GENERALIZATION.md`

### ❌ Pending Tasks

**21 Ablation Experiments** organized in 4 phases:

#### Phase 1: Network Architecture (Priority: Critical)
1. ❌ No RNN (feedforward only)
2. ❌ No Layer Normalization
3. ❌ Hidden dimension: 64
4. ❌ Hidden dimension: 256
5. ❌ No GRU (use LSTM)

#### Phase 2: Hyperparameters (Priority: High)
6. ❌ Entropy coefficient: 0.01
7. ❌ Entropy coefficient: 0.1
8. ❌ Value loss coefficient: 0.1
9. ❌ Value loss coefficient: 0.5
10. ❌ Learning rate: 5e-5
11. ❌ Learning rate: 5e-4
12. ❌ Gradient clipping: 0.25
13. ❌ Gradient clipping: 1.0

#### Phase 3: Environment Configuration (Priority: Medium)
14. ❌ Workers: 3
15. ❌ Workers: 10
16. ❌ Cloud resources: 500
17. ❌ Cloud resources: 2000
18. ❌ Velocity: 25 km/h
19. ❌ Velocity: 100 km/h

#### Phase 4: Reward Design (Priority: Low)
20. ❌ No energy penalty
21. ❌ Doubled latency penalty

---

## How to Resume

### Quick Start (Recommended)

Run a single ablation experiment with generalization testing:

```bash
# 1. Choose an ablation from ablation_configs.py
# 2. Run training (500-1000 episodes recommended)
cd /Users/taewonsong/Code/uav_a3c

# Example: No RNN ablation
export ABLATION_NAME="no_rnn"
~/miniconda3/envs/torch-cert/bin/python -c "
from ablation_configs import get_config, apply_config_to_params
config = get_config('no_rnn')
apply_config_to_params(config)
"

# 3. Train with 3-5 seeds
for SEED in 42 123 456; do
    export RANDOM_SEED=$SEED
    ~/miniconda3/envs/torch-cert/bin/python main_train.py
    # Move results to organized directory
done

# 4. Run generalization test
~/miniconda3/envs/torch-cert/bin/python test_baseline_generalization.py \
    --ablation-name no_rnn \
    --baseline-dir ablation_results/no_rnn_YYYYMMDD_HHMMSS
```

### Automated Approach (Needs Fixing)

The automated script `run_ablation_study.py` has dependency issues and needs modification:

**Issues to fix:**
1. Import error: pandas not found
2. Need to use correct Python path
3. Subprocess communication complexity

**Alternative:** Use manual approach above with shell script wrapper

---

## Key Scripts and Their Usage

### 1. Training Scripts

**`main_train.py`**
- Main training script
- Reads configuration from `drl_framework/params.py`
- Environment variables:
  - `RANDOM_SEED`: Set random seed (default: 42)
  - `ABLATION_NAME`: Optional ablation identifier

**`run_baseline_simple.sh`**
- Sequential baseline training with multiple seeds
- Organizes results by seed
- Usage: `./run_baseline_simple.sh`

### 2. Configuration Files

**`drl_framework/params.py`**
- Central configuration file
- Key parameters:
  - `target_episode_count`: Episodes per worker (currently 2000)
  - `n_workers`: Number of parallel workers (currently 5)
  - `device`: CPU/CUDA (currently CPU)
  - `ENV_PARAMS`: Environment settings
  - `REWARD_PARAMS`: Reward coefficients

**`ablation_configs.py`**
- Defines all 21 ablation configurations
- Functions:
  - `get_config(ablation_name)`: Get specific ablation config
  - `apply_config_to_params(config)`: Apply config to params.py

### 3. Evaluation Scripts

**`test_baseline_generalization.py`**
- Tests trained models across velocity sweep
- Key parameters:
  - `baseline_dir`: Directory with trained models
  - `test_velocities`: List of velocities to test
  - `n_episodes`: Episodes per velocity (default: 100)
- Output: CSV with per-velocity performance

**`analyze_baseline_results.py`**
- Aggregates training results across seeds
- Computes statistics and comparison metrics
- Output: Console summary and optional CSV

### 4. Support Scripts

**`run_ablation_study.py`** (NEEDS FIXING)
- Automated ablation execution
- Currently has import errors
- Needs manual approach instead

**`analyze_ablation_results.py`**
- Analyzes completed ablation results
- Compares against baseline
- Generates summary tables

---

## Recommended Execution Strategy

### Option 1: Generalization-Based (Fast, Recommended)

**Time per ablation:** 2-3 hours
**Total time estimate:** ~50 hours for all 21 ablations

```bash
# For each ablation:
# 1. Train 3-5 seeds × 500-1000 episodes (1-2 hours)
# 2. Test on velocity sweep (30-60 minutes)
# 3. Compare generalization score to baseline

# Expected insight: Which components are critical for generalization
```

**Pros:**
- 10-20x faster than full training
- More relevant to deployment scenarios
- Stronger paper contribution

**Cons:**
- May miss some training dynamics insights

### Option 2: Training-Based (Slow, Traditional)

**Time per ablation:** 8-10 hours
**Total time estimate:** ~200 hours for all 21 ablations

```bash
# For each ablation:
# 1. Train 5 seeds × 2000 episodes (8-10 hours)
# 2. Compare final training performance to baseline

# Expected insight: Which components affect training convergence
```

**Pros:**
- Traditional ablation methodology
- Detailed training dynamics analysis

**Cons:**
- Very time-consuming
- Less relevant to real-world deployment

### Option 3: Hybrid (Balanced)

**Priority ablations with full training:** No RNN, No LayerNorm, Worker count (3 ablations)
**Other ablations with quick training:** Remaining 18 ablations

**Total time estimate:** ~60-70 hours

---

## Data Organization

### Directory Structure

```
ablation_results/
├── baseline_20251029_165119/          # Baseline results
│   ├── seed_42/
│   │   ├── a3c/                       # A3C model and logs
│   │   └── individual/                # Individual models and logs
│   ├── seed_123/
│   ├── seed_456/
│   ├── seed_789/
│   └── seed_1024/
├── no_rnn_YYYYMMDD_HHMMSS/           # Future: No RNN ablation
├── no_layer_norm_YYYYMMDD_HHMMSS/    # Future: No LayerNorm ablation
└── ...

generalization_results_2000ep/
├── baseline_2000ep_velocity_generalization.csv
└── (future ablation generalization results)

docs/analysis/
├── BASELINE_EXPERIMENT_SUMMARY.md
├── ABLATION_STUDY_PLAN.md
├── ABLATION_WITH_GENERALIZATION.md
└── (other analysis documents)
```

### Key Data Files

**Training Results:**
- Individual worker metrics: `runs/individual_*/worker_*.csv`
- A3C global metrics: `runs/a3c_*/training_log.csv`
- All combined: `all_training_metrics_*.csv`

**Generalization Results:**
- Velocity sweep: `generalization_results_2000ep/*.csv`
- Columns: seed, worker, velocity, mean_reward, std_reward

---

## Baseline Performance Reference

Use these numbers to compare ablation results:

### Training Performance (2000 episodes)
- **A3C:** 60.31 ± 6.41
- **Individual:** 57.57 ± 4.84
- **Gap:** +2.74 (+4.76%)

### Generalization Performance (Velocity Sweep)
- **A3C:** 49.57 ± 14.35
- **Individual:** 38.22 ± 16.24
- **Gap:** +11.35 (+29.7%)

### Robustness Metrics
- **A3C CV:** 0.290
- **Individual CV:** 0.425
- **A3C Worst-Case:** 31.72
- **Individual Worst-Case:** 1.25

---

## Troubleshooting

### Common Issues

**Issue 1: Import errors (pandas, numpy, etc.)**
```bash
# Use conda environment directly
~/miniconda3/envs/torch-cert/bin/python script.py
```

**Issue 2: PyTorch weights_only error**
```python
# Use weights_only=False in torch.load()
checkpoint = torch.load(path, map_location=device, weights_only=False)
```

**Issue 3: Line ending issues in shell scripts**
```bash
# Fix with sed (macOS)
sed -i '' 's/\r$//' script.sh

# Or use dos2unix (if installed)
dos2unix script.sh
```

**Issue 4: Checkpoint format mismatch**
```python
# Handle both direct state_dict and wrapped checkpoint
checkpoint = torch.load(path, weights_only=False)
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)
```

---

## Next Steps Checklist

Before resuming, decide:

- [ ] Which ablation strategy to use (generalization-based recommended)
- [ ] Which ablations to prioritize (suggest: no_rnn, no_layer_norm, workers_3)
- [ ] How many seeds per ablation (suggest: 3-5 for speed)
- [ ] How many episodes per ablation (suggest: 500-1000 for generalization-based)

Then:

1. [ ] Modify `params.py` if changing episode count
2. [ ] Choose first ablation from `ablation_configs.py`
3. [ ] Run training with selected seeds
4. [ ] Run generalization test
5. [ ] Compare to baseline
6. [ ] Document findings
7. [ ] Repeat for remaining ablations

---

## Expected Paper Contributions

Based on baseline results, ablation study will reveal:

1. **RNN contribution to generalization:** Expected to be significant
2. **Layer normalization impact:** Expected to improve stability
3. **Worker count sweet spot:** Expected optimal around 5-7 workers
4. **Hyperparameter sensitivity:** Expected entropy and value loss coefficients matter
5. **Environment robustness:** Expected velocity and cloud resources affect generalization

**Target paper sections:**
- Introduction: A3C generalization advantage
- Methodology: Generalization-based ablation approach
- Results: 29.7% improvement breakdown by component
- Discussion: Why parameter sharing enables generalization
- Conclusion: Deployment recommendations for real UAV systems

---

**Files to Check Before Resuming:**
1. `drl_framework/params.py` - Verify episode count and other settings
2. `ablation_configs.py` - Review ablation configurations
3. `test_baseline_generalization.py` - Verify test parameters
4. `docs/analysis/BASELINE_EXPERIMENT_SUMMARY.md` - Review baseline results

**Estimated Total Time Remaining:**
- Generalization-based: 50 hours
- Training-based: 200 hours
- Hybrid: 60-70 hours

Good luck! 🚁
