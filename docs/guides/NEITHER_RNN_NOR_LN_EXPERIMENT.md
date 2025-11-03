# Experiment Guide: Neither RNN nor LayerNorm

**Purpose**: Complete the 2×2 architecture matrix by testing feedforward network without LayerNorm

**Status**: 🔴 NOT STARTED

---

## 📋 Experiment Overview

### What We Have

| Configuration | RNN | LayerNorm | Status |
|--------------|-----|-----------|--------|
| Baseline | ✅ | ✅ | ✅ Complete |
| RNN Only | ✅ | ❌ | ✅ Complete |
| LN Only | ❌ | ✅ | ✅ Complete |
| **Neither** | ❌ | ❌ | **🔴 Missing** |

### Why This Matters

**Scientific Value:**
1. Completes 2×2 factorial design
2. Isolates baseline A3C advantage (without architectural tricks)
3. Quantifies synergy between RNN and LayerNorm
4. Tests if worker diversity alone is sufficient

**Expected Insights:**
- Baseline A3C advantage from pure algorithm (no architecture help)
- Interaction effect between RNN and LayerNorm
- Minimum gap achievable (theoretical lower bound)

---

## 🎯 Predictions

### Based on Current Data

**Performance (Mean):**
- A3C: ~53-54 (similar to LN only: 52.94)
- Individual: ~48-49 (slightly better than LN only: 46.76)
- **Expected Gap: ~10-12%**

**Reasoning:**
- Removing LN from "No RNN" configuration
- A3C should improve slightly (~2%, like removing LN from Baseline)
- Individual should improve more (~3-4%, like removing LN from Baseline)
- Net: Gap shrinks by ~2 percentage points

**Stability (CV):**
- A3C: ~0.38-0.40 (worse than all tested configs)
- Individual: ~0.22-0.25 (similar to LN only: 0.217)
- **Expected: Individual more stable**

**Worst-case:**
- A3C: ~31-32 (consistent across configs)
- Individual: ~27-29 (similar to LN only: 29.11)
- **Expected: Similar robustness, both stable**

---

## 🔬 Experimental Design

### Training Configuration

**File**: `ablation_configs.py` (add new config)

```python
ARCHITECTURE_ABLATIONS = {
    # ... existing ablations ...

    'ablation_22_neither_rnn_nor_ln': {
        'name': 'ablation_22_neither_rnn_nor_ln',
        'description': 'Remove both RNN and LayerNorm (pure feedforward)',
        'use_recurrent': False,
        'use_layer_norm': False,
        'phase': 1,
        'priority': 'high',
    },
}
```

**Training Parameters:**
- Seeds: 5 random seeds (42, 123, 456, 789, 1024) for consistency
- Episodes: 2000 per worker (same as baseline)
- Workers: 5 (baseline configuration)
- Network: Feedforward ActorCritic (no GRU, no LayerNorm)
- Hidden dim: 128 (baseline)
- Other hyperparameters: Same as baseline

### Testing Configuration

**Generalization Test:**
- Velocity sweep: 5, 10, 20, 30, 50, 70, 80, 90, 100 km/h (9 velocities)
- Episodes per velocity: 100
- Policy: Greedy (deterministic)
- Same as other ablation tests

---

## 📝 Execution Steps

### Step 1: Add Configuration

Edit `ablation_configs.py`:

```python
# Add to ARCHITECTURE_ABLATIONS dictionary
'ablation_22_neither_rnn_nor_ln': {
    'name': 'ablation_22_neither_rnn_nor_ln',
    'description': 'Remove both RNN and LayerNorm',
    'use_recurrent': False,
    'use_layer_norm': False,
    'phase': 1,
    'priority': 'high',
},
```

### Step 2: Modify params.py

Edit `drl_framework/params.py`:

```python
# Set for this experiment
use_recurrent = False
use_layer_norm = False
target_episode_count = 2000
n_workers = 5
```

Or use environment variables if script supports it.

### Step 3: Run Training (Multiple Seeds)

Create training script `run_neither_experiment.sh`:

```bash
#!/bin/bash

PYTHON_PATH=~/miniconda3/envs/torch-cert/bin/python
SEEDS=(42 123 456 789 1024)
RESULTS_DIR="ablation_results/neither_rnn_nor_ln_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$RESULTS_DIR"

echo "Starting Neither RNN nor LN experiment"
echo "Results will be saved to: $RESULTS_DIR"

for i in "${!SEEDS[@]}"; do
    SEED=${SEEDS[$i]}
    echo "======================================"
    echo "Training seed $SEED ($((i+1))/5)"
    echo "======================================"

    SEED_DIR="$RESULTS_DIR/seed_$SEED"
    mkdir -p "$SEED_DIR"

    # Set random seed and train
    export RANDOM_SEED=$SEED
    $PYTHON_PATH main_train.py

    # Move results to organized location
    mv runs/a3c_* "$SEED_DIR/a3c" 2>/dev/null || true
    mv runs/individual_* "$SEED_DIR/individual" 2>/dev/null || true

    echo "Seed $SEED complete"
done

echo "======================================"
echo "Training complete for all seeds"
echo "Results saved in: $RESULTS_DIR"
echo "======================================"
```

Make executable and run:

```bash
chmod +x run_neither_experiment.sh
./run_neither_experiment.sh
```

**Expected Runtime:** ~8-10 hours (similar to baseline)

### Step 4: Run Generalization Test

Modify `test_baseline_generalization.py` or create new script:

```bash
$PYTHON_PATH test_neither_generalization.py \
    --baseline-dir ablation_results/neither_rnn_nor_ln_YYYYMMDD_HHMMSS \
    --output-csv generalization_results/neither_generalization.csv
```

**Expected Runtime:** ~2-3 hours

### Step 5: Analyze Results

```bash
$PYTHON_PATH analyze_neither_results.py \
    --results-dir ablation_results/neither_rnn_nor_ln_YYYYMMDD_HHMMSS
```

---

## 📊 Analysis Plan

### Metrics to Collect

**Training Performance:**
- Final episode rewards (A3C global, Individual per worker)
- Mean ± Std across 5 seeds
- Coefficient of Variation (CV)

**Generalization Performance:**
- Mean reward across velocity sweep
- Std across velocity sweep
- CV across velocity sweep
- Worst-case performance (minimum across all velocities)

**Comparison with Other Configs:**
- Gap vs Baseline (RNN+LN): Expected ~10-12% vs 29.7%
- Gap vs LN only: Expected ~10-12% vs 13.2%
- Gap vs RNN only: Expected ~10-12% vs 27.8%

### Expected Complete Matrix

|              | With LN | Without LN |
|--------------|---------|------------|
| **With RNN** | 29.7% ⭐ | 27.8% |
| **Without RNN** | 13.2% | **~10-12%** ❓ |

### Statistical Tests

**Compare "Neither" vs other configs:**
- t-test for mean difference
- F-test for variance difference
- Compare worst-case robustness

**Test for interaction effects:**
- Two-way ANOVA: RNN × LayerNorm interaction
- If significant interaction, RNN and LN are synergistic
- If not significant, effects are additive

---

## 🔍 Key Questions to Answer

### Question 1: What is baseline A3C advantage?

**Hypothesis:** Even without RNN or LayerNorm, A3C has ~10-12% advantage

**Test:** Compare "Neither" gap to worker diversity ablations
- 3 workers: +2.2% gap
- 5 workers (Neither): ~10-12% gap (expected)
- This ~10% is pure worker diversity effect

### Question 2: Are RNN and LayerNorm synergistic?

**Test for synergy:**
```
Synergy = (RNN+LN gap) - (RNN only gap) - (LN only gap) + (Neither gap)
Synergy = 29.7 - 27.8 - 13.2 + Neither

If Neither = 11:
Synergy = 29.7 - 27.8 - 13.2 + 11 = -0.3 (no synergy, additive)

If Neither = 8:
Synergy = 29.7 - 27.8 - 13.2 + 8 = -3.3 (negative synergy!)

If Neither = 13:
Synergy = 29.7 - 27.8 - 13.2 + 13 = +1.7 (positive synergy)
```

**Expected:** Near-zero synergy (additive effects)

### Question 3: Is this the "simplest" A3C?

**Hypothesis:** Neither config represents "pure" A3C algorithm
- No architectural tricks
- Only worker diversity and parameter sharing
- Minimal gap (~10-12%)

**Interpretation:**
- RNN adds +16.5 pp (amplifier)
- LayerNorm adds +1.9 pp (stabilizer)
- Base algorithm: ~11 pp
- Total: 11 + 16.5 + 1.9 = 29.4 ≈ 29.7 ✓

### Question 4: Is feedforward + no LN practical?

**Practical considerations:**
- Training stability: Expected to be worst (highest CV)
- Performance: Expected to be best for A3C (~53-54)
- Robustness: Expected to be similar to LN only (stable)

**Verdict:** If stable enough, could be deployment option

---

## 📈 Expected Outcomes

### Outcome 1: Gap ≈ 10-12% (Most Likely)

**Interpretation:**
- Confirms worker diversity as primary source (~10-12%)
- RNN and LN are amplifiers, not sources
- Architecture contributes ~18 pp (29.7 - 11 = 18.7)
- Worker diversity: 37%, Architecture: 63%

**Revises Current Understanding:**
- Current claim: "Worker diversity: 92%"
- This is relative to worker count ablation only
- When including architecture, split is ~40-60%

### Outcome 2: Gap ≈ 5-8% (Lower than Expected)

**Interpretation:**
- Architecture matters more than thought
- RNN+LN contribute >70% of total gap
- Worker diversity alone insufficient
- A3C needs architectural support

**Implication:** Focus paper on RNN+LN importance

### Outcome 3: Gap ≈ 15-18% (Higher than Expected)

**Interpretation:**
- Negative interaction between RNN and LN
- Each hurts performance when combined
- Simpler is better for A3C
- Consider using "Neither" as baseline

**Implication:** Challenge current architectural choices

---

## 💡 Integration with Current Results

### Update Component Contribution Analysis

**Current Analysis (Incomplete):**
- RNN: +16.5 pp (55.5%)
- LayerNorm: +1.9 pp (6.4%)
- ??? (Missing baseline)

**After "Neither" Experiment:**
- Baseline (Neither): ~11 pp (37%)
- RNN: +16.5 pp (55.5%)
- LayerNorm: +1.9 pp (6.4%)
- Synergy: ~0 pp (1%)
- **Total: 11 + 16.5 + 1.9 + 0 = 29.4 ≈ 29.7 ✓**

### Update Figures

**Figure: 2×2 Architecture Matrix (Complete)**
- Show all 4 configurations
- Highlight gradient effects (RNN, LN)
- Show interaction surface

**Table: Complete Architecture Comparison**
- Include "Neither" row
- Show all metrics (mean, CV, worst-case, gap)
- Statistical significance tests

---

## ⏱️ Time Estimate

**Training:**
- 5 seeds × 2000 episodes × ~1 hour/seed
- **Total: 8-10 hours**

**Testing:**
- 10 models (5 A3C + 5 Individual) × 9 velocities × 100 episodes
- **Total: 2-3 hours**

**Analysis:**
- Data aggregation: 30 minutes
- Statistical tests: 1 hour
- Figure generation: 1 hour
- **Total: 2-3 hours**

**Grand Total: ~12-16 hours**

---

## 📝 Documentation Updates Needed

After completing experiment:

1. **Update RNN_LAYERNORM_INTERACTION.md**
   - Fill in "Neither" data
   - Complete 2×2 matrix
   - Calculate interaction effects

2. **Update COMPLETE_ABLATION_RESULTS.md**
   - Add "Neither" configuration
   - Update ranking (likely rank ~17-18)

3. **Update Component Contribution Analysis**
   - Decompose total gap into baseline + RNN + LN + synergy
   - Recalculate percentages

4. **Update Paper Figures**
   - fig1_worker_impact.pdf: May need update if baseline changes
   - New figure: 2×2 architecture matrix (complete)

---

## 🎯 Success Criteria

**Experiment successful if:**
- ✅ All 5 seeds complete training (20,000 episodes total)
- ✅ Generalization test runs on all 10 models
- ✅ Results are statistically consistent (reasonable variance)
- ✅ Gap falls within predicted range (8-15%)

**Analysis complete if:**
- ✅ 2×2 matrix fully populated
- ✅ Interaction effects quantified
- ✅ Component contributions decomposed
- ✅ Figures and tables updated

---

## 🚀 Quick Start Command

```bash
# 1. Create experiment directory
mkdir -p ablation_results/neither_rnn_nor_ln

# 2. Modify params.py
# Set: use_recurrent = False, use_layer_norm = False

# 3. Run training
SEEDS=(42 123 456 789 1024)
for SEED in "${SEEDS[@]}"; do
    export RANDOM_SEED=$SEED
    ~/miniconda3/envs/torch-cert/bin/python main_train.py
done

# 4. Run generalization test
~/miniconda3/envs/torch-cert/bin/python test_baseline_generalization.py \
    --baseline-dir ablation_results/neither_rnn_nor_ln_YYYYMMDD_HHMMSS

# 5. Analyze
~/miniconda3/envs/torch-cert/bin/python analyze_neither_results.py
```

---

**Last Updated**: 2025-11-03
**Status**: 🔴 Experiment not started
**Priority**: HIGH - Completes critical 2×2 matrix
**Estimated Effort**: 12-16 hours total
**Expected Value**: HIGH - Quantifies baseline A3C advantage and interaction effects
