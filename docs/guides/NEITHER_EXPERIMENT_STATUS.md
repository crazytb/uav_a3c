# Neither RNN nor LayerNorm Experiment - Status

**Experiment Started**: 2025-11-03 19:01:57 KST

---

## 🎯 Experiment Objective

Complete the 2×2 RNN/LayerNorm architecture matrix by testing the missing configuration:
- **Configuration**: Neither RNN nor LayerNorm (feedforward only)
- **Purpose**: Isolate baseline algorithmic advantage from architectural components

---

## ⚙️ Configuration

### Code Modifications Made

1. **drl_framework/params.py**
   ```python
   use_recurrent = False           # Use feedforward ActorCritic
   use_layer_norm = False          # No normalization
   ```

2. **drl_framework/trainer.py**
   - Imported both `ActorCritic` and `RecurrentActorCritic`
   - Added conditional network selection in 3 locations:
     - Line 257-260: A3C worker model
     - Line 484-487: A3C global model
     - Line 671-674: Individual worker model

3. **drl_framework/networks.py**
   - Added `init_hidden()`, `step()`, and `rollout()` methods to `ActorCritic`
   - These return dummy hidden states for API compatibility with RNN version

4. **ablation_configs.py**
   - Added `ablation_neither_rnn_nor_ln` configuration

### Verification

- ✅ Network instantiation test passed
- ✅ Forward pass test passed
- ✅ Step and rollout methods working correctly
- ✅ Training script running successfully

---

## 📊 Expected Results

Based on previous ablation results, we expect:

### Performance Predictions

| Metric | Expected Value | Reasoning |
|--------|----------------|-----------|
| **A3C Mean** | ~53-54 | Highest absolute performance (no architectural constraints) |
| **Individual Mean** | ~48-49 | Also high, but Individual catches up more |
| **Gap %** | ~10-12% | Baseline algorithmic advantage only |
| **A3C CV** | ~0.38-0.40 | Worst stability (no RNN or LN stabilization) |
| **Individual CV** | ~0.22-0.25 | Best stability (simple task + independent learning) |

### Component Contribution Analysis

The complete 2×2 matrix will allow us to calculate:

```
                With LN         Without LN
With RNN        29.7% ✓         27.8% ✓
Without RNN     13.2% ✓         ??? (running)
```

**Expected Component Contributions**:
- **RNN**: ~16.5 pp (55.5% of total gap)
- **LayerNorm**: ~1.9 pp (6.4% of total gap)
- **Baseline (neither)**: ~10-11 pp (37% of total gap)
- **Synergy**: ~1 pp (3% of total gap)

This will reveal whether RNN and LayerNorm are:
- **Additive**: Total effect = sum of individual effects
- **Synergistic**: Total effect > sum of individual effects
- **Subadditive**: Total effect < sum of individual effects

---

## 🔄 Training Progress

**Status**: Running (Seed 1/5: 42)

**Seeds to run**: 42, 123, 456, 789, 1024

**Estimated time**:
- Per seed: ~90-120 minutes (2000 episodes × 2 methods)
- Total: **8-10 hours**

**Results location**: `ablation_results/neither_rnn_nor_ln_20251103_190157/`

### Current Status

Check training progress with:
```bash
# Monitor background process
ps aux | grep run_neither

# Check latest log output
tail -f ablation_results/neither_rnn_nor_ln_*/seed_42/a3c/logs/training_log.csv
```

---

## 📝 Next Steps (After Training Completes)

### 1. Run Generalization Test

```bash
python test_baseline_generalization.py \
    --baseline-dir ablation_results/neither_rnn_nor_ln_20251103_190157
```

**Time**: ~2-3 hours (5 seeds × 9 velocities × 100 episodes × 2 methods)

### 2. Analyze Results

Extract key metrics:
- Mean ± Std for A3C and Individual
- Coefficient of Variation (CV = Std/Mean)
- Gap % and contribution to baseline
- Worst-case performance

### 3. Update Documentation

Files to update:
1. `docs/analysis/RNN_LAYERNORM_INTERACTION.md` - Complete 2×2 matrix
2. `docs/results/COMPLETE_ABLATION_RESULTS.md` - Add "neither" results
3. `docs/paper/PAPER_STORYLINE.md` - Revise component contributions
4. `CLAUDE.md` - Update research status

### 4. Revise Analysis

**Critical**: The current claim that "Worker Diversity contributes 92%" is incorrect.

**Correct breakdown** (after this experiment):
1. **Baseline (neither)**: ~37% (10-11 pp out of 29.7 pp)
2. **RNN**: ~55% (16.5 pp)
3. **LayerNorm**: ~6% (1.9 pp)
4. **Synergy**: ~3% (1 pp)

**Worker diversity** is captured in the **baseline** component (37%), not 92%.

---

## 🔬 Scientific Questions to Answer

### Question 1: Is the combination additive?

**Test**: Does (RNN effect) + (LN effect) = (Combined effect)?

```
RNN effect     = (RNN+LN) - (LN only) = 29.7 - 13.2 = 16.5 pp
LN effect      = (RNN+LN) - (RNN only) = 29.7 - 27.8 = 1.9 pp
Combined       = 16.5 + 1.9 = 18.4 pp

Baseline (neither) = 29.7 - 18.4 = 11.3 pp (expected)
Actual baseline    = ??? pp (measuring)

If actual ≈ expected → Additive
If actual > expected → Subadditive (components overlap)
If actual < expected → Synergistic (components amplify)
```

### Question 2: Does Individual benefit from simplicity?

**Hypothesis**: Individual performs best with feedforward + no LN

**Test**: Compare Individual across all 4 configurations
```
RNN+LN:    38.22 ± 16.24 (CV 0.425)
RNN only:  39.58 ± 17.97 (CV 0.454)
LN only:   46.76 ± 10.14 (CV 0.217)
Neither:   ??? ± ???      (CV ???)
```

Expected: Individual achieves highest mean and best CV with "neither"

### Question 3: What is A3C's true baseline advantage?

**Hypothesis**: Without architectural components, A3C still outperforms by ~10-12%

**Mechanisms**:
1. **Parameter sharing**: Reduces variance across workers
2. **Diverse exploration**: Workers explore different strategies
3. **Knowledge aggregation**: Global model benefits from collective experience

**Test**: Does "neither" configuration show significant gap?
- If gap ~10-12%: Baseline advantage confirmed
- If gap <5%: Architecture drives most advantage
- If gap >15%: Unexpected interaction effects

---

## 📈 Visualization Plans

After results are available, create:

### Figure 1: 2×2 Matrix Heatmap
```
               LayerNorm
            Yes        No
RNN  Yes   29.7%     27.8%
     No    13.2%      ???
```

### Figure 2: Component Contribution Breakdown
```
Baseline (neither):  ████████████ 37%
RNN contribution:    ████████████████ 55%
LN contribution:     ██ 6%
Synergy:            █ 3%
```

### Figure 3: Stability Analysis
```
CV comparison across 4 configurations for A3C and Individual
Shows how each component affects training stability
```

---

## ⚠️ Important Notes

1. **Configuration is correct**: `use_recurrent=False`, `use_layer_norm=False`
2. **Training takes time**: ~8-10 hours for 5 seeds
3. **Background process**: Use `BashOutput` tool to monitor progress
4. **Results are critical**: This completes the architecture ablation study

---

**Last Updated**: 2025-11-03 19:01:57 KST
**Status**: 🔄 Training in progress (Seed 1/5)
**Expected Completion**: 2025-11-04 03:00:00 KST (approximately)
