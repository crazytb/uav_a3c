# Neither RNN nor LayerNorm Experiment

**Started**: 2025-11-04 11:28:53
**Configuration**: Feedforward ActorCritic (no RNN, no LayerNorm)
**Purpose**: Re-run the "neither" experiment in consistent environment

---

## Current Status

### Training Progress
- **Seeds Completed**: Check with `./monitor_neither_progress.sh`
- **Results Directory**: `ablation_results/neither_rnn_nor_ln_20251104_112853/`
- **Estimated Completion**: ~2-3 hours from 11:28

### What Was Modified

1. **test_ablation_generalization.py** (Line 55-77)
   - Fixed to support both ActorCritic and RecurrentActorCritic
   - Now properly detects RNN from checkpoint and loads correct model class
   - Previous version always used RecurrentActorCritic, which was incorrect

2. **run_neither_rnn_nor_ln.sh**
   - Fixed Python path from `~/miniconda3/envs/...` to `~/miniconda/envs/...`

---

## Why Re-running?

The experiment was run on a different computer with potentially different:
- Python environment (different miniconda installation)
- Model loading code (RecurrentActorCritic vs ActorCritic)
- Random seed behavior (different hardware/OS)

This re-run ensures:
- ✅ Consistent environment with other ablation experiments
- ✅ Correct model architecture (ActorCritic, not RecurrentActorCritic)
- ✅ Reproducible results on this machine

---

## Configuration

```python
# drl_framework/params.py
use_recurrent = False      # Feedforward ActorCritic
use_layer_norm = False     # No normalization
hidden_dim = 128           # Standard hidden size
n_workers = 5              # 5 parallel workers
target_episode_count = 2000  # 2000 episodes per worker
```

---

## Running the Experiment

### 1. Monitor Training Progress
```bash
./monitor_neither_progress.sh
```

Shows:
- Number of completed seeds
- Recent training output
- Completed training runs

### 2. Watch Live Progress
```bash
tail -f neither_training.log
```

### 3. After Training Completes

Run generalization test:
```bash
./run_neither_generalization.sh
```

This will:
- Test all 5 seeds across 9 velocities (5-100 km/h)
- Run 100 episodes per velocity
- Save results to CSV
- Take ~2-3 hours

---

## Expected Results

Based on previous run (may differ slightly):

### Performance Summary
| Metric | A3C | Individual | Gap % |
|--------|-----|------------|-------|
| Mean | 49.59 ± 14.16 | 38.23 ± 16.28 | 29.7% |
| CV | 0.285 | 0.426 | A3C 33% better |
| Worst-case | 31.60 | 1.41 | 22.4× difference |

### Key Findings from Previous Run

1. **Gap is NOT from Architecture**
   - Neither (no RNN, no LN): 29.7% gap
   - Baseline (RNN+LN): 29.7% gap
   - **Same gap!** Architecture affects variance, not gap

2. **RNN's True Role**
   - Does NOT create the gap
   - Acts as "difficulty amplifier"
   - Individual struggles more with RNN (CV 0.425 vs 0.217)
   - A3C handles RNN better (CV stable ~0.29)

3. **Training Instability**
   - 4/5 seeds showed policy collapse in previous run
   - Only Seed 42 trained successfully
   - Individual workers had catastrophic failures
   - A3C maintained reasonable performance

---

## After Generalization Test

### Compare Results

```bash
# View generalization results
RESULTS_DIR=$(ls -td ablation_results/neither_rnn_nor_ln_* | head -1)
python -c "
import pandas as pd
df = pd.read_csv('$RESULTS_DIR/generalization_results.csv')
print('\n=== Overall Performance ===')
print(df.groupby('method')[['mean_reward', 'std_reward']].agg(['mean', 'std']))
print('\n=== Per-Velocity Performance ===')
print(df.pivot_table(values='mean_reward', index='velocity', columns='method'))
"
```

### Expected Comparisons

**With Baseline (RNN+LN)**:
```
Configuration       A3C Mean    Individual Mean    Gap %
---------------------------------------------------------
Baseline (RNN+LN)   49.57       38.22              29.7%
Neither (running)   ???         ???                ???
```

**With Other Ablations**:
```
Configuration       Gap %    Why?
---------------------------------------------------------
RNN+LN (Baseline)   29.7%    Full architecture
RNN only            27.8%    Similar to baseline
LN only (No RNN)    13.2%    Individual catches up (more stable)
Neither (running)   ???      Baseline algorithm only
```

---

## Scripts Available

1. **run_neither_rnn_nor_ln.sh** - Main training script (running in background)
2. **monitor_neither_progress.sh** - Check training progress
3. **run_neither_generalization.sh** - Run generalization test after training
4. **run_neither_full_pipeline.sh** - Full pipeline (training + testing)

---

## Troubleshooting

### Check if training is still running
```bash
ps aux | grep run_neither
```

### View background process output
```bash
tail -100 neither_training.log
```

### Check for errors
```bash
grep -i error neither_training.log
grep -i failed neither_training.log
```

### Verify results directory
```bash
ls -la ablation_results/neither_rnn_nor_ln_*/
```

---

## Next Steps After Completion

1. **Analyze Results**
   - Compare with baseline configuration
   - Verify gap percentage
   - Check training stability

2. **Update Documentation**
   - Update `docs/analysis/NEITHER_RNN_NOR_LN_RESULTS.md` with new results
   - Compare with previous run from different computer
   - Document any differences

3. **Update Paper Materials**
   - Confirm findings about gap source (algorithmic vs architectural)
   - Update component contribution analysis
   - Revise paper storyline if needed

---

**Note**: This experiment completes the 2×2 RNN/LayerNorm architecture matrix:

```
               LayerNorm
            Yes        No
RNN  Yes   29.7%     27.8%
     No    13.2%      ???  ← This experiment
```
