# Neither RNN nor LayerNorm - Experimental Results (Current Environment)

**Experiment Date**: 2025-11-04
**Configuration**: Feedforward ActorCritic (no RNN, no LayerNorm)
**Seeds**: 5 (42, 123, 456, 789, 1024)
**Episodes**: 2000 per worker
**Generalization Test**: 9 velocities (5-100 km/h), 100 episodes each

---

## 📊 Main Results (All 5 Seeds Included)

### Overall Performance

| Metric | A3C | Individual | Difference |
|--------|-----|------------|------------|
| **Mean** | 40.74 ± 24.32 | 49.86 ± 17.06 | **-18.3%** (Individual better) |
| **CV** | 0.597 | 0.342 | Individual more stable |
| **Worst-case** | 0.00 | 0.00 | Both have catastrophic failures |

**Note**: Results include 1 catastrophic failure (Seed 123 A3C completely failed). This significantly impacts the overall mean.

---

## 🔍 Per-Seed Breakdown

| Seed | A3C Mean | A3C Std | Individual Mean | Individual Std | Gap | Status |
|------|----------|---------|-----------------|----------------|-----|--------|
| 42 | 30.83 | 0.56 | 31.93 | 18.32 | -3.5% | Both suboptimal |
| **123** | **0.00** | **0.00** | 65.30 | 15.19 | **-100%** | **A3C FAILED** ❌ |
| **456** | **70.83** | 1.89 | 49.93 | 10.76 | **+41.9%** | A3C excellent ✅ |
| 789 | 49.59 | 0.33 | 45.49 | 10.55 | +9.0% | Both good ✅ |
| 1024 | 52.48 | 2.98 | 56.62 | 7.09 | -7.3% | Both good ✅ |

### Catastrophic Failures Detected

1. **Seed 123 - A3C Global Model**
   - Mean reward: 0.0 across ALL velocities
   - Complete policy collapse during training
   - Never recovered

2. **Seed 42 - Individual Worker 2**
   - Mean reward: 0.0 across ALL velocities
   - One worker completely failed
   - Other 4 workers in Seed 42 performed normally

---

## 📈 Comparison with Other Computer

### Other Computer Results (Different Environment)

| Metric | A3C | Individual | Gap |
|--------|-----|------------|-----|
| **Mean** | 49.59 ± 14.16 | 38.23 ± 16.28 | **+29.7%** |
| **CV** | 0.285 | 0.426 | A3C 33% better |
| **Worst-case** | 31.60 | 1.41 | 22.4× difference |
| **Claim** | "Gap is 100% algorithmic, 0% architectural" | | |

### Current Environment Results

| Metric | A3C | Individual | Gap |
|--------|-----|------------|-----|
| **Mean** | 40.74 ± 24.32 | 49.86 ± 17.06 | **-18.3%** |
| **CV** | 0.597 | 0.342 | Individual 43% better |
| **Worst-case** | 0.00 | 0.00 | Both have failures |
| **Finding** | "Results highly unstable and environment-dependent" | | |

---

## 🎯 Critical Differences

### 1. Overall Gap Direction is OPPOSITE

- **Other computer**: A3C +29.7% better
- **Current**: Individual +18.3% better
- **Complete reversal of conclusion!**

### 2. Stability Pattern is REVERSED

- **Other computer**: A3C more stable (CV 0.285 vs 0.426)
- **Current**: Individual more stable (CV 0.342 vs 0.597)
- **Contradicts original finding!**

### 3. A3C Performance

- **Other computer**: 49.59 (consistent)
- **Current**: 40.74 (includes 1 complete failure)
- **Without failure (Seed 123)**: 50.93 ✅ (similar to other computer)

### 4. Individual Performance

- **Other computer**: 38.23 (low)
- **Current**: 49.86 (high)
- **+30% difference in Individual performance!**

---

## 💡 Key Insights

### Why Results Differ Across Environments

**Hypothesis 1: Catastrophic Failures are Random**
- Seed 123 A3C failed here, but may not fail on other computer
- Seed 42 Individual Worker 2 failed here
- **20% failure rate (1/5 seeds)** makes results unreliable

**Hypothesis 2: Environment Affects Individual More**
- A3C performance similar when successful (~50)
- Individual performance varies widely (38.23 vs 49.86)
- **Individual learning is more sensitive to environment**

**Hypothesis 3: Without Stabilization, Results are Chaotic**
- No RNN + No LayerNorm = no stability mechanisms
- Small environment differences amplified during training
- **Results depend on which seeds fail**

---

## 🔬 Training Instability Evidence

### Policy Collapse Observed During Training

**Example from Seed 123 (A3C - Failed)**:
```
Episode 100: Reward 33.70, Loss 2.61, Probs [0.008, 0.136, 0.856]
Episode 200: Reward 36.15, Loss 938M, Probs [1.0, 0.0, 0.0]  ← COLLAPSED!
Episode 500: Reward 29.35, Loss 4102, Probs [1.0, 0.0, 0.0]
...continues with single action forever...
```

**Example from Seed 456 (A3C - Success)**:
```
Episode 100: Reward 34.45, Loss 1.82, Probs [0.009, 0.148, 0.843]
Episode 500: Reward 50.80, Loss 21.5, Probs [0.346, 0.336, 0.318]  ← Healthy!
Episode 1000: Reward 55.60, Loss 15.7, Probs [0.372, 0.341, 0.287]
Episode 2000: Reward 64.85, Loss 8.27, Probs [0.389, 0.367, 0.244]
```

**Pattern**: Without LayerNorm, ~20% of seeds experience catastrophic policy collapse with exploding losses.

---

## 🎨 Per-Velocity Performance (All Seeds)

| Velocity (km/h) | A3C Mean | Individual Mean | Difference |
|-----------------|----------|-----------------|------------|
| 5 | 40.34 | 48.28 | -16.4% |
| 10 | 40.91 | 49.37 | -17.1% |
| 20 | 41.08 | 49.89 | -17.7% |
| 30 | 41.61 | 49.95 | -16.7% |
| 50 | 41.18 | 50.30 | -18.1% |
| 70 | 40.42 | 50.20 | -19.5% |
| 80 | 40.41 | 50.29 | -19.6% |
| 90 | 40.36 | 50.38 | -19.9% |
| 100 | 40.38 | 50.04 | -19.3% |

**Observation**: Individual consistently outperforms A3C across all velocities (when including the failed Seed 123).

---

## 📊 Complete 2×2 Matrix

```
                   LayerNorm
                Yes         No
RNN    Yes     29.7% ✅    27.8% ✅
       No      13.2% ✅     ???

Current Environment:  -18.3% ❌ (Individual wins!)
Other Environment:    +29.7% ✅ (A3C wins!)
```

**Conclusion**: Without architectural stabilization, **results are environment-dependent and unreliable**.

---

## 🚨 Critical Problem: Reproducibility Failure

### Same Configuration, Opposite Conclusions

| Aspect | Other Computer | Current Environment |
|--------|---------------|---------------------|
| **Winner** | A3C (+29.7%) | Individual (+18.3%) |
| **Stability** | A3C better (CV 0.285) | Individual better (CV 0.342) |
| **Claim** | "Gap is 100% algorithmic" | "Results are chaotic" |
| **Recommendation** | "Neither shows pure A3C advantage" | "Neither is too unstable" |

### Why This Matters

1. **Cannot publish conflicting results**
   - Reviewers will question methodology
   - "Which result is correct?"
   - Answer: Neither - both are artifacts of instability

2. **Scientific conclusion depends on which computer ran the experiment**
   - Not reproducible science
   - Cannot build theory on unstable foundation

3. **Previous "gap is 100% algorithmic" claim is questionable**
   - Only true in one specific environment
   - Reverses in another environment
   - Likely an artifact of which seeds failed

---

## 📝 Revised Understanding

### What We Learned

1. **Neither Configuration is Too Unstable for Research**
   - 20% catastrophic failure rate
   - Results flip between environments
   - Cannot draw reliable conclusions

2. **Architecture Provides Essential Stability**
   - RNN+LN: Consistent 29.7% gap across environments ✅
   - RNN only: Consistent 27.8% gap ✅
   - LN only: Consistent 13.2% gap ✅
   - **Neither: -18.3% to +29.7% depending on environment** ❌

3. **Previous "100% Algorithmic" Claim Was Wrong**
   - Based on unstable configuration
   - Results not reproducible
   - Architecture DOES matter for stability

4. **True Decomposition**:
   - **RNN**: +16.5 pp (adds stability and task difficulty)
   - **LayerNorm**: +1.9 pp (crucial stability component)
   - **Base algorithm**: ~13% (when stable)
   - **Instability without architecture**: -18% to +30% variation!

---

## 🎯 Final Recommendations

### For This Experiment

**DO NOT USE** the neither configuration results for:
- ❌ Publishing
- ❌ Drawing conclusions about A3C
- ❌ Claiming "gap is 100% algorithmic"
- ❌ Baseline performance measurement

**DO USE** the neither configuration results to:
- ✅ Demonstrate importance of architectural stability
- ✅ Show reproducibility challenges
- ✅ Motivate RNN and LayerNorm design choices
- ✅ Highlight risk of unstable configurations

### For Paper

**Original claim** (from other computer):
> "A3C's 29.7% advantage is entirely algorithmic (100%), not architectural (0%)"

**Corrected claim** (from both experiments):
> "Without architectural stabilization (RNN and LayerNorm), results vary wildly across environments (-18% to +30% gap), demonstrating that architecture provides essential stability for reliable A3C advantages. The consistent gaps with RNN+LN (29.7%), RNN-only (27.8%), and LN-only (13.2%) show that architectural components contribute both to absolute performance and training stability."

---

## 📊 Summary Statistics (All Seeds, No Exclusions)

### Overall Performance
- A3C: 40.74 ± 24.32 (CV: 0.597)
- Individual: 49.86 ± 17.06 (CV: 0.342)
- Gap: -18.3% (Individual wins)

### Failure Analysis
- A3C catastrophic failures: 1/5 seeds (20%)
- Individual catastrophic failures: 1/25 workers (4%)
- Total experiments with failures: 2/10 (20%)

### Environment Comparison
- This environment: Individual +18.3%
- Other environment: A3C +29.7%
- **Total swing: 48 percentage points!**

---

**Last Updated**: 2025-11-04 11:47 KST
**Experiment Duration**: 3 hours training + 3 minutes testing
**Key Finding**: Neither configuration is too unstable for reliable research. Results are environment-dependent and not reproducible. This experiment validates the importance of RNN and LayerNorm for stable, reproducible results.

**Critical Lesson**: When you remove all stabilization mechanisms (no RNN, no LayerNorm), you don't get "pure algorithmic advantage" - you get chaos.
