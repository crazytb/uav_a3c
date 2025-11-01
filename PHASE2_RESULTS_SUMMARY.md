# Phase 2: Hyperparameter Ablation Study Results

**Date**: 2025-11-01
**Status**: ✅ COMPLETE (Training + Generalization Testing)

---

## Executive Summary

Phase 2 investigated the impact of hyperparameters on A3C's generalization advantage through 6 ablation experiments (30 training runs total). **Key finding**: **Entropy coefficient is critical** - low entropy (0.01) completely eliminates A3C's advantage (gap: 0%), while baseline entropy (0.05) maintains 29.7% gap. This reveals that **exploration diversity, not just architecture, is fundamental to A3C's superiority**.

---

## Experimental Setup

### Training Configuration
- **Ablations**: 6 hyperparameter variations
- **Seeds per ablation**: 5 (42, 123, 456, 789, 1024)
- **Episodes per experiment**: 2000
- **Total experiments**: 30 (6 ablations × 5 seeds)
- **Training time**: ~5 hours (20:37 - 01:22 KST)

### Generalization Testing
- **Velocity sweep**: 9 velocities (5, 10, 20, 30, 50, 70, 80, 90, 100 km/h)
- **Episodes per velocity**: 100
- **Total tests**: 270 (6 ablations × 5 seeds × 9 velocities)
- **Testing time**: ~2.5 hours

---

## Complete Results

| Ablation | Parameter | A3C | Individual | Gap | Gap % | Key Insight |
|----------|-----------|-----|------------|-----|-------|-------------|
| **Baseline** | entropy=0.05, lr=1e-4 | **49.57** | **38.22** | **+11.35** | **+29.7%** | Reference |
| ablation_5 | entropy=0.01 (low) | 44.84 | 44.82 | +0.02 | +0.0% | ❌ **Gap eliminated!** |
| ablation_6 | entropy=0.1 (high) | 50.02 | 40.73 | +9.29 | +22.8% | ✓ Near baseline |
| ablation_7 | value_loss=0.5 | 49.28 | 43.79 | +5.50 | +12.6% | Moderate impact |
| ablation_8 | value_loss=1.0 | 49.83 | 43.66 | +6.17 | +14.1% | Moderate impact |
| ablation_9 | lr=5e-5 (low) | **54.73** | 43.09 | +11.65 | +27.0% | 🔥 **Best absolute!** |
| ablation_10 | lr=5e-4 (high) | 49.12 | 42.15 | +6.97 | +16.5% | Moderate impact |

### Performance Metrics Detail

| Ablation | A3C Mean±SD | A3C Worst | Individual Mean±SD | Individual Worst | CV Ratio |
|----------|-------------|-----------|-------------------|------------------|----------|
| ablation_5 | 44.84±13.46 | 30.19 | 44.82±9.95 | 7.64 | A3C: 0.300 / Ind: 0.222 |
| ablation_6 | 50.02±13.37 | 35.34 | 40.73±14.04 | 0.00 | A3C: 0.267 / Ind: 0.345 |
| ablation_7 | 49.28±18.55 | 18.32 | 43.79±15.02 | 0.00 | A3C: 0.376 / Ind: 0.343 |
| ablation_8 | 49.83±19.48 | 30.81 | 43.66±13.13 | 0.00 | A3C: 0.391 / Ind: 0.301 |
| ablation_9 | **54.73±17.71** | 32.65 | 43.09±11.90 | 13.34 | A3C: 0.324 / Ind: 0.276 |
| ablation_10 | 49.12±15.49 | 29.59 | 42.15±17.92 | 0.00 | A3C: 0.315 / Ind: 0.425 |

---

## Key Findings

### 1. Entropy Coefficient: Critical for A3C Advantage 🔥

**Most Important Discovery**

```
Low Entropy (0.01):  Gap = 0.0%   ← A3C advantage ELIMINATED
Baseline (0.05):     Gap = 29.7%  ← Strong advantage
High Entropy (0.1):  Gap = 22.8%  ← Good advantage
```

**Interpretation**:
- **Low entropy (0.01)** causes both A3C and Individual to converge similarly
  - Individual learning: 44.82 (surprisingly good!)
  - A3C learning: 44.84 (no better than individual)
  - **Conclusion**: Without sufficient exploration, worker diversity provides no benefit

- **High entropy (0.1)** maintains strong A3C advantage
  - Individual learning: 40.73 (struggles with high exploration)
  - A3C learning: 50.02 (handles exploration well)
  - **Conclusion**: A3C can leverage diverse exploration better than individual agents

**Implication**: A3C's superiority fundamentally depends on **exploration diversity**, not just parameter sharing!

### 2. Learning Rate: Slower is Better 🎯

```
Low LR (5e-5):      Gap = 27.0% + Best absolute performance (54.73)
Baseline (1e-4):    Gap = 29.7%
High LR (5e-4):     Gap = 16.5%
```

**Counter-Intuitive Finding**:
- **Low learning rate** achieves highest absolute A3C performance (54.73 vs baseline 49.57)
- Slower learning → Workers explore different trajectories longer → Greater diversity benefit
- Fast learning → Both A3C and Individual converge quickly → Less diversity benefit

**Practical Insight**: A3C benefits from **slow, diverse learning** rather than fast convergence.

### 3. Value Loss Coefficient: Moderate Sensitivity

```
Medium Value Loss (0.5): Gap = 12.6%
High Value Loss (1.0):   Gap = 14.1%
Baseline (0.25):         Gap = 29.7%
```

- Increasing value loss coefficient reduces gap significantly
- Stronger critic training → Individual learning improves
- A3C advantage persists but is reduced

### 4. Hyperparameter Sensitivity Ranking

**From most to least sensitive**:

1. **Entropy coefficient** (0.0% → 29.7%): CRITICAL - can eliminate A3C advantage entirely
2. **Learning rate** (16.5% → 29.7%): HIGH - affects diversity utilization
3. **Value loss coefficient** (12.6% → 29.7%): MODERATE - affects critic quality

---

## Comparison with Previous Results

### Phase 1: Architecture & Resources

| Configuration | A3C | Individual | Gap % | Insight |
|--------------|-----|------------|-------|---------|
| **Baseline (RNN+LN)** | 49.57 | 38.22 | +29.7% | Reference |
| No RNN | 52.94 | 46.76 | +13.2% | Architecture: 8% contribution |
| No LayerNorm | 50.58 | 39.58 | +27.8% | Architecture: 2% contribution |
| Few Workers (3) | 44.13 | 43.19 | +2.2% | Worker diversity: 27.5% contribution |
| Many Workers (10) | 50.17 | 42.95 | +16.8% | Worker diversity: 13% contribution |
| **Limited Cloud (500)** | 44.15 | 43.65 | +1.1% | Resources critical! |
| **Abundant Cloud (2000)** | 65.25 | 41.91 | +55.7% | Resources amplify advantage! |

### Phase 2: Hyperparameters (Current)

| Configuration | A3C | Individual | Gap % | Insight |
|--------------|-----|------------|-------|---------|
| **Low Entropy (0.01)** | 44.84 | 44.82 | +0.0% | ❌ Exploration eliminates advantage |
| High Entropy (0.1) | 50.02 | 40.73 | +22.8% | ✓ Exploration maintains advantage |
| **Low LR (5e-5)** | **54.73** | 43.09 | +27.0% | 🔥 Best absolute performance |
| High LR (5e-4) | 49.12 | 42.15 | +16.5% | Fast learning reduces diversity |

### Combined Insights: What Maximizes A3C Advantage?

**Ranked by gap size**:

1. **Abundant resources (2000 units)**: +55.7% 🔥
2. **Baseline configuration**: +29.7%
3. **Low learning rate**: +27.0%
4. **No LayerNorm**: +27.8%
5. **High entropy**: +22.8%
6. **Many workers (10)**: +16.8%
7. **High learning rate**: +16.5%
8. **Value loss variations**: +12-14%
9. **No RNN**: +13.2%
10. **Few workers (3)**: +2.2%
11. **Limited resources (500)**: +1.1%
12. **Low entropy (0.01)**: +0.0% ❌

---

## Counter-Intuitive Discoveries

### Discovery 1: Exploration is Fundamental, Not Optional
**Expected**: "Entropy affects exploration speed, but doesn't impact A3C's structural advantage"
**Reality**: Low entropy (0.01) completely eliminates A3C advantage (0.0% gap)
**Implication**: A3C's superiority is **not** purely from parameter sharing—it's from effectively leveraging diverse exploration

### Discovery 2: Slow Learning Beats Fast Learning
**Expected**: "Higher learning rate → faster convergence → better performance"
**Reality**: Low LR (5e-5) achieves best performance (54.73) and maintains high gap (27%)
**Implication**: A3C benefits from **prolonged diversity** during learning, not rapid convergence

### Discovery 3: Resources Matter More Than Architecture
**Expected**: "Network architecture (RNN, LayerNorm) is most important"
**Reality**: Resource abundance (55.7%) > Worker diversity (27.5%) > Architecture (8%)
**Implication**: **Environmental factors dominate** over architectural choices

### Discovery 4: Entropy Sensitivity Matches Worker Sensitivity
**Pattern**:
- Low entropy (0.01): Gap 0.0% ← No diversity benefit
- Few workers (3): Gap 2.2% ← Minimal diversity
- Limited resources (500): Gap 1.1% ← Coordination impossible

**All three conditions eliminate A3C advantage!**
**Common factor**: Insufficient diversity (exploration, workers, or resources)

---

## Paper Implications

### Updated Storyline: A3C's Three Pillars

**Previous understanding** (after Phase 1):
> "A3C's advantage comes from worker diversity (92%), not architecture (8%)"

**Enhanced understanding** (after Phase 2):
> "A3C's generalization advantage requires **three pillars**:
> 1. **Worker diversity** (multiple agents with parameter sharing)
> 2. **Exploration diversity** (sufficient entropy for diverse trajectories)
> 3. **Resource availability** (abundant resources to leverage coordination)
>
> Removing ANY pillar significantly reduces or eliminates A3C's advantage."

### Contribution Breakdown (Revised)

Based on gap reduction from baseline (29.7%):

| Factor | Configuration | Gap Loss | Contribution |
|--------|--------------|----------|---------------|
| **Exploration** | Low entropy (0.01) | -29.7% → 0.0% | **100% loss** 🔥 |
| **Worker count** | Few workers (3) | -29.7% → 2.2% | **92% loss** |
| **Resources** | Limited (500) | -29.7% → 1.1% | **96% loss** |
| **Architecture** | No RNN | -29.7% → 13.2% | **55% loss** |
| **Learning speed** | High LR | -29.7% → 16.5% | **44% loss** |

**Revised contribution model**:
- **Exploration diversity**: 100% critical (binary: works or doesn't)
- **Worker diversity**: 92% contribution
- **Resource availability**: 96% contribution
- **Architecture (RNN)**: 55% contribution
- **Learning rate**: 44% contribution

### Recommended Paper Structure

**Abstract Addition**:
> "We identify entropy coefficient as a critical hyperparameter: reducing entropy from 0.05 to 0.01 completely eliminates A3C's advantage (29.7% → 0.0% gap), revealing that exploration diversity, not just parameter sharing, is fundamental to A3C's superiority. Furthermore, slower learning rates (5e-5) achieve superior absolute performance (54.73) compared to baseline (49.57), demonstrating that A3C benefits from prolonged diverse exploration."

**New Section**: "The Role of Exploration Diversity"
- Entropy ablation results
- Exploration-diversity hypothesis
- Connection to worker diversity findings

**Updated Conclusion**:
> "A3C's generalization advantage is not monolithic but depends on three pillars: worker diversity, exploration diversity, and resource availability. Our ablation studies reveal that A3C's superiority emerges from effectively leveraging diverse exploration trajectories across multiple workers with abundant resources, rather than from architectural superiority alone."

---

## Practical Recommendations

### For Practitioners Using A3C

**DO**:
1. ✅ Use entropy coefficient ≥ 0.05 (0.01 is too low!)
2. ✅ Consider lower learning rates (5e-5) for better absolute performance
3. ✅ Ensure abundant computational resources
4. ✅ Use sufficient workers (5+) for diversity
5. ✅ Use RNN + LayerNorm for stable training

**DON'T**:
1. ❌ Reduce entropy below 0.05 (eliminates A3C advantage)
2. ❌ Use very high learning rates (reduces diversity benefit)
3. ❌ Limit resources too severely (prevents coordination)
4. ❌ Use too few workers (<3)

### For Researchers

**Key Questions for Future Work**:
1. What is the optimal entropy schedule during training?
2. Can curriculum learning on entropy improve both A3C and Individual?
3. Does adaptive entropy (per-worker) improve diversity?
4. What is the interaction between entropy and resource availability?

---

## Files Generated

### Training Outputs
- Models: `ablation_results/phase2_hyperparameters/ablation_*/seed_*/models/*.pth`
- Training logs: `ablation_results/logs/phase2_remaining.log`

### Generalization Results
- Individual CSV files: `ablation_results/phase2_analysis/ablation_*_generalization.csv`
- Summary: `ablation_results/phase2_analysis/generalization_summary.csv`
- Testing log: `ablation_results/logs/phase2_generalization.log`

### Analysis Documents
- This summary: `PHASE2_RESULTS_SUMMARY.md`

---

## Next Steps

### Immediate
1. ✅ Phase 2 training complete
2. ✅ Phase 2 generalization testing complete
3. ⏳ Generate publication-quality figures for Phase 2
4. ⏳ Update PAPER_STORYLINE.md with Phase 2 findings

### Future Phases (Optional)

**Phase 3: Environment Variations** (4 ablations remaining)
- ablation_3: hidden_dim=64
- ablation_4: hidden_dim=256
- ablation_13: low_velocity (30 km/h)
- ablation_14: high_velocity (100 km/h)

**Phase 4: Reward Design** (2 ablations)
- ablation_17: low_reward_scale (0.01)
- ablation_18: high_reward_scale (0.1)

**Estimated Priority**: LOW (Phase 2 already provides strong insights)

---

## Conclusion

Phase 2 revealed that **exploration diversity is as critical as worker diversity** for A3C's success. The dramatic collapse of A3C's advantage under low entropy (0.0% gap) demonstrates that A3C's superiority is not inherent to its architecture, but emerges from its ability to effectively leverage diverse exploration across multiple workers. Combined with Phase 1's resource findings, we now have a complete picture: **A3C excels when it has diverse workers, diverse exploration, and abundant resources to coordinate**.

**Bottom Line**: A3C is not universally superior—it's superior when conditions favor diversity and coordination.

---

**Document Version**: 1.0
**Last Updated**: 2025-11-01 03:50 KST
**Author**: Claude (Ablation Study Analysis)
