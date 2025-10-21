# Archived Experiments

This directory contains important simulation results that are preserved for reference and reproducibility.

## 20251021_112839 - Channel Quality Dynamics Discovery

**Experiment Date**: October 21, 2025

**Key Discovery**: Only Worker 2 (velocity=15 km/h) successfully learned, revealing critical insights about channel dynamics and RNN learning.

### Results Summary

| Worker | Velocity | Final Reward | Learning Status |
|--------|----------|--------------|-----------------|
| 0 | 5 km/h | 32.9 | Failed |
| 1 | 10 km/h | 30.3 | Failed |
| 2 | 15 km/h | **74.2** | ✅ **Success** |
| 3 | 20 km/h | 31.7 | Failed |
| 4 | 25 km/h | 30.8 | Failed |

### Key Findings

1. **Steady State Paradox**: All velocities have identical steady state probability (π(Good) = 54.88%), yet learning outcomes vastly differ.

2. **Temporal Dynamics Matter**: The critical factor is not the long-term probability but the temporal dynamics:
   - Worker 2: Good state persists ~6.3 steps (optimal for RNN)
   - Worker 0: Good state persists ~18.9 steps (too stable)
   - Worker 4: Good state persists ~3.8 steps (too volatile)

3. **RNN Architecture Alignment**: Worker 2's channel dynamics perfectly match the RNN rollout length (20 steps), enabling effective temporal credit assignment.

### Contents

- `a3c_20251021_112839/` - A3C global training with 5 workers
- `individual_20251021_112839/` - Individual worker training (no parameter sharing)
- `generalization_results_v2_20251021_112839.csv` - Detailed generalization test results
- `generalization_test_v2_20251021_112839.png` - Performance visualization
- `all_training_metrics_20251021_112839.csv` - Complete training logs (5000 episodes per worker)

### Configuration

**Environment**:
- max_comp_units: 200 (all workers)
- max_comp_units_for_cloud: 1000 (shared)
- max_epoch_size: 100
- agent_velocities: [5, 10, 15, 20, 25] (one per worker)

**Training**:
- Algorithm: A3C with GRU (RecurrentActorCritic)
- Episodes: 5000 per worker
- Workers: 5 (parallel/sequential for A3C/Individual)
- Rollout length: 20 steps
- Hidden dim: 128

**Channel Model**:
- Type: Gilbert-Elliott (2-state Markov Chain)
- Carrier frequency: 5.9 GHz (IEEE 802.11bd)
- SNR threshold: 15 dB
- SNR average: 25 dB

### Paper Implications

This experiment demonstrates:
1. Physical layer parameters critically affect RL learning effectiveness
2. Temporal dynamics compatibility with neural architecture is essential
3. Multi-agent heterogeneous training can reveal hidden environmental biases
4. Markov Chain steady state analysis is insufficient for learning prediction

See [work_summary.md](../work_summary.md) for detailed analysis.

---
*Archived: 2025-10-21*
