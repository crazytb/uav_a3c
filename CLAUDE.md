# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a UAV (Unmanned Aerial Vehicle) reinforcement learning project implementing A3C (Asynchronous Advantage Actor-Critic) algorithms with RNN support. The project focuses on multi-UAV task offloading optimization using deep reinforcement learning.

## Core Architecture

### Main Entry Points
- `main_train.py` - Primary training script for A3C global and individual workers
- `main_evaluation.py` - Basic model evaluation script
- `main_evaluation_fixed.py` - Enhanced evaluation with action logging and CSV output
- `main_evaluation_fixed_rev.py` - Extended evaluation with comprehensive analysis
- `main_plot_rewards.py` - Plotting utilities for training metrics visualization

### Framework Structure (`drl_framework/`)
- `trainer.py` - Core A3C training implementation with multiprocessing
- `networks.py` - Neural network architectures (ActorCritic and RecurrentActorCritic)
- `custom_env.py` - Custom UAV environment implementation
- `params.py` - Centralized configuration parameters
- `network_state.py` - Network state management utilities
- `utils.py` - Utility functions for data processing

### Key Components
1. **A3C Implementation**: Multi-worker asynchronous training with global model sharing
2. **RNN Support**: RecurrentActorCritic with GRU for sequential decision making
3. **Custom Environment**: UAV task offloading simulation with action masking
4. **Evaluation Framework**: Comprehensive model comparison and action logging

## Common Commands

### Training
```bash
python main_train.py
```
Trains both A3C global model and individual worker models with configured parameters.

### Evaluation
```bash
python main_evaluation_fixed.py
```
Comprehensive model evaluation with action logging and CSV export.

### Plotting
```bash
python main_plot_rewards.py
```
Generate training reward plots and metrics visualization.

## Configuration

All parameters are centralized in `drl_framework/params.py`:
- `n_workers`: Number of parallel A3C workers (default: 5)
- `target_episode_count`: Episodes per worker (default: 5000)
- `ENV_PARAMS`: Environment configuration (UAV units, velocities, etc.)
- `REWARD_PARAMS`: Reward function coefficients
- `device`: Set to CPU by default for compatibility

## Model Storage

- Training runs create timestamped directories in `runs/`
- A3C global models: `runs/a3c_{timestamp}/models/global_final.pth`
- Individual models: `runs/individual_{timestamp}/models/individual_worker_{i}_final.pth`
- Training logs and evaluation CSVs are saved to project root

## Data Files

The repository contains CSV files with action logs and evaluation results:
- `*_actions.csv`: Detailed action logs per environment and worker
- `evaluation_results.csv`: Model comparison metrics
- Training metrics are logged to `runs/all_training_metrics_{timestamp}.csv`

## Development Notes

- The codebase uses PyTorch with multiprocessing for A3C implementation
- Action masking is implemented in the environment for valid action selection
- Models support both feedforward and recurrent architectures
- Evaluation includes both greedy and stochastic policy evaluation

## Current Research Status (Last Updated: 2025-10-29)

### Completed Work

**Baseline Experiments (2000 episodes, 5 seeds)**
- Training: A3C 60.31 ± 6.41 vs Individual 57.57 ± 4.84 (+4.76%)
- **Generalization: A3C 49.57 ± 14.35 vs Individual 38.22 ± 16.24 (+29.7%)**
- Key finding: A3C demonstrates superior generalization across velocity variations (5-100 km/h)
- Results saved in: `ablation_results/baseline_20251029_165119/`
- Documentation: `docs/analysis/BASELINE_EXPERIMENT_SUMMARY.md`

**Key Insight**
A3C's primary advantage is **generalization performance**, not training performance. Individual learning suffers from catastrophic failures in diverse conditions (worst-case: 1.25 vs A3C: 31.72).

### Next Steps: Ablation Study

**Goal**: Identify which components contribute to A3C's generalization advantage

**Planned Ablations** (21 experiments in 4 phases):
1. Network architecture: No RNN, No LayerNorm, hidden dim variations
2. Hyperparameters: entropy, value loss, learning rate variations
3. Environment: worker count, cloud resources, velocity variations
4. Reward design: penalty coefficient variations

**Recommended Approach**: Generalization-based ablation (10-20x faster)
- Train each ablation for 500-1000 episodes (vs 2000 for baseline)
- Test on velocity sweep (5-100 km/h)
- Compare generalization scores, not training scores
- Estimated time: ~50 hours total vs 200 hours for training-based

**To Resume Ablation Study**:
1. Read `RESUME_ABLATION_STUDY.md` for detailed instructions
2. Choose ablation from `ablation_configs.py`
3. Run training with `run_baseline_simple.sh` pattern
4. Test with `test_baseline_generalization.py`
5. Compare results to baseline

**Priority Ablations**:
- `no_rnn`: Test RNN contribution to generalization (expected: major impact)
- `no_layer_norm`: Test LayerNorm stability effect (expected: moderate impact)
- `workers_3` and `workers_10`: Test worker diversity impact (expected: important)

### Important Scripts

**Training**: `main_train.py` with environment variable `RANDOM_SEED`
**Generalization Testing**: `test_baseline_generalization.py --baseline-dir <dir>`
**Analysis**: `analyze_baseline_results.py` for statistical comparison
**Configuration**: `ablation_configs.py` with `get_config(ablation_name)` function

### Python Environment

Use conda environment directly to avoid import errors:
```bash
~/miniconda3/envs/torch-cert/bin/python script.py
```

### Known Issues

1. PyTorch 2.6+ requires `weights_only=False` in `torch.load()`
2. Checkpoints saved with `model_state_dict` key, handle both formats
3. Shell scripts need Unix line endings (use `sed -i '' 's/\r$//'` on macOS)