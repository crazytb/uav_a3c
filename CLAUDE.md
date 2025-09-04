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