# Generalization Test v2 - Usage Guide

## Overview

`test_generalization_v2.py` has been enhanced to support selective model validation from both `runs/` and `archived_experiments/` folders.

## Key Features

1. **Dual Source Support**: Evaluate models from either:
   - `runs/` folder (default, not git-tracked)
   - `archived_experiments/` folder (git-tracked important experiments)

2. **Flexible Selection**:
   - Auto-detect latest experiment
   - Specify exact timestamp
   - List all available experiments

3. **Command-line Interface**: Easy-to-use arguments for batch processing

## Usage

### List Available Experiments

```bash
python test_generalization_v2.py --list
```

This will show all available experiments in both `runs/` and `archived_experiments/` folders.

### Evaluate from runs/ folder (default)

```bash
# Use latest experiment
python test_generalization_v2.py

# Specify timestamp
python test_generalization_v2.py --source runs --timestamp 20251021_153805
```

### Evaluate from archived_experiments/ folder

```bash
# Use latest archived experiment
python test_generalization_v2.py --source archived

# Specify timestamp
python test_generalization_v2.py --source archived --timestamp 20251021_153805
```

## Command-line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--source` | choice | `runs` | Source folder: `runs` or `archived` |
| `--timestamp` | string | None | Specific timestamp (e.g., 20251021_153805) |
| `--list` | flag | False | List available experiments and exit |

## Output Files

Results are saved with source type in filename:

- **Visualization**: `generalization_test_v2_{source}_{timestamp}.png`
- **CSV Results**: `generalization_results_v2_{source}_{timestamp}.csv`

Examples:
- `generalization_test_v2_runs_20251021_153805.png`
- `generalization_test_v2_archived_20251021_153805.png`

## Model Path Structure

### runs/ folder:
```
runs/
├── a3c_{timestamp}/
│   └── models/
│       └── global_final.pth
└── individual_{timestamp}/
    └── models/
        ├── individual_worker_0_final.pth
        ├── individual_worker_1_final.pth
        ├── ...
        └── individual_worker_4_final.pth
```

### archived_experiments/ folder:
```
archived_experiments/
└── {timestamp}/
    ├── a3c_{timestamp}/
    │   └── models/
    │       └── global_final.pth
    └── individual_{timestamp}/
        └── models/
            ├── individual_worker_0_final.pth
            ├── individual_worker_1_final.pth
            ├── ...
            └── individual_worker_4_final.pth
```

## Error Handling

The script will automatically:
- Detect if models exist before loading
- Show helpful error messages if paths are invalid
- Suggest using `--list` to see available experiments

## Examples

### 1. Quick test with latest model
```bash
python test_generalization_v2.py
```

### 2. Compare archived vs current runs
```bash
# Test archived experiment
python test_generalization_v2.py --source archived --timestamp 20251021_143814

# Test latest runs experiment
python test_generalization_v2.py --source runs
```

### 3. Batch evaluation of all archived experiments
```bash
# List experiments first
python test_generalization_v2.py --list

# Then evaluate each
for ts in 20251021_112839 20251021_143814 20251021_153805; do
    python test_generalization_v2.py --source archived --timestamp $ts
done
```

## Implementation Details

### Changes Made

1. **Argument Parsing**: Added argparse for command-line interface
2. **Early Import Optimization**: Lightweight imports before heavy torch modules
3. **Path Resolution**: Dynamic path construction based on source
4. **Validation**: Model existence check before evaluation
5. **Output Naming**: Include source type in output filenames

### Code Structure

```python
# Early parsing (before torch import)
parser = argparse.ArgumentParser()
args = parser.parse_args()

# List mode (no heavy imports needed)
if args.list:
    list_available_experiments()
    exit(0)

# Heavy imports only when needed
import torch
import pandas as pd
...

# Dynamic path construction
if args.source == 'runs':
    A3C_MODEL_PATH = f"runs/a3c_{TIMESTAMP}/models/global_final.pth"
else:  # archived
    A3C_MODEL_PATH = f"archived_experiments/{TIMESTAMP}/a3c_{TIMESTAMP}/models/global_final.pth"
```

## Troubleshooting

**Problem**: ModuleNotFoundError: No module named 'torch'
**Solution**: Activate conda environment before running:
```bash
conda activate torch-cert
python test_generalization_v2.py --list
```

**Problem**: Model not found error
**Solution**: Use `--list` to see available experiments, then specify correct timestamp

**Problem**: Want to test specific model
**Solution**: Use `--source` and `--timestamp` arguments together:
```bash
python test_generalization_v2.py --source archived --timestamp 20251021_153805
```
