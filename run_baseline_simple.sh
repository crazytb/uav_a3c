#!/bin/bash
# Simple baseline ablation study runner using torch-cert environment

echo "=========================================="
echo "Baseline Ablation Study - 5 Seeds"
echo "=========================================="
echo ""

# Use torch-cert conda environment
PYTHON_PATH=~/miniconda3/envs/torch-cert/bin/python

# Check if python exists
if [ ! -f "$PYTHON_PATH" ]; then
    echo "ERROR: Python not found at $PYTHON_PATH"
    echo "Please update PYTHON_PATH in this script"
    exit 1
fi

echo "Using Python: $PYTHON_PATH"
$PYTHON_PATH --version
echo ""

SEEDS=(42 123 456 789 1024)
RESULTS_DIR="ablation_results/baseline_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$RESULTS_DIR"

echo "Results will be saved to: $RESULTS_DIR"
echo ""

for i in "${!SEEDS[@]}"; do
    SEED=${SEEDS[$i]}
    echo "=========================================="
    echo "[$((i+1))/5] Running with SEED=$SEED"
    echo "=========================================="

    export RANDOM_SEED=$SEED

    # Run training with torch-cert python
    $PYTHON_PATH main_train.py

    if [ $? -eq 0 ]; then
        echo "[SUCCESS] Seed $SEED completed"

        # Move results to ablation directory
        SEED_DIR="$RESULTS_DIR/seed_$SEED"
        mkdir -p "$SEED_DIR"

        # Find latest runs
        LATEST_A3C=$(ls -td runs/a3c_* 2>/dev/null | head -1)
        LATEST_IND=$(ls -td runs/individual_* 2>/dev/null | head -1)

        if [ -n "$LATEST_A3C" ]; then
            mv "$LATEST_A3C" "$SEED_DIR/a3c"
            echo "  Moved A3C results to $SEED_DIR/a3c"
        fi

        if [ -n "$LATEST_IND" ]; then
            mv "$LATEST_IND" "$SEED_DIR/individual"
            echo "  Moved Individual results to $SEED_DIR/individual"
        fi

    else
        echo "[FAILED] Seed $SEED failed with exit code $?"
    fi

    echo ""
done

echo "=========================================="
echo "Baseline Study Complete!"
echo "Results saved to: $RESULTS_DIR"
echo "=========================================="
