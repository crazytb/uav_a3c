#!/bin/bash

# Phase 2: Hyperparameter Ablation Study
# 6 ablations × 5 seeds = 30 experiments
# Estimated time: ~30 hours

set -e

PYTHON_BIN=~/miniconda/envs/torch-cert/bin/python
OUTPUT_DIR="ablation_results/phase2_hyperparameters"
LOG_DIR="${OUTPUT_DIR}/logs"

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

SEEDS=(42 123 456 789 1024)

# Ablation configurations
ABLATIONS=(
    "ablation_5_low_entropy"         # entropy_coef=0.01 (vs baseline 0.05)
    "ablation_6_high_entropy"        # entropy_coef=0.1 (vs baseline 0.05)
    "ablation_7_medium_value_loss"   # value_loss_coef=0.5 (vs baseline 0.25)
    "ablation_8_high_value_loss"     # value_loss_coef=1.0 (vs baseline 0.25)
    "ablation_9_low_lr"              # lr=5e-5 (vs baseline 1e-4)
    "ablation_10_high_lr"            # lr=5e-4 (vs baseline 1e-4)
)

echo "========================================"
echo "Phase 2: Hyperparameter Ablation Study"
echo "========================================"
echo "Total experiments: ${#ABLATIONS[@]} ablations × ${#SEEDS[@]} seeds = $((${#ABLATIONS[@]} * ${#SEEDS[@]}))"
echo "Output directory: $OUTPUT_DIR"
echo "Start time: $(date)"
echo ""

total_experiments=$((${#ABLATIONS[@]} * ${#SEEDS[@]}))
current=0

for ablation in "${ABLATIONS[@]}"; do
    echo "----------------------------------------"
    echo "Starting ablation: $ablation"
    echo "----------------------------------------"

    for seed in "${SEEDS[@]}"; do
        current=$((current + 1))

        echo "[$current/$total_experiments] Running $ablation with seed=$seed"

        log_file="${LOG_DIR}/${ablation}_seed${seed}.log"

        # Run training
        $PYTHON_BIN run_single_ablation.py \
            --ablation "$ablation" \
            --seed "$seed" \
            --output-dir "$OUTPUT_DIR" \
            > "$log_file" 2>&1

        if [ $? -eq 0 ]; then
            echo "  ✓ Completed successfully"
        else
            echo "  ✗ Failed (see $log_file)"
            exit 1
        fi
    done

    echo ""
done

echo "========================================"
echo "Phase 2 Training Complete!"
echo "========================================"
echo "End time: $(date)"
echo "Total experiments: $total_experiments"
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Next step: Run generalization testing"
echo "  $PYTHON_BIN test_ablation_generalization.py \\"
echo "    --ablation-dir $OUTPUT_DIR \\"
echo "    --output-dir ablation_results/phase2_analysis \\"
echo "    --velocities 5 10 20 30 50 70 80 90 100 \\"
echo "    --n-episodes 100 \\"
echo "    --ablations ${ABLATIONS[@]}"
echo ""
