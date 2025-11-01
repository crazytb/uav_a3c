#!/bin/bash

# Phase 2 Remaining: Complete hyperparameter ablations (ablation 7-10)
# 4 ablations × 5 seeds = 20 experiments remaining

set -e

PYTHON_BIN=~/miniconda/envs/torch-cert/bin/python
OUTPUT_DIR="ablation_results/phase2_hyperparameters"
LOG_DIR="${OUTPUT_DIR}/logs"

SEEDS=(42 123 456 789 1024)

# Remaining ablations (7-10)
ABLATIONS=(
    "ablation_7_medium_value_loss"   # value_loss_coef=0.5 (vs baseline 0.25)
    "ablation_8_high_value_loss"     # value_loss_coef=1.0 (vs baseline 0.25)
    "ablation_9_low_lr"              # lr=5e-5 (vs baseline 1e-4)
    "ablation_10_high_lr"            # lr=5e-4 (vs baseline 1e-4)
)

echo "========================================"
echo "Phase 2: Remaining Hyperparameter Ablations"
echo "========================================"
echo "Completing ablations 7-10"
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
echo "Phase 2 Remaining Complete!"
echo "========================================"
echo "End time: $(date)"
echo "Total experiments: $total_experiments"
echo ""
echo "Full Phase 2 status:"
$PYTHON_BIN -c "
import os
phase2_dir = '$OUTPUT_DIR'
ablations = ['ablation_5_low_entropy', 'ablation_6_high_entropy', 'ablation_7_medium_value_loss', 'ablation_8_high_value_loss', 'ablation_9_low_lr', 'ablation_10_high_lr']
total_seeds = 0
for abl in ablations:
    abl_path = os.path.join(phase2_dir, abl)
    if os.path.exists(abl_path):
        seeds = [d for d in os.listdir(abl_path) if d.startswith('seed_')]
        print(f'{abl}: {len(seeds)}/5 seeds')
        total_seeds += len(seeds)
    else:
        print(f'{abl}: 0/5 seeds')
print(f'\nTotal: {total_seeds}/30 experiments ({total_seeds*100//30}%)')
"
echo ""
echo "Next step: Run generalization testing"
echo "  $PYTHON_BIN test_ablation_generalization.py \\"
echo "    --ablation-dir $OUTPUT_DIR \\"
echo "    --output-dir ablation_results/phase2_analysis \\"
echo "    --velocities 5 10 20 30 50 70 80 90 100 \\"
echo "    --n-episodes 100 \\"
echo "    --ablations ablation_5_low_entropy ablation_6_high_entropy ablation_7_medium_value_loss ablation_8_high_value_loss ablation_9_low_lr ablation_10_high_lr"
echo ""
