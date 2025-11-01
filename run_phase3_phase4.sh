#!/bin/bash

# Phase 3 & 4: Environment and Reward Ablations
# 6 ablations × 5 seeds = 30 experiments

set -e

PYTHON_BIN=~/miniconda/envs/torch-cert/bin/python
OUTPUT_DIR="ablation_results/phase3_phase4"
LOG_DIR="${OUTPUT_DIR}/logs"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

SEEDS=(42 123 456 789 1024)

# Phase 3: Environment ablations (architecture + velocity)
# Phase 4: Reward design ablations
ABLATIONS=(
    "ablation_3_small_hidden"       # hidden_dim=64 (vs baseline 128)
    "ablation_4_large_hidden"       # hidden_dim=256 (vs baseline 128)
    "ablation_13_low_velocity"      # velocity=30 km/h (vs baseline 50)
    "ablation_14_high_velocity"     # velocity=100 km/h (vs baseline 50)
    "ablation_17_low_reward_scale"  # reward_scale=0.01 (vs baseline 0.05)
    "ablation_18_high_reward_scale" # reward_scale=0.1 (vs baseline 0.05)
)

echo "========================================"
echo "Phase 3 & 4: Environment and Reward Ablations"
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
echo "Phase 3 & 4 Training Complete!"
echo "========================================"
echo "End time: $(date)"
echo "Total experiments: $total_experiments"
echo ""

# Show final status
$PYTHON_BIN -c "
import os
phase_dir = '$OUTPUT_DIR'
ablations = ['ablation_3_small_hidden', 'ablation_4_large_hidden', 'ablation_13_low_velocity', 'ablation_14_high_velocity', 'ablation_17_low_reward_scale', 'ablation_18_high_reward_scale']
print('Final Status:')
print('='*50)
total_seeds = 0
for abl in ablations:
    abl_path = os.path.join(phase_dir, abl)
    if os.path.exists(abl_path):
        seeds = [d for d in os.listdir(abl_path) if d.startswith('seed_')]
        print(f'✓ {abl}: {len(seeds)}/5 seeds')
        total_seeds += len(seeds)
    else:
        print(f'⏸  {abl}: 0/5 seeds')
print('='*50)
print(f'Total: {total_seeds}/30 experiments ({total_seeds*100//30}%)')
"

echo ""
echo "Next step: Run generalization testing"
echo "  $PYTHON_BIN test_ablation_generalization.py \\"
echo "    --ablation-dir $OUTPUT_DIR \\"
echo "    --output-dir ablation_results/phase3_phase4_analysis \\"
echo "    --velocities 5 10 20 30 50 70 80 90 100 \\"
echo "    --n-episodes 100 \\"
echo "    --ablations ${ABLATIONS[@]}"
echo ""
