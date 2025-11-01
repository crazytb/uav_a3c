#!/bin/bash
#
# Run Phase 1: Resource Constraint Ablations
# - ablation_11_limited_cloud (500 units)
# - ablation_12_abundant_cloud (2000 units)
#
# Total: 2 ablations × 5 seeds = 10 experiments
#

set -e

PYTHON="/Users/crazytb/miniconda/envs/torch-cert/bin/python"
SCRIPT="run_single_ablation.py"
OUTPUT_DIR="ablation_results/resource_constraints"
SEEDS=(42 123 456 789 1024)

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p ablation_results/logs

# Log file
LOG_FILE="ablation_results/logs/resource_ablations.log"

echo "================================================================================" | tee -a "$LOG_FILE"
echo "Phase 1: Resource Constraint Ablations" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "Start time: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Counter
total_experiments=10
current=0

# Ablation 11: Limited Cloud (500 units)
echo "================================================================================" | tee -a "$LOG_FILE"
echo "Ablation 11: Limited Cloud Resources (500 units)" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

for seed in "${SEEDS[@]}"; do
    current=$((current + 1))
    echo "[$current/$total_experiments] Running ablation_11_limited_cloud with seed $seed..." | tee -a "$LOG_FILE"
    echo "Start: $(date)" | tee -a "$LOG_FILE"

    $PYTHON $SCRIPT \
        --ablation ablation_11_limited_cloud \
        --seed $seed \
        --output-dir $OUTPUT_DIR \
        >> "$LOG_FILE" 2>&1

    if [ $? -eq 0 ]; then
        echo "✓ Completed: ablation_11_limited_cloud seed $seed" | tee -a "$LOG_FILE"
    else
        echo "✗ Failed: ablation_11_limited_cloud seed $seed" | tee -a "$LOG_FILE"
    fi
    echo "End: $(date)" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
done

# Ablation 12: Abundant Cloud (2000 units)
echo "================================================================================" | tee -a "$LOG_FILE"
echo "Ablation 12: Abundant Cloud Resources (2000 units)" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

for seed in "${SEEDS[@]}"; do
    current=$((current + 1))
    echo "[$current/$total_experiments] Running ablation_12_abundant_cloud with seed $seed..." | tee -a "$LOG_FILE"
    echo "Start: $(date)" | tee -a "$LOG_FILE"

    $PYTHON $SCRIPT \
        --ablation ablation_12_abundant_cloud \
        --seed $seed \
        --output-dir $OUTPUT_DIR \
        >> "$LOG_FILE" 2>&1

    if [ $? -eq 0 ]; then
        echo "✓ Completed: ablation_12_abundant_cloud seed $seed" | tee -a "$LOG_FILE"
    else
        echo "✗ Failed: ablation_12_abundant_cloud seed $seed" | tee -a "$LOG_FILE"
    fi
    echo "End: $(date)" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
done

echo "================================================================================" | tee -a "$LOG_FILE"
echo "Phase 1 Complete!" | tee -a "$LOG_FILE"
echo "End time: $(date)" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Results saved to: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
