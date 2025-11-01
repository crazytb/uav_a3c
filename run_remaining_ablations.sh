#!/bin/bash
#
# Run Remaining High Priority Ablations
# Continues from seed 123 onwards
#

set -e

PYTHON="/Users/crazytb/miniconda/envs/torch-cert/bin/python"
OUTPUT_DIR="ablation_results/high_priority"
LOG_DIR="ablation_results/logs"

mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="$LOG_DIR/remaining_execution_${TIMESTAMP}.log"

echo "========================================================================"
echo "Continuing High Priority Ablation Execution"
echo "========================================================================"
echo "Start time: $(date)"
echo "Main log: $MAIN_LOG"
echo "========================================================================"
echo ""

{
    echo "========================================================================"
    echo "Remaining Ablation Execution Log"
    echo "Start time: $(date)"
    echo "========================================================================"
    echo ""
} > "$MAIN_LOG"

# Wait for current seed 123 to complete
echo "Waiting for ablation_1_no_rnn seed 123 to complete..."
echo "This is currently running in the background."
echo ""

# Function to run single experiment
run_experiment() {
    local ablation=$1
    local seed=$2

    echo "[$ablation / seed $seed] Starting at $(date)"

    exp_log="$LOG_DIR/${ablation}_seed_${seed}_$(date +%Y%m%d_%H%M%S).log"

    $PYTHON run_single_ablation.py \
        --ablation "$ablation" \
        --seed "$seed" \
        --output-dir "$OUTPUT_DIR" 2>&1 | tee "$exp_log"

    exit_code=${PIPESTATUS[0]}

    if [ $exit_code -eq 0 ]; then
        echo "[$ablation / seed $seed] ✓ SUCCESS at $(date)"
        echo "  $ablation seed $seed: SUCCESS" >> "$MAIN_LOG"
    else
        echo "[$ablation / seed $seed] ✗ FAILED at $(date)"
        echo "  $ablation seed $seed: FAILED" >> "$MAIN_LOG"
    fi

    echo ""
    return $exit_code
}

# Wait for current process to finish
echo "Checking if seed 123 is still running..."
while ps aux | grep "ablation_1_no_rnn --seed 123" | grep -v grep > /dev/null; do
    echo "  Still running... ($(date))"
    sleep 60
done
echo "Seed 123 completed!"
echo ""

# Continue with remaining seeds for ablation_1
for seed in 456 789 1024; do
    run_experiment "ablation_1_no_rnn" "$seed"
done

echo "========================================================================"
echo "ablation_1_no_rnn completed! (5/5 seeds)"
echo "========================================================================"
echo ""

# Run ablation_2_no_layer_norm
echo "Starting ablation_2_no_layer_norm..."
for seed in 42 123 456 789 1024; do
    run_experiment "ablation_2_no_layer_norm" "$seed"
done

echo "========================================================================"
echo "ablation_2_no_layer_norm completed! (5/5 seeds)"
echo "========================================================================"
echo ""

# Run ablation_15_few_workers
echo "Starting ablation_15_few_workers..."
for seed in 42 123 456 789 1024; do
    run_experiment "ablation_15_few_workers" "$seed"
done

echo "========================================================================"
echo "ablation_15_few_workers completed! (5/5 seeds)"
echo "========================================================================"
echo ""

# Run ablation_16_many_workers
echo "Starting ablation_16_many_workers..."
for seed in 42 123 456 789 1024; do
    run_experiment "ablation_16_many_workers" "$seed"
done

echo "========================================================================"
echo "All ablations completed!"
echo "End time: $(date)"
echo "========================================================================"

{
    echo ""
    echo "========================================================================"
    echo "All experiments completed!"
    echo "End time: $(date)"
    echo "========================================================================"
} >> "$MAIN_LOG"

echo ""
echo "Summary:"
echo "  - ablation_1_no_rnn: 5/5 seeds ✓"
echo "  - ablation_2_no_layer_norm: 5/5 seeds ✓"
echo "  - ablation_15_few_workers: 5/5 seeds ✓"
echo "  - ablation_16_many_workers: 5/5 seeds ✓"
echo ""
echo "Next steps:"
echo "  1. Analyze results: python analyze_high_priority_ablations.py"
echo "  2. Test generalization: python test_ablation_generalization.py"
echo "  3. Generate paper outputs: python generate_paper_tables.py"
