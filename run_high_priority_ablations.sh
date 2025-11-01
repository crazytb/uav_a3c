#!/bin/bash
#
# Run High Priority Ablations
# Executes 4 high-priority ablation studies with 5 seeds each
#

set -e  # Exit on error

# Configuration
PYTHON="/Users/crazytb/miniconda/envs/torch-cert/bin/python"
SEEDS=(42 123 456 789 1024)
OUTPUT_DIR="ablation_results/high_priority"
LOG_DIR="ablation_results/logs"

# High priority ablations in order
ABLATIONS=(
    "ablation_1_no_rnn"
    "ablation_2_no_layer_norm"
    "ablation_15_few_workers"
    "ablation_16_many_workers"
)

# Create log directory
mkdir -p "$LOG_DIR"

# Main execution log
MAIN_LOG="$LOG_DIR/ablation_execution_$(date +%Y%m%d_%H%M%S).log"

echo "========================================================================"
echo "High Priority Ablation Study Execution"
echo "========================================================================"
echo "Start time: $(date)"
echo "Total ablations: ${#ABLATIONS[@]}"
echo "Seeds per ablation: ${#SEEDS[@]}"
echo "Total experiments: $((${#ABLATIONS[@]} * ${#SEEDS[@]}))"
echo "Output directory: $OUTPUT_DIR"
echo "Main log: $MAIN_LOG"
echo "========================================================================"
echo ""

# Log header
{
    echo "========================================================================"
    echo "High Priority Ablation Study Execution Log"
    echo "========================================================================"
    echo "Start time: $(date)"
    echo ""
} > "$MAIN_LOG"

# Counter for progress tracking
total_experiments=$((${#ABLATIONS[@]} * ${#SEEDS[@]}))
current_experiment=0

# Run each ablation
for ablation in "${ABLATIONS[@]}"; do
    echo "--------------------------------------------------------------------"
    echo "Starting ablation: $ablation"
    echo "--------------------------------------------------------------------"
    echo ""

    {
        echo "========================================================================"
        echo "Ablation: $ablation"
        echo "Start time: $(date)"
        echo "========================================================================"
        echo ""
    } >> "$MAIN_LOG"

    # Run each seed
    for seed in "${SEEDS[@]}"; do
        current_experiment=$((current_experiment + 1))

        echo "[$current_experiment/$total_experiments] Running $ablation with seed=$seed"
        echo "  Start time: $(date)"

        # Create experiment-specific log
        exp_log="$LOG_DIR/${ablation}_seed_${seed}_$(date +%Y%m%d_%H%M%S).log"

        # Run experiment
        {
            echo "Experiment: $ablation, Seed: $seed"
            echo "Start time: $(date)"
            echo ""

            $PYTHON run_single_ablation.py \
                --ablation "$ablation" \
                --seed "$seed" \
                --output-dir "$OUTPUT_DIR" 2>&1

            exit_code=$?
            echo ""
            echo "Exit code: $exit_code"
            echo "End time: $(date)"

            if [ $exit_code -eq 0 ]; then
                echo "Status: SUCCESS"
            else
                echo "Status: FAILED"
            fi
        } 2>&1 | tee "$exp_log"

        # Check if experiment succeeded
        if [ ${PIPESTATUS[0]} -eq 0 ]; then
            echo "  Status: ✓ SUCCESS"
            {
                echo "  Seed $seed: SUCCESS"
            } >> "$MAIN_LOG"
        else
            echo "  Status: ✗ FAILED"
            {
                echo "  Seed $seed: FAILED"
            } >> "$MAIN_LOG"
            echo "  Error log: $exp_log"
        fi

        echo "  End time: $(date)"
        echo ""

        # Small delay between experiments
        sleep 5
    done

    {
        echo "End time: $(date)"
        echo ""
    } >> "$MAIN_LOG"

    echo ""
done

# Summary
echo "========================================================================"
echo "All experiments completed!"
echo "========================================================================"
echo "End time: $(date)"
echo "Results directory: $OUTPUT_DIR"
echo "Main log: $MAIN_LOG"
echo "========================================================================"

{
    echo "========================================================================"
    echo "All experiments completed!"
    echo "End time: $(date)"
    echo "========================================================================"
} >> "$MAIN_LOG"

echo ""
echo "Next steps:"
echo "  1. Run generalization tests: python test_ablation_generalization.py"
echo "  2. Analyze results: python analyze_high_priority_ablations.py"
echo "  3. Generate paper tables: python generate_paper_tables.py"
