#!/bin/bash
# Monitor Neither RNN nor LN training progress

echo "=========================================="
echo "Neither Model Training Progress Monitor"
echo "=========================================="
echo ""

# Get the latest results directory
RESULTS_DIR=$(ls -td ablation_results/neither_rnn_nor_ln_* 2>/dev/null | head -1)

if [ -z "$RESULTS_DIR" ]; then
    echo "No training directory found!"
    exit 1
fi

echo "Results directory: $RESULTS_DIR"
echo ""

# Count completed seeds
COMPLETED_SEEDS=$(find "$RESULTS_DIR" -type d -name "seed_*" | wc -l | tr -d ' ')
echo "Completed seeds: $COMPLETED_SEEDS / 5"
echo ""

# Show current training log
echo "Recent training output:"
echo "----------------------------------------"
tail -30 neither_training.log
echo "----------------------------------------"
echo ""

# Show completed training directories
echo "Completed training runs:"
for seed_dir in "$RESULTS_DIR"/seed_*; do
    if [ -d "$seed_dir" ]; then
        seed=$(basename "$seed_dir" | sed 's/seed_//')
        echo "  - Seed $seed:"

        if [ -d "$seed_dir/a3c/models" ]; then
            echo "    ✓ A3C training completed"
        fi

        if [ -d "$seed_dir/individual/models" ]; then
            echo "    ✓ Individual training completed"
        fi
    fi
done

echo ""
echo "=========================================="
echo "Use 'tail -f neither_training.log' to watch live progress"
echo "=========================================="
