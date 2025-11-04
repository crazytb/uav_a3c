#!/bin/bash
# Run generalization test for Neither RNN nor LN ablation

PYTHON_PATH=~/miniconda/envs/torch-cert/bin/python

echo "=========================================="
echo "Neither RNN nor LN - Generalization Test"
echo "=========================================="
echo ""

# Find the latest neither results directory
RESULTS_DIR=$(ls -td ablation_results/neither_rnn_nor_ln_* 2>/dev/null | head -1)

if [ -z "$RESULTS_DIR" ]; then
    echo "ERROR: No neither results directory found!"
    echo "Please run training first with: ./run_neither_rnn_nor_ln.sh"
    exit 1
fi

echo "Testing models from: $RESULTS_DIR"
echo ""

# Check if all 5 seeds are completed
COMPLETED_SEEDS=$(find "$RESULTS_DIR" -type d -name "seed_*" | wc -l | tr -d ' ')

if [ "$COMPLETED_SEEDS" -lt 5 ]; then
    echo "WARNING: Only $COMPLETED_SEEDS/5 seeds completed"
    echo "Continue anyway? (y/n)"
    read -r response
    if [ "$response" != "y" ]; then
        echo "Aborted."
        exit 1
    fi
fi

echo "Running generalization test..."
echo "Estimated time: 2-3 hours (5 seeds × 9 velocities × 100 episodes × 2 methods)"
echo ""

$PYTHON_PATH test_ablation_generalization.py \
    --ablation-dir "$RESULTS_DIR" \
    --ablation-name ablation_neither_rnn_nor_ln \
    --velocities 5 10 20 30 50 70 80 90 100 \
    --n-episodes 100

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Generalization Test Complete!"
    echo "=========================================="
    echo ""
    echo "Results saved to: $RESULTS_DIR/generalization_results.csv"
    echo ""
    echo "Next steps:"
    echo "1. Analyze results:"
    echo "   python -c \"import pandas as pd; df = pd.read_csv('$RESULTS_DIR/generalization_results.csv'); print(df.groupby('method')[['mean_reward', 'std_reward']].agg(['mean', 'std']))\""
    echo ""
    echo "2. Compare with baseline:"
    echo "   - Baseline (RNN+LN): A3C 49.57 ± 14.35 vs Individual 38.22 ± 16.24 (Gap: 29.7%)"
    echo "   - Check your results in the CSV file"
    echo ""
else
    echo ""
    echo "ERROR: Generalization test failed!"
    exit 1
fi
