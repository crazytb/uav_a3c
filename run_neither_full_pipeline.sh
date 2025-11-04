#!/bin/bash
# Complete pipeline: Train Neither models and run generalization test

echo "=========================================="
echo "Neither RNN nor LN - Full Pipeline"
echo "=========================================="
echo ""

PYTHON_PATH=~/miniconda/envs/torch-cert/bin/python

# Step 1: Training
echo "Step 1: Training neither models (5 seeds)..."
echo "Estimated time: 8-10 hours"
echo ""

./run_neither_rnn_nor_ln.sh

if [ $? -ne 0 ]; then
    echo "Training failed!"
    exit 1
fi

# Step 2: Find results directory
RESULTS_DIR=$(ls -td ablation_results/neither_rnn_nor_ln_* 2>/dev/null | head -1)

if [ -z "$RESULTS_DIR" ]; then
    echo "Results directory not found!"
    exit 1
fi

echo ""
echo "=========================================="
echo "Step 2: Running generalization test..."
echo "Estimated time: 2-3 hours"
echo "=========================================="
echo ""

$PYTHON_PATH test_baseline_generalization.py --baseline-dir "$RESULTS_DIR"

if [ $? -ne 0 ]; then
    echo "Generalization test failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo ""
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "Next steps:"
echo "1. Analyze results: python analyze_baseline_results.py"
echo "2. Compare with baseline configuration"
echo "3. Update documentation with findings"
echo ""
