#!/bin/bash
#
# Monitor Generalization Testing Progress
#

echo "========================================================================"
echo "Generalization Test Monitor"
echo "========================================================================"
echo ""

# Check if process is running
if ps aux | grep "test_ablation_generalization.py" | grep -v grep > /dev/null; then
    echo "✓ Generalization test is RUNNING"
    echo ""

    # Show process details
    echo "Process details:"
    ps aux | grep "test_ablation_generalization.py" | grep -v grep | head -1
    echo ""
else
    echo "✗ Generalization test is NOT running"
    echo ""
fi

# Check for completed results
echo "Results generated:"
if [ -d "ablation_results/analysis" ]; then
    echo ""
    ls -lh ablation_results/analysis/*.csv 2>/dev/null | while read -r line; do
        echo "  $line"
    done
    echo ""

    # Show summary if available
    if [ -f "ablation_results/analysis/generalization_summary.csv" ]; then
        echo "========================================================================"
        echo "Summary Results (when available):"
        echo "========================================================================"
        echo ""
        cat ablation_results/analysis/generalization_summary.csv | column -t -s','
        echo ""
    fi
else
    echo "  No results yet"
    echo ""
fi

echo "========================================================================"
echo "Estimated completion time:"
echo "========================================================================"
echo ""
echo "Total tests to run:"
echo "  - 4 ablations × 5 seeds = 20 model sets"
echo "  - Each set: 9 velocities × 100 episodes"
echo "  - Plus: Individual workers (5 workers per seed)"
echo ""
echo "Estimated time: ~2-3 hours total"
echo ""
echo "To check detailed progress:"
echo "  ps aux | grep test_ablation_generalization"
echo ""
echo "To view results when complete:"
echo "  cat ablation_results/analysis/generalization_summary.csv"
echo "========================================================================"
