#!/bin/bash

# Monitor Phase 2 Hyperparameter Ablation Progress

OUTPUT_DIR="ablation_results/phase2_hyperparameters"
LOG_FILE="ablation_results/logs/phase2_execution.log"

echo "=========================================="
echo "Phase 2 Hyperparameter Ablation Progress"
echo "=========================================="
echo ""

# Check if training is running
if ps aux | grep -q "[r]un_phase2_hyperparameters.sh"; then
    echo "Status: 🟢 RUNNING"
else
    echo "Status: 🔴 NOT RUNNING or COMPLETED"
fi
echo ""

# Count completed models
total_expected=60  # 6 ablations × 5 seeds × 2 models (a3c + individual)
completed=$(find "$OUTPUT_DIR" -name "*_final.pth" 2>/dev/null | wc -l | tr -d ' ')

echo "Progress: $completed / $total_expected models completed"
echo "Percentage: $(( completed * 100 / total_expected ))%"
echo ""

# Show completed ablations by seed
echo "Completed Models by Ablation:"
echo "-------------------------------------------"
for ablation in ablation_5_low_entropy ablation_6_high_entropy ablation_7_low_value_loss ablation_8_high_value_loss ablation_9_low_lr ablation_10_high_lr; do
    count=$(find "$OUTPUT_DIR" -path "*/${ablation}_seed*/models/*_final.pth" 2>/dev/null | wc -l | tr -d ' ')
    expected=10  # 5 seeds × 2 models each
    if [ $count -eq $expected ]; then
        echo "✓ $ablation: $count/$expected"
    elif [ $count -gt 0 ]; then
        echo "⏳ $ablation: $count/$expected"
    else
        echo "⏸  $ablation: $count/$expected"
    fi
done
echo ""

# Show last 15 lines of log
echo "Recent Activity:"
echo "-------------------------------------------"
tail -15 "$LOG_FILE" 2>/dev/null || echo "Log file not found"
echo ""

# Estimate time remaining
if [ $completed -gt 0 ] && ps aux | grep -q "[r]un_phase2_hyperparameters.sh"; then
    echo "-------------------------------------------"
    echo "Note: Each experiment takes ~1 hour"
    experiments_done=$(( completed / 2 ))
    experiments_left=$(( 30 - experiments_done ))
    echo "Estimated experiments remaining: $experiments_left"
    echo "Estimated time remaining: ~${experiments_left} hours"
fi
