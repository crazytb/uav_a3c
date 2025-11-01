#!/bin/bash
#
# Monitor Ablation Study Progress
# Shows current training status and progress
#

echo "========================================================================"
echo "Ablation Study Progress Monitor"
echo "========================================================================"
echo ""

# Check for running Python processes
echo "Running training processes:"
ps aux | grep "run_single_ablation.py\|main_train.py" | grep -v grep | head -10
echo ""

# Check most recent runs
echo "Most recent training runs:"
ls -lht runs/ | head -6
echo ""

# Check ablation results
echo "Ablation results collected:"
if [ -d "ablation_results/high_priority" ]; then
    for ablation_dir in ablation_results/high_priority/*/; do
        if [ -d "$ablation_dir" ]; then
            ablation_name=$(basename "$ablation_dir")
            seed_count=$(find "$ablation_dir" -maxdepth 1 -type d -name "seed_*" | wc -l | tr -d ' ')
            echo "  $ablation_name: $seed_count/5 seeds completed"
        fi
    done
else
    echo "  No results yet"
fi
echo ""

# Check training logs if available
echo "Recent training progress (last A3C run):"
latest_a3c=$(ls -td runs/a3c_* 2>/dev/null | head -1)
if [ -n "$latest_a3c" ] && [ -f "$latest_a3c/training_log.csv" ]; then
    echo "  Run: $(basename $latest_a3c)"
    total_episodes=$(tail -1 "$latest_a3c/training_log.csv" | cut -d',' -f1)
    latest_reward=$(tail -1 "$latest_a3c/training_log.csv" | cut -d',' -f2)
    echo "  Episodes: $total_episodes"
    echo "  Latest reward: $latest_reward"
else
    echo "  No training log available yet"
fi
echo ""

# Estimated time remaining
echo "========================================================================"
echo "Time Estimates:"
echo "========================================================================"
echo "Per seed (2000 episodes): ~8-10 hours"
echo "Per ablation (5 seeds):   ~40-50 hours"
echo "All 4 ablations:          ~160-200 hours"
echo ""
echo "Current ablation order:"
echo "  1. ablation_1_no_rnn      (No RNN)"
echo "  2. ablation_2_no_layer_norm (No LayerNorm)"
echo "  3. ablation_15_few_workers (3 workers)"
echo "  4. ablation_16_many_workers (10 workers)"
echo ""
echo "To check detailed logs:"
echo "  tail -f ablation_results/logs/ablation_execution_*.log"
echo ""
echo "To check latest training progress:"
echo "  tail -f runs/a3c_*/training_log.csv"
echo "========================================================================"
