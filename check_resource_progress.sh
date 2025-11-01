#!/bin/bash
echo "==================================="
echo "Resource Ablation Progress"
echo "==================================="
completed=$(ls -d ablation_results/resource_constraints/*/seed_*/ 2>/dev/null | wc -l | tr -d ' ')
echo "Completed: $completed / 10 experiments"
pct=$((completed * 10))
echo "Progress: ${pct}%"
echo ""
echo "Running processes:"
ps aux | grep -E "ablation_1[12]_(limited|abundant)" | grep -v grep || echo "  None"
echo ""
echo "Last log entries:"
tail -10 ablation_results/logs/resource_ablations.log 2>/dev/null || echo "  Log file not created yet"
