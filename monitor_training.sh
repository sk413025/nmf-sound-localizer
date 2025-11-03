#!/bin/bash
# Monitor DTMinPointerV2 functional test progress
# Usage: watch -n 5 ./monitor_training.sh

LOG_FILE="results/dt_v2_functional/run.log"

if [ ! -f "$LOG_FILE" ]; then
    echo "Log file not found: $LOG_FILE"
    exit 1
fi

echo "==================================================="
echo "DTMinPointerV2 Functional Test - Training Monitor"
echo "==================================================="
echo ""

# Extract latest epoch info
echo "Latest Progress:"
echo "----------------"
tail -6 "$LOG_FILE" | head -5

echo ""
echo "==================================================="
echo "Recent Epochs Summary:"
echo "==================================================="
grep "^Epoch" "$LOG_FILE" | tail -3

echo ""
echo "==================================================="
echo "Best Model So Far:"
echo "==================================================="
grep "Best:" "$LOG_FILE" | tail -1

echo ""
echo "Process Status:"
ps aux | grep "dt_pointer_ldv_v2.py" | grep -v grep | awk '{print "Running (PID: " $2 ", CPU: " $3 "%, MEM: " $4 "%)"}'
if [ $? -ne 0 ]; then
    echo "Training completed or not running"
fi
