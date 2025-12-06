#!/bin/bash
# Quick status check script

if [ -f "full_results.csv" ]; then
    LINES=$(wc -l < full_results.csv)
    RESPONSES=$((LINES - 1))
    PERCENT=$((RESPONSES * 100 / 775))
    echo "Progress: $RESPONSES/775 responses ($PERCENT%)"
    
    if [ $RESPONSES -eq 775 ]; then
        echo "✓ SURVEY COMPLETE!"
        exit 0
    fi
else
    echo "Survey starting... (file not yet created)"
fi
exit 1
