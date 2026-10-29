#!/bin/bash
source $(dirname $0)/env.sh
cd $AM
rm -rf $S/out/analysis_replay
/usr/bin/time -v $PY tools/analyze_crossed_account_results.py --run $S/run --engine-root $STRATEGY_ENGINE_ROOT --export $S/out/analysis_replay --replay-p1 2>&1 | grep -v "^\s*\(Page\|Socket\|Signals\|Swaps\|File system\|Voluntary\|Involuntary\|Average\|Exit\|Minor\|Major\|Percent\)"
