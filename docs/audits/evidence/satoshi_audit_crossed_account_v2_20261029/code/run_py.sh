#!/bin/bash
source $(dirname $0)/env.sh
cd $S/code
/usr/bin/time -f "MAXRSS_KB %M ELAPSED %e" $PY -I "$@"
