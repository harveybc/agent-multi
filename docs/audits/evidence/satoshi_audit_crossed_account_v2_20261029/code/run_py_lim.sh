#!/bin/bash
source $(dirname $0)/env.sh
export LIMIT=$1; shift
cd $S/code
/usr/bin/time -f "MAXRSS_KB %M ELAPSED %e" $PY -I "$@"
