. <scratch>/env.sh
RUN=<scratch>
AM_WORKTREE=$AM HS_ENGINE_ROOT=$HS python -I $S/scripts/C_F10_DA12_DA13.py $S/scripts $RUN > $S/C_F10_DA12_DA13_out.json 2> $S/C_F10_DA12_DA13_err.txt; echo "exit $?"
