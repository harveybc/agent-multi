. <scratch>/env.sh
PRE=<worktree>
cd $S
echo "== BEFORE (original scripts, agent-multi 3644ece9)"
cp -r $S/scripts_orig $S/run_orig 2>/dev/null
python -I $S/run_orig/D_native_fixtures.py $PRE $HS > $S/C_D_before_out.json 2> $S/C_D_before_err.txt; echo "D exit $?"
AM_ROOT=$PRE HS_ROOT=$HS python -I $S/run_orig/D2_native_more.py > $S/C_D2_before_out.json 2> $S/C_D2_before_err.txt; echo "D2 exit $?"
echo "== AFTER (adapted scripts, agent-multi 2318b0e3)"
python -I $S/scripts/D_native_fixtures.py $AM $HS > $S/C_D_after_out.json 2> $S/C_D_after_err.txt; echo "D exit $?"
AM_ROOT=$AM HS_ROOT=$HS python -I $S/scripts/D2_native_more.py > $S/C_D2_after_out.json 2> $S/C_D2_after_err.txt; echo "D2 exit $?"
