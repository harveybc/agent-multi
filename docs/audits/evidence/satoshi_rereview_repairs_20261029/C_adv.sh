. <scratch>/env.sh
RUN=<scratch>
PRE=<worktree>
cd $S
echo "== BEFORE: original E_adversarial.py on 3644ece9"
AM_WORKTREE=$PRE HS_ENGINE_ROOT=$HS python -I $S/run_orig/E_adversarial.py $S/run_orig $RUN > $S/C_E_adversarial_before_out.json 2> $S/C_E_adversarial_before_err.txt; echo "exit $?"
echo "== AFTER: original E_adversarial.py UNADAPTED on 2318b0e3 (expected to fail at migrated call sites)"
AM_WORKTREE=$AM HS_ENGINE_ROOT=$HS python -I $S/scripts_orig/E_adversarial.py $S/scripts_orig $RUN > $S/C_E_adversarial_unadapted_after_out.json 2> $S/C_E_adversarial_unadapted_after_err.txt; echo "exit $?"; tail -n 2 $S/C_E_adversarial_unadapted_after_err.txt
echo "== AFTER: adapted F10/DA12/DA13 on 2318b0e3"
AM_WORKTREE=$AM HS_ENGINE_ROOT=$HS python -I $S/scripts/C_F10_DA12_DA13.py $S/scripts $RUN > $S/C_F10_DA12_DA13_out.json 2> $S/C_F10_DA12_DA13_err.txt; echo "exit $?"; tail -n 5 $S/C_F10_DA12_DA13_err.txt
echo "== F07 pure sweep BEFORE (3644ece9)"
SWEEP_ONLY=1 AM_ROOT=$PRE HS_ROOT=$HS python -I $S/scripts/C_F07_DA24.py > $S/C_F07_sweep_before_out.json 2> $S/C_F07_sweep_before_err.txt; echo "exit $?"
echo "== F07 + DA24 AFTER (2318b0e3)"
AM_ROOT=$AM HS_ROOT=$HS python -I $S/scripts/C_F07_DA24.py > $S/C_F07_DA24_out.json 2> $S/C_F07_DA24_err.txt; echo "exit $?"; tail -n 5 $S/C_F07_DA24_err.txt
