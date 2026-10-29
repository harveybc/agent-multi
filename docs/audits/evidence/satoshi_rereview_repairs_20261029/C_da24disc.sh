. <scratch>/env.sh
echo "== BEFORE 3644ece9"; python -I $S/scripts/C_DA24_discrimination.py <worktree> $HS > $S/C_DA24_disc_before_out.json; echo "exit $?"
echo "== AFTER 2318b0e3"; python -I $S/scripts/C_DA24_discrimination.py $AM $HS > $S/C_DA24_disc_after_out.json; echo "exit $?"
