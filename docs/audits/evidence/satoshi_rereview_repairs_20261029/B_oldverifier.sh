. <scratch>/env.sh
cd $AM
python tools/verify_dynamic_account_results.py --run <scratch> --engine-root $HS --export $S/B_oldverifier_export 2>&1 | tail -4
echo "exit=$?"
