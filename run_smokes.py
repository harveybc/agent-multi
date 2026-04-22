import json
import os
import subprocess
import sys

configs = ["dqn_btc_1h_twelve_atr.json", "dqn_eth_1h_twelve_atr.json", "sac_btc_1h_twelve_atr.json"]

for cfg in configs:
    print(f"=== smoke {cfg} ===")
    cfg_path = os.path.join("examples/config", cfg)
    if not os.path.exists(cfg_path):
        print(f"Config {cfg_path} not found")
        continue

    with open(cfg_path, 'r') as f:
        c = json.load(f)

    c['total_timesteps'] = 1000
    c['learning_starts'] = 200
    c['save_model'] = f'/tmp/_smoke_{cfg}.zip'
    c['results_file'] = f'/tmp/_smoke_{cfg}_summary.json'
    c['save_config'] = f'/tmp/_smoke_{cfg}_cfg_out.json'
    c['device'] = 'cpu'

    with open('/tmp/_smoke_cfg.json', 'w') as f:
        json.dump(c, f, indent=2)

    log_file = "/tmp/_smoke.log"
    with open(log_file, "w") as f:
        res = subprocess.run(["agent-multi", "--load_config", "/tmp/_smoke_cfg.json", "--quiet_mode"], stdout=f, stderr=subprocess.STDOUT)
    
    print(f"EXIT={res.returncode}")
    if res.returncode == 0:
        summary_path = c['results_file']
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                summary = json.load(f)
                print(json.dumps(summary, indent=2)[:500] + "...")
        else:
            print("Summary file not found.")
    else:
        with open(log_file, "r") as f:
            lines = f.readlines()
            print("".join(lines[-30:]))
