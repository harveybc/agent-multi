from __future__ import annotations

import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from app.campaign_supervisor import PLAN_SCHEMA, PROFILE_SCHEMA, _domain_semantic_hash
from doin_node.versioning import compute_component_versions


FAKE_DOIN = r'''
import argparse, base64, hashlib, json, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--log-level'); a=p.parse_args()
c=json.loads(Path(a.config).read_text()); domain=c['domains'][0]['domain_id']; started=time.time()
counter=Path(c['launch_counter_file']); counter.parent.mkdir(parents=True,exist_ok=True)
with counter.open('a') as f: f.write('start\n')
global_counter=Path(c['global_launch_counter_file']); global_counter.parent.mkdir(parents=True,exist_ok=True)
with global_counter.open('a') as f: f.write(c['node_label']+'\n')
model=('model:'+domain).encode(); digest=hashlib.sha256(model).hexdigest(); performance=float(c.get('performance',0.1))
seed=int(c['domains'][0]['optimization_config']['ga_seed'])
pop={'population':[{'x':0.5}],'generation':0,'stage_idx':0,'bootstrap_seed':seed,'bootstrap_domain_id':domain}
pop_fingerprint=hashlib.sha256(json.dumps(pop,sort_keys=True,separators=(',',':')).encode()).hexdigest()
genesis_hash='genesis-hash'
population_hash='population-'+hashlib.sha256(domain.encode()).hexdigest()
peer_id=hashlib.sha256(c['node_label'].encode()).hexdigest()
class H(BaseHTTPRequestHandler):
 def log_message(self,*args): pass
 def do_GET(self):
  if self.path=='/status':
   value={'status':'healthy','peer_id':peer_id[:12],'domains':{domain:{'converged':time.time()-started>0.25,'best_performance':performance}},'peers':3}
  elif self.path=='/chain/status':
   value={'chain_height':2,'tip_hash':population_hash,'finalized_height':0,'component_versions':c['component_versions']}
  elif self.path=='/chain/block/0':
   value={'hash':genesis_hash,'transactions':[]}
  elif self.path=='/chain/block/1':
   value={'hash':population_hash,'transactions':[{'id':'shared-'+domain,'tx_type':'optimae_accepted','domain_id':domain,'peer_id':'bootstrap','payload':{'_shared_population':pop,'_shared_population_fingerprint':pop_fingerprint,'_shared_population_seed':seed,'performance':0.0}}, {'id':'tx-'+domain,'tx_type':'optimae_accepted','domain_id':domain,'peer_id':'bootstrap','payload':{'verified_performance':performance,'parameters':{'x':0.5,'_model_b64':base64.b64encode(model).decode()},'champion_metrics':{'risk_adjusted_total_return':performance,'model_artifact_sha256':digest,'model_artifact_bytes':len(model),'model_artifact_format':'stable_baselines3_zip'}}}]}
  elif self.path.startswith('/api/shared/candidates'):
   value={'domain_id':domain,'generation':0,'pop_size':1,'bootstrap_seed':seed,'population_fingerprint':pop_fingerprint,'evaluated':1,'claimed':0,'free':0,'candidates':[{'index':0,'state':'evaluated','generation':0,'fitness':performance}]}
  else: self.send_response(404); self.end_headers(); return
  body=json.dumps(value).encode(); self.send_response(200); self.send_header('Content-Type','application/json'); self.send_header('Content-Length',str(len(body))); self.end_headers(); self.wfile.write(body)
ThreadingHTTPServer(('127.0.0.1',int(c['port'])),H).serve_forever()
'''


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=0.5) as response:
        return json.loads(response.read())


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_three_supervisors_advance_two_jobs_in_order_from_one_bootstrap(tmp_path: Path):
    agent_root = Path(__file__).resolve().parents[2]
    fake_root = tmp_path / "fake-doin"
    package = fake_root / "doin_node"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "cli.py").write_text(FAKE_DOIN)
    supervisor_ports = [_free_port() for _ in range(3)]
    worker_ports = [_free_port() for _ in range(4)]
    participants = [
        {"node_id": "omega", "supervisor_url": f"http://127.0.0.1:{supervisor_ports[0]}", "workers": ["omega"]},
        {"node_id": "dragon", "supervisor_url": f"http://127.0.0.1:{supervisor_ports[1]}", "workers": ["dragon"]},
        {"node_id": "gamma", "supervisor_url": f"http://127.0.0.1:{supervisor_ports[2]}", "workers": ["gamma-0", "gamma-1"]},
    ]
    worker_ids = [worker for participant in participants for worker in participant["workers"]]
    component_versions = compute_component_versions()
    jobs = []
    for job_index in range(2):
        domain_id = f"integration-domain-{job_index}"
        configs = {}
        semantic_hash = None
        for worker_index, worker_id in enumerate(worker_ids):
            config = {
                "node_label": worker_id,
                "port": worker_ports[worker_index],
                "launch_counter_file": str(tmp_path / "launches" / f"{job_index}-{worker_id}.txt"),
                "global_launch_counter_file": str(tmp_path / "launches" / f"{job_index}-global.txt"),
                "performance": 0.1 + job_index,
                "require_deterministic_seed": True,
                "component_versions": component_versions,
                "domains": [{
                    "domain_id": domain_id,
                    "higher_is_better": True,
                    "optimization_config": {
                        "shared_population": True,
                        "shared_population_size": 1,
                        "ga_seed": 1701,
                        "runtime_overlay": f"{worker_id}.json",
                        "hyperparameter_bounds": {"x": [0.0, 1.0]},
                    },
                    "param_bounds": {"x": [0.0, 1.0]},
                }],
            }
            config_path = fake_root / f"job-{job_index}-{worker_id}.json"
            config_path.write_text(json.dumps(config))
            configs[worker_id] = str(config_path)
            semantic_hash = _domain_semantic_hash(config)
        jobs.append({
            "ordinal": job_index,
            "job_id": f"job-{job_index}",
            "domain_id": domain_id,
            "higher_is_better": True,
            "domain_semantic_hash": semantic_hash,
            "worker_configs": configs,
        })
    plan = {
        "schema_version": PLAN_SCHEMA,
        "plan_id": "integration-plan",
        "participants": participants,
        "jobs": jobs,
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    profiles = []
    for participant, supervisor_port in zip(participants, supervisor_ports):
        profile = {
            "schema_version": PROFILE_SCHEMA,
            "node_id": participant["node_id"],
            "plan_file": str(plan_path),
            "state_dir": str(tmp_path / "state" / participant["node_id"]),
            "listen_host": "127.0.0.1",
            "listen_port": supervisor_port,
            "poll_seconds": 0.1,
            "peer_timeout_seconds": 0.2,
            "convergence_stability_seconds": 0.2,
            "stop_timeout_seconds": 1.0,
                "worker_restart_limit": 3,
                "component_versions": component_versions,
                "workers": {
                worker_id: {
                    "doin_node_root": str(fake_root),
                    "python": sys.executable,
                }
                for worker_id in participant["workers"]
            },
        }
        profile_path = tmp_path / f"{participant['node_id']}-profile.json"
        profile_path.write_text(json.dumps(profile))
        profiles.append(profile_path)

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(agent_root), str(fake_root)))
    supervisors = [
        subprocess.Popen(
            [sys.executable, "-m", "app.campaign_supervisor", "--profile", str(profile)],
            cwd=agent_root,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        for profile in profiles
    ]
    try:
        deadline = time.monotonic() + 20
        statuses = {}
        while time.monotonic() < deadline:
            try:
                statuses = {
                    participant["node_id"]: _get_json(participant["supervisor_url"] + "/api/status")
                    for participant in participants
                }
            except Exception:
                time.sleep(0.1)
                continue
            if all(status.get("phase") == "complete" for status in statuses.values()):
                break
            time.sleep(0.1)
        assert statuses and all(status["phase"] == "complete" for status in statuses.values())
        for participant in participants:
            history = _get_json(participant["supervisor_url"] + "/api/history")["history"]
            assert [item["job_id"] for item in history] == ["job-0", "job-1"]
            assert all(item["status"] == "completed" for item in history)
            assert all(Path(item["artifact_path"]).is_file() for item in history)
        for job_index in range(2):
            launch_order = (
                tmp_path / "launches" / f"{job_index}-global.txt"
            ).read_text().splitlines()
            assert launch_order == worker_ids
            for worker_id in worker_ids:
                launches = (tmp_path / "launches" / f"{job_index}-{worker_id}.txt").read_text().splitlines()
                assert launches == ["start"]
    finally:
        for process in supervisors:
            process.send_signal(signal.SIGTERM)
        for process in supervisors:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        for participant in participants:
            state_path = tmp_path / "state" / participant["node_id"] / "state.json"
            if not state_path.exists():
                continue
            state = json.loads(state_path.read_text())
            for worker in (state.get("workers") or {}).values():
                pid = worker.get("pid")
                if pid:
                    try:
                        os.kill(int(pid), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
