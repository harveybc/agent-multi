#!/usr/bin/env python
"""Write the TRAIN-only StandardScaler (ddof=0) artifact used by M07's forecast cells for the
manifest's 83 features, so it can be bound to an encoder export identity (parity receipt --scaler).
M07's scaler_identity string is recorded verbatim; its hash recipe is M07's, NOT recomputed here."""
import argparse, hashlib, json
import numpy as np, pandas as pd
ap = argparse.ArgumentParser()
ap.add_argument("--view", required=True); ap.add_argument("--manifest", required=True)
ap.add_argument("--train-rows", type=int, nargs=2, default=[0, 13699]); ap.add_argument("--m07-identity", required=True)
ap.add_argument("--out", required=True); a = ap.parse_args()
feats = json.load(open(a.manifest))["features"]
df = pd.read_csv(a.view, usecols=feats).iloc[a.train_rows[0]:a.train_rows[1]]
x = df.to_numpy(dtype=np.float64)
doc = {"schema": "rl_temporal.scaler_artifact.v1", "kind": "StandardScaler(ddof=0), TRAIN rows only",
       "train_rows": a.train_rows, "view_sha256": hashlib.sha256(open(a.view, "rb").read()).hexdigest(),
       "feature_order": feats, "mean": x.mean(0).tolist(), "std": x.std(0, ddof=0).tolist(),
       "m07_scaler_identity_declared": a.m07_identity,
       "note": "recomputed here from the view; equality with M07's identity hash is NOT verified (recipe not published)"}
doc["scaler_sha256"] = hashlib.sha256(json.dumps({k: doc[k] for k in ("feature_order", "mean", "std", "train_rows")}, sort_keys=True).encode()).hexdigest()
open(a.out, "w").write(json.dumps(doc, indent=1) + "\n"); print(doc["scaler_sha256"], len(feats))
