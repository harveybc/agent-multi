"""M4-C19: the deterministic generator bank.

One canonical API emits arrays and a manifest from a sealed
generator identity. DEVELOPMENT, CALIBRATION and CONFIRMATION
identities are disjoint BY CONSTRUCTION (the role is inside the
seed derivation and the id), every scale/threshold/noise
parameter is fitted on the TRAIN slice only, temporal tasks
persist the clean latent signal and the realized disturbance
separately, and consumers recompute digests from the bytes they
use — a producer-declared digest is never enough.

Structured Boolean families: identity, majority, dnf3, parity4.
Negative control: random_label (never a structured family).
Positive control: easy_constant (deliberately easy).
Temporal families: sine, chirp, am, discontinuity, state_space.
Noise regimes (temporal only): clean, white, colored, impulsive,
heteroscedastic. Boolean x noise products other than `clean` are
REFUSED as nonsensical rather than silently manufactured.

CPU only. No financial data. No live authority.
"""
import hashlib
import json

import numpy as np

MASTER_PHRASE = "m4_intervention_generators_2026_09_09"
ROLES = ("DEVELOPMENT", "CALIBRATION", "CONFIRMATION")
BOOL_FAMILIES = ("identity", "majority", "dnf3", "parity4")
TEMPORAL_FAMILIES = ("sine", "chirp", "am", "discontinuity",
                     "state_space")
CONTROL_FAMILIES = ("random_label", "easy_constant")
FAMILIES = BOOL_FAMILIES + TEMPORAL_FAMILIES + CONTROL_FAMILIES
NOISE_REGIMES = ("clean", "white", "colored", "impulsive",
                 "heteroscedastic")
N_IN = 8            # boolean input bits / temporal window length
N_TRAIN = 192
N_HELD = 64
FS_HZ = 1.0         # temporal sampling rate (declared, verified)


class GeneratorRefusal(SystemExit):
    def __init__(self, msg):
        super().__init__(f"REFUSED: {msg}")


def _seed(*parts) -> int:
    h = hashlib.sha256(("|".join(
        [MASTER_PHRASE, *map(str, parts)])).encode())
    return int.from_bytes(h.digest()[:8], "big")


def _arr_sha(a) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(a, dtype=np.float64)
                             ).tobytes()).hexdigest()


def generator_id(role, family, noise, idx) -> str:
    validate_cell(role, family, noise, idx)
    return f"{role}-{family}-{noise}-g{idx}"


def validate_cell(role, family, noise, idx):
    if role not in ROLES:
        raise GeneratorRefusal(f"unknown generator role {role!r}")
    if family not in FAMILIES:
        raise GeneratorRefusal(f"unknown family {family!r}")
    if noise not in NOISE_REGIMES:
        raise GeneratorRefusal(f"unknown noise regime {noise!r}")
    if family in BOOL_FAMILIES + CONTROL_FAMILIES and \
            noise != "clean":
        raise GeneratorRefusal(
            f"nonsensical product: Boolean/control family "
            f"{family!r} admits only the `clean` regime — noise "
            "regimes are defined for temporal observation "
            "processes, and the design never silently "
            "manufactures the combination")
    if type(idx) is not int or idx < 0 or idx > 9999:
        raise GeneratorRefusal("generator index out of domain")


def _bool_inputs(rng, n):
    return rng.choice([-1.0, 1.0], size=(n, N_IN))


def _bool_target(family, X, rng):
    if family == "identity":
        return (X[:, 0] > 0).astype(np.float64)
    if family == "majority":
        return (X[:, :5].sum(axis=1) > 0).astype(np.float64)
    if family == "dnf3":
        # three fixed conjunctions of three literals, drawn once
        # from the generator's own stream
        terms = []
        for _ in range(3):
            cols = rng.choice(N_IN, size=3, replace=False)
            signs = rng.choice([-1.0, 1.0], size=3)
            terms.append((cols, signs))
        out = np.zeros(len(X), dtype=bool)
        for cols, signs in terms:
            out |= (X[:, cols] * signs > 0).all(axis=1)
        return out.astype(np.float64)
    if family == "parity4":
        cols = rng.choice(N_IN, size=4, replace=False)
        return ((X[:, cols] > 0).sum(axis=1) % 2
                ).astype(np.float64)
    raise GeneratorRefusal(f"not a Boolean family: {family!r}")


def _latent_series(family, rng, n):
    t = np.arange(n, dtype=np.float64) / FS_HZ
    if family == "sine":
        f = rng.uniform(0.02, 0.08)
        ph = rng.uniform(0, 2 * np.pi)
        return np.sin(2 * np.pi * f * t + ph)
    if family == "chirp":
        f0 = rng.uniform(0.01, 0.03)
        f1 = rng.uniform(0.06, 0.12)
        k = (f1 - f0) / max(n / FS_HZ, 1.0)
        return np.sin(2 * np.pi * (f0 * t + 0.5 * k * t * t))
    if family == "am":
        fc = rng.uniform(0.08, 0.16)
        fm = rng.uniform(0.005, 0.02)
        return (1.0 + 0.6 * np.sin(2 * np.pi * fm * t)) * \
            np.sin(2 * np.pi * fc * t)
    if family == "discontinuity":
        n_seg = int(rng.integers(3, 7))
        cuts = np.sort(rng.choice(
            np.arange(8, n - 8), size=n_seg - 1, replace=False))
        levels = rng.uniform(-1.5, 1.5, size=n_seg)
        out = np.empty(n)
        prev = 0
        for i, c in enumerate(list(cuts) + [n]):
            out[prev:c] = levels[i]
            prev = c
        return out
    if family == "state_space":
        a1 = rng.uniform(1.2, 1.6)
        a2 = -rng.uniform(0.5, 0.8)
        x = np.zeros(n + 2)
        drive = rng.standard_normal(n + 2) * 0.3
        for i in range(2, n + 2):
            x[i] = a1 * x[i - 1] + a2 * x[i - 2] + drive[i]
        return x[2:]
    raise GeneratorRefusal(f"not a temporal family: {family!r}")


def _disturbance(noise, rng, latent, train_slice):
    """Realized disturbance for a temporal series; every scale is
    fitted on the TRAIN slice only."""
    n = len(latent)
    base = 0.1 * float(np.std(latent[train_slice]) + 1e-12)
    if noise == "clean":
        return np.zeros(n)
    if noise == "white":
        return rng.standard_normal(n) * base
    if noise == "colored":
        w = rng.standard_normal(n)
        c = np.empty(n)
        c[0] = w[0]
        for i in range(1, n):
            c[i] = 0.8 * c[i - 1] + w[i]
        c = c / (np.std(c[train_slice]) + 1e-12) * base
        return c
    if noise == "impulsive":
        w = rng.standard_normal(n) * base
        mask = rng.uniform(size=n) < 0.05
        w[mask] += rng.standard_normal(int(mask.sum())) * 5 * base
        return w
    if noise == "heteroscedastic":
        scale_ref = np.abs(latent)
        denom = float(np.mean(scale_ref[train_slice]) + 1e-12)
        return rng.standard_normal(n) * base * \
            (0.3 + scale_ref / denom)
    raise GeneratorRefusal(f"unknown noise regime {noise!r}")


def _windows(series, w):
    n = len(series) - w
    X = np.empty((n, w))
    for i in range(n):
        X[i] = series[i:i + w]
    return X, series[w:]


def generate(role, family, noise, idx) -> dict:
    """The ONE canonical entry: arrays + manifest, deterministic
    for the same id, disjoint across roles by seed construction."""
    gid = generator_id(role, family, noise, idx)
    rng = np.random.default_rng(
        _seed(role, family, noise, idx))
    if family in BOOL_FAMILIES:
        n = N_TRAIN + N_HELD
        X = _bool_inputs(rng, n)
        y = _bool_target(family, X, rng)
        latent = None
        dist = None
    elif family == "random_label":
        n = N_TRAIN + N_HELD
        X = rng.standard_normal((n, N_IN))
        y = rng.choice([0.0, 1.0], size=n)
        latent = None
        dist = None
    elif family == "easy_constant":
        n = N_TRAIN + N_HELD
        X = rng.standard_normal((n, N_IN))
        y = np.full(n, 0.8) + rng.standard_normal(n) * 0.01
        latent = None
        dist = None
    else:
        total = N_TRAIN + N_HELD + N_IN
        lat = _latent_series(family, rng, total)
        train_slice = slice(0, N_TRAIN + N_IN)
        d = _disturbance(noise, rng, lat, train_slice)
        obs = lat + d
        X, y = _windows(obs, N_IN)
        _, y_lat = _windows(lat, N_IN)
        latent = y_lat
        dist = y - y_lat
    for name, a in (("X", X), ("y", y)):
        if not np.isfinite(np.asarray(a, dtype=np.float64)).all():
            raise GeneratorRefusal(
                f"generator {gid} produced non-finite {name}")
    out = {
        "generator_id": gid,
        "role": role, "family": family,
        "noise": noise if family in TEMPORAL_FAMILIES
        else "NOT_APPLICABLE",
        "index": idx,
        "X_train": X[:N_TRAIN], "y_train": y[:N_TRAIN],
        "X_held": X[N_TRAIN:N_TRAIN + N_HELD],
        "y_held": y[N_TRAIN:N_TRAIN + N_HELD],
    }
    if latent is not None:
        out["latent_train"] = latent[:N_TRAIN]
        out["latent_held"] = latent[N_TRAIN:N_TRAIN + N_HELD]
        out["disturbance_train"] = dist[:N_TRAIN]
        out["disturbance_held"] = dist[N_TRAIN:N_TRAIN + N_HELD]
    out["manifest"] = build_manifest(out)
    return out


def build_manifest(g) -> dict:
    is_temporal = g["family"] in TEMPORAL_FAMILIES
    man = {
        "schema": "agent_multi.m4_generator_manifest.v1",
        "generator_id": g["generator_id"],
        "role": g["role"], "family": g["family"],
        "noise": g["noise"], "index": g["index"],
        "dtype": "float64",
        "n_in": N_IN, "n_train": N_TRAIN, "n_held": N_HELD,
        "sampling_rate_hz": FS_HZ if is_temporal else None,
        "target_rule": (
            "next-step regression on the observed series over an "
            f"{N_IN}-step window" if is_temporal else
            "fixed Boolean function of the input bits"
            if g["family"] in BOOL_FAMILIES else
            "control target (never structured)"),
        "train_only_fitting": True,
        "X_train_sha256": _arr_sha(g["X_train"]),
        "y_train_sha256": _arr_sha(g["y_train"]),
        "X_held_sha256": _arr_sha(g["X_held"]),
        "y_held_sha256": _arr_sha(g["y_held"]),
        "latent_train_sha256":
            _arr_sha(g["latent_train"]) if is_temporal else None,
        "disturbance_train_sha256":
            _arr_sha(g["disturbance_train"])
            if is_temporal else None,
        "summary": {
            "y_train_mean": round(float(np.mean(g["y_train"])), 8),
            "y_train_std": round(float(np.std(g["y_train"])), 8),
        },
    }
    body = {k: man[k] for k in sorted(man)}
    man["manifest_sha256"] = hashlib.sha256(json.dumps(
        body, sort_keys=True).encode()).hexdigest()
    return man


def consumer_verify(g) -> dict:
    """C19: the CONSUMER recomputes every digest from the bytes it
    will use; producer-declared values are claims, not evidence.
    Also proves the latent/disturbance decomposition and the
    declared sampling identity for temporal tasks."""
    man = g["manifest"]
    body = {k: man[k] for k in sorted(man)
            if k != "manifest_sha256"}
    if hashlib.sha256(json.dumps(body, sort_keys=True).encode()
                      ).hexdigest() != man["manifest_sha256"]:
        raise GeneratorRefusal(
            "manifest self-digest does not re-derive")
    for key in ("X_train", "y_train", "X_held", "y_held"):
        if _arr_sha(g[key]) != man[f"{key}_sha256"]:
            raise GeneratorRefusal(
                f"declared digest for {key} does not match the "
                "bytes in use")
    if g["family"] in TEMPORAL_FAMILIES:
        if man["sampling_rate_hz"] != FS_HZ:
            raise GeneratorRefusal(
                "manifest mislabels the sampling frequency")
        recon = g["latent_train"] + g["disturbance_train"]
        if not np.allclose(recon, g["y_train"], atol=1e-12):
            raise GeneratorRefusal(
                "latent + disturbance does not reconstruct the "
                "observed training target")
    fresh = generate(g["role"], g["family"],
                     g["noise"] if g["noise"] != "NOT_APPLICABLE"
                     else "clean", g["index"])
    if fresh["manifest"]["manifest_sha256"] != \
            man["manifest_sha256"]:
        raise GeneratorRefusal(
            "generator bytes are not deterministic for this id")
    return {"verified": True,
            "generator_id": g["generator_id"]}


def assert_role_disjointness(family="sine", noise="white",
                             idx=0) -> dict:
    """Executable proof that the three role namespaces yield
    different bytes for otherwise identical coordinates."""
    shas = {}
    for role in ROLES:
        g = generate(role, family, noise, idx)
        shas[role] = g["manifest"]["y_train_sha256"]
    if len(set(shas.values())) != len(ROLES):
        raise GeneratorRefusal(
            "role namespaces are NOT disjoint — identical bytes "
            "across roles")
    return shas
