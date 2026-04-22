#!/usr/bin/env python
"""
NEAT-style Optimizer Plugin (Parameters-as-Genes)

Implements a NeuroEvolution of Augmenting Topologies inspired approach
where hyperparameters are treated as genes that can be activated/deactivated.
Key features:
  - Variable-length genomes: each individual has a subset of active parameters
  - Speciation: groups individuals with similar parameter structures
  - Fitness sharing: within-species fitness adjustment to maintain diversity
  - Structural mutations: add/remove parameters organically
  - Value mutations: standard gaussian mutation on parameter values
  - Innovation numbers: global tracking for crossover alignment
  - Level-2 early stopping: patience-based convergence detection (same as default)

Uses the same candidate_worker subprocess for evaluation as default_optimizer.
"""

import copy
import random
import numpy as np
import time
import json
import csv
import gc
import os
import sys
import subprocess
import tempfile
import pty
import select
from pathlib import Path
from app.plugin_loader import load_plugin

# Reverse mapping: GA encodes activation as int [0..7], model needs string.
ACTIVATION_INDEX_TO_NAME = [
    "relu", "elu", "selu", "tanh", "sigmoid", "swish", "gelu", "leaky_relu",
]

# Encoding type mapping: GA encodes as int [0..2], preprocessor needs string.
ENCODING_INDEX_TO_NAME = ["none", "sincos", "onehot"]

# Loss type mapping: GA encodes as int [0..4], ioin needs string.
LOSS_TYPE_INDEX_TO_NAME = [
    "mae", "trend_sigma", "pearson_structural", "soft_dtw", "combined_diff",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_repo_path(p):
    if not p:
        return None
    try:
        pp = Path(str(p))
        if pp.is_absolute():
            return str(pp)
        return str((_repo_root() / pp).resolve())
    except Exception:
        return str(p)


def _json_sanitize(obj):
    """Recursively convert numpy/tensor types into JSON-serializable Python types."""
    try:
        import tensorflow as tf
        if isinstance(obj, tf.Tensor):
            obj = obj.numpy()
    except Exception:
        pass
    if callable(obj) and not isinstance(obj, type):
        return None
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()
                if k != "optimization_callbacks" and k != "_non_serializable_keys"}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]
    return obj


def _atomic_json_dump(path, payload):
    """Write JSON atomically to avoid corrupt files on crashes."""
    tmp_path = f"{path}.tmp"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(tmp_path, "w") as f:
        json.dump(_json_sanitize(payload), f, indent=4)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


# ── NEAT Genome Representation ───────────────────────────────

class NeatGene:
    """A single gene representing one hyperparameter."""
    __slots__ = ("innovation", "param_name", "value")

    def __init__(self, innovation, param_name, value):
        self.innovation = innovation
        self.param_name = param_name
        self.value = value

    def copy(self):
        return NeatGene(self.innovation, self.param_name, self.value)


class NeatGenome:
    """Variable-length genome for NEAT-style optimization."""

    def __init__(self):
        self.genes = {}          # innovation_number -> NeatGene
        self.fitness = None      # Raw fitness (lower is better)
        self.adjusted_fitness = None  # After fitness sharing
        self.species_id = None
        # Evaluation metrics
        self.val_mae = None
        self.naive_mae = None
        self.train_mae = None
        self.train_naive_mae = None
        self.test_mae = None
        self.test_naive_mae = None
        self.model_summary = None
        self.hyper_dict = None

    def to_hyper_dict(self, param_types):
        """Convert genome to hyperparameter dict for evaluation."""
        hyper = {}
        for gene in self.genes.values():
            key = gene.param_name
            val = gene.value
            if key == "use_log1p_features":
                hyper[key] = ["typical_price"] if int(round(val)) == 1 else None
            elif key == "positional_encoding":
                hyper[key] = bool(int(round(val)))
            elif key == "use_temporal_features":
                hyper[key] = bool(int(round(val)))
            elif key == "add_window_stats":
                hyper[key] = bool(int(round(val)))
            elif key == "add_multi_scale_returns":
                hyper[key] = bool(int(round(val)))
            elif key == "activation":
                act_idx = max(0, min(int(round(val)), len(ACTIVATION_INDEX_TO_NAME) - 1))
                hyper[key] = ACTIVATION_INDEX_TO_NAME[act_idx]
            elif key in ("hod_encoding", "dow_encoding", "moy_encoding"):
                enc_idx = max(0, min(int(round(val)), len(ENCODING_INDEX_TO_NAME) - 1))
                hyper[key] = ENCODING_INDEX_TO_NAME[enc_idx]
            elif key == "loss_type":
                lt_idx = max(0, min(int(round(val)), len(LOSS_TYPE_INDEX_TO_NAME) - 1))
                hyper[key] = LOSS_TYPE_INDEX_TO_NAME[lt_idx]
            elif param_types.get(key) == "int":
                hyper[key] = int(round(val))
            else:
                hyper[key] = val
        return hyper

    @property
    def active_params(self):
        return sorted(gene.param_name for gene in self.genes.values())

    @property
    def complexity(self):
        return len(self.genes)

    def deep_copy(self):
        g = NeatGenome()
        g.genes = {k: v.copy() for k, v in self.genes.items()}
        g.species_id = self.species_id
        g.fitness = self.fitness
        g.adjusted_fitness = self.adjusted_fitness
        g.val_mae = self.val_mae
        g.naive_mae = self.naive_mae
        g.train_mae = self.train_mae
        g.train_naive_mae = self.train_naive_mae
        g.test_mae = self.test_mae
        g.test_naive_mae = self.test_naive_mae
        g.model_summary = self.model_summary
        g.hyper_dict = self.hyper_dict
        return g

    def to_serializable(self):
        """For checkpoint saving."""
        return {
            "genes": {str(k): {"innovation": v.innovation, "param_name": v.param_name, "value": v.value}
                      for k, v in self.genes.items()},
            "fitness": self.fitness,
            "species_id": self.species_id,
        }

    @classmethod
    def from_serializable(cls, data):
        """Restore from checkpoint."""
        g = cls()
        for k, v in data.get("genes", {}).items():
            inn = int(k)
            g.genes[inn] = NeatGene(inn, v["param_name"], v["value"])
        g.fitness = data.get("fitness")
        g.species_id = data.get("species_id")
        return g


# ── Innovation Tracking ──────────────────────────────────────

class InnovationTracker:
    """Global innovation number assignment for parameters."""

    def __init__(self):
        self._counter = 0
        self._param_to_innovation = {}

    def get_innovation(self, param_name):
        if param_name not in self._param_to_innovation:
            self._counter += 1
            self._param_to_innovation[param_name] = self._counter
        return self._param_to_innovation[param_name]

    def to_serializable(self):
        return {"counter": self._counter, "map": dict(self._param_to_innovation)}

    @classmethod
    def from_serializable(cls, data):
        t = cls()
        t._counter = data.get("counter", 0)
        t._param_to_innovation = data.get("map", {})
        return t


# ── Species ──────────────────────────────────────────────────

class Species:
    """A group of structurally similar genomes."""

    def __init__(self, species_id, representative):
        self.id = species_id
        self.representative = representative  # NeatGenome
        self.members = []
        self.best_fitness = float("inf")
        self.generations_without_improvement = 0

    @property
    def size(self):
        return len(self.members)


def compatibility_distance(g1, g2, full_bounds, c1=1.0, c3=0.4):
    """Compute NEAT compatibility distance between two genomes.
    c1: coefficient for structural difference (disjoint/excess genes)
    c3: coefficient for value difference in matching genes
    """
    inn1 = set(g1.genes.keys())
    inn2 = set(g2.genes.keys())
    matching = inn1 & inn2
    disjoint_excess = len((inn1 - inn2) | (inn2 - inn1))
    N = max(len(g1.genes), len(g2.genes), 1)

    # Normalized weight difference for matching genes
    if matching:
        diffs = []
        for i in matching:
            gene1 = g1.genes[i]
            low, high = full_bounds[gene1.param_name]
            range_val = (high - low) if high != low else 1
            diffs.append(abs(g1.genes[i].value - g2.genes[i].value) / range_val)
        w_diff = sum(diffs) / len(diffs)
    else:
        w_diff = 0

    return c1 * disjoint_excess / N + c3 * w_diff


def speciate(population, species_list, full_bounds, threshold):
    """Assign each genome to a species based on compatibility distance."""
    # Clear old members
    for sp in species_list:
        sp.members = []

    unassigned = list(population)
    for genome in unassigned:
        placed = False
        for sp in species_list:
            if compatibility_distance(genome, sp.representative, full_bounds) < threshold:
                sp.members.append(genome)
                genome.species_id = sp.id
                placed = True
                break
        if not placed:
            # Create new species
            new_id = max((s.id for s in species_list), default=0) + 1
            new_sp = Species(new_id, genome.deep_copy())
            new_sp.members.append(genome)
            genome.species_id = new_id
            species_list.append(new_sp)

    # Remove empty species
    species_list[:] = [s for s in species_list if s.members]

    # Update representatives (random member from each species)
    for sp in species_list:
        sp.representative = random.choice(sp.members).deep_copy()


def adjust_fitness(species_list, higher_is_better=False):
    """Fitness sharing: adjusted_fitness = raw_fitness / species_size."""
    _worst = float("-inf") if higher_is_better else float("inf")
    for sp in species_list:
        for genome in sp.members:
            if genome.fitness is not None and np.isfinite(genome.fitness):
                genome.adjusted_fitness = genome.fitness / max(sp.size, 1)
            else:
                genome.adjusted_fitness = _worst


# ── Mutation Operators ───────────────────────────────────────

def mutate_add_param(genome, all_params, full_bounds, innovation_tracker, add_prob=0.15):
    """Structural mutation: add 1-3 random new parameters per trigger."""
    if random.random() > add_prob:
        return False
    active = {g.param_name for g in genome.genes.values()}
    candidates = [p for p in all_params if p not in active]
    if not candidates:
        return False
    # Add 1-3 params at once (weighted toward 1) to accelerate exploration
    n_to_add = min(len(candidates), random.choices([1, 2, 3], weights=[0.50, 0.35, 0.15], k=1)[0])
    for new_param in random.sample(candidates, n_to_add):
        inn = innovation_tracker.get_innovation(new_param)
        low, high = full_bounds[new_param]
        if isinstance(low, int) and isinstance(high, int):
            value = random.randint(low, high)
        else:
            value = random.uniform(low, high)
        genome.genes[inn] = NeatGene(inn, new_param, value)
    return True


def mutate_remove_param(genome, min_params=2, remove_prob=0.05):
    """Structural mutation: remove a random parameter."""
    if random.random() > remove_prob or len(genome.genes) <= min_params:
        return False
    remove_key = random.choice(list(genome.genes.keys()))
    del genome.genes[remove_key]
    return True


def mutate_values(genome, full_bounds, mutpb=0.2, sigma_scale=0.15,
                  frozen_params=None):
    """Type-aware mutation on parameter values.

    - Boolean [0,1] int params: low-probability bit-flip (10% when triggered)
    - Categorical int params (range > 1): neighbor step ±1 or ±2
    - Wide-range int params (range >= 10): Gaussian perturbation rounded to int
    - Float params: Gaussian perturbation (unchanged)

    *frozen_params*: set of param names to skip during mutation.
    """
    mutated = False
    _frozen = frozen_params or set()
    for gene in genome.genes.values():
        if gene.param_name in _frozen:
            continue
        if random.random() < mutpb:
            low, high = full_bounds[gene.param_name]
            if isinstance(low, int) and isinstance(high, int):
                span = high - low
                if span <= 1:
                    # Boolean: flip with 10% probability (not 50% coin-flip)
                    if random.random() < 0.1:
                        gene.value = 1 - int(round(gene.value))
                elif span <= 5:
                    # Categorical / small-range int: step ±1 (or ±2 rarely)
                    step = random.choice([-1, 1]) if random.random() > 0.2 else random.choice([-2, 2])
                    gene.value = max(low, min(high, int(round(gene.value)) + step))
                else:
                    # Wide-range int: Gaussian perturbation rounded to int
                    sigma = span * sigma_scale
                    gene.value = max(low, min(high, int(round(gene.value + random.gauss(0, sigma)))))
            else:
                sigma = (high - low) * sigma_scale
                gene.value = max(low, min(high, gene.value + random.gauss(0, sigma)))
            mutated = True
    return mutated


def clamp_genome(genome, full_bounds):
    """Ensure all gene values are within bounds."""
    for gene in genome.genes.values():
        low, high = full_bounds[gene.param_name]
        gene.value = max(low, min(high, gene.value))


# ── Crossover ────────────────────────────────────────────────

def neat_crossover(parent1, parent2):
    """NEAT-style crossover. Fitter parent contributes disjoint/excess genes."""
    # Ensure parent1 is fitter (lower fitness = better)
    if parent2.fitness is not None and (parent1.fitness is None or parent2.fitness < parent1.fitness):
        parent1, parent2 = parent2, parent1

    child = NeatGenome()
    p1_inn = set(parent1.genes.keys())
    p2_inn = set(parent2.genes.keys())

    # Matching genes: randomly inherit from either parent
    for inn in p1_inn & p2_inn:
        gene = random.choice([parent1.genes[inn], parent2.genes[inn]]).copy()
        child.genes[inn] = gene

    # Disjoint/excess genes: inherit from fitter parent (parent1)
    for inn in p1_inn - p2_inn:
        child.genes[inn] = parent1.genes[inn].copy()

    return child


# ── Staged Optimization ──────────────────────────────────────

# Known parameter groups for auto-stage detection
_FEATURE_PARAMS = {"window_size", "use_log1p_features", "positional_encoding",
                   "use_temporal_features", "add_window_stats", "add_multi_scale_returns",
                   "hod_encoding", "dow_encoding", "moy_encoding"}
_ARCH_PARAMS = {"tcn_filters", "tcn_kernel_size", "tcn_stack_layers",
                "tcn_dilations_per_stack", "tcn_head_layers", "tcn_head_units",
                "tcn_use_batch_norm", "tcn_use_layer_norm"}


def _build_default_stages(all_params, n_gens):
    """Auto-detect parameter stages from known parameter name patterns."""
    features = [p for p in all_params if p in _FEATURE_PARAMS]
    architecture = [p for p in all_params if p in _ARCH_PARAMS]
    training = [p for p in all_params if p not in _FEATURE_PARAMS and p not in _ARCH_PARAMS]

    stages = []
    if features:
        stages.append({"name": "features", "params": features})
    if architecture:
        stages.append({"name": "architecture", "params": architecture})
    if training:
        stages.append({"name": "training", "params": training})
    # Always add a refinement stage with all params
    stages.append({"name": "refinement", "params": "all"})

    # Allocate generations proportionally (refinement gets fewer)
    n_stages = len(stages)
    if n_stages <= 1:
        stages[0]["generations"] = n_gens
    else:
        # Refinement gets ~20% of budget, rest split evenly
        refine_gens = max(3, n_gens // 5)
        remaining = n_gens - refine_gens
        per_stage = max(3, remaining // (n_stages - 1))
        for s in stages[:-1]:
            s["generations"] = per_stage
        stages[-1]["generations"] = n_gens - per_stage * (n_stages - 1)

    return stages


# ── Plugin Class ─────────────────────────────────────────────

class Plugin:
    """NEAT-style optimizer plugin with parameters-as-genes."""

    plugin_params = {
        "population_size": 20,
        "n_generations": 10,
        "mutpb": 0.2,
        "optimization_patience": 6,
        "hyperparameter_bounds": {
            "learning_rate": (1e-5, 1e-2),
            "num_layers": (1, 5),
            "layer_size": (16, 256),
        },
        # NEAT-specific defaults
        "neat_initial_params": None,       # List of initial params (None = first N from bounds)
        "neat_add_param_prob": 0.35,       # Probability of adding a parameter
        "neat_remove_param_prob": 0.05,    # Probability of removing a parameter
        "neat_compatibility_threshold": 2.0,  # Speciation distance threshold
        "neat_min_params": 6,              # Minimum active parameters per genome
        "neat_survival_rate": 0.5,         # Fraction of species that reproduces
        "neat_interspecies_mate_rate": 0.01,  # Cross-species mating probability
        "neat_elitism": 1,                 # Number of elites per species
    }
    plugin_debug_vars = ["population_size", "n_generations", "mutpb", "optimization_patience"]

    def __init__(self):
        self.params = self.plugin_params.copy()

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            self.params[key] = value

    def get_debug_info(self):
        return {var: self.params.get(var) for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        debug_info.update(self.get_debug_info())

    def optimize(self, predictor_plugin, preprocessor_plugin, config):
        """Run NEAT-style optimization.

        Same interface as default_optimizer: returns dict of best hyperparameters.
        """
        # ── Setup ────────────────────────────────────────────
        if "predictor_plugin" in config:
            config["plugin"] = config["predictor_plugin"]
        elif "plugin" not in config:
            config["plugin"] = "default_predictor"

        target_plugin_name = config.get("target_plugin", "default_target")
        try:
            target_class, _ = load_plugin("target.plugins", target_plugin_name)
            target_plugin = target_class()
            target_plugin.set_params(**config)
        except Exception as e:
            print(f"Failed to load Target Plugin: {e}")
            raise

        full_bounds = config.get("hyperparameter_bounds", self.params.get("hyperparameter_bounds"))
        all_params = list(full_bounds.keys())

        # Determine parameter types
        param_types = {}
        for key in all_params:
            low, up = full_bounds[key]
            param_types[key] = "int" if isinstance(low, int) and isinstance(up, int) else "float"

        # NEAT config
        population_size = config.get("population_size", self.params.get("population_size", 20))
        n_generations = config.get("n_generations", self.params.get("n_generations", 10))
        patience = config.get("optimization_patience", self.params.get("optimization_patience", 6))
        add_param_prob = config.get("neat_add_param_prob", self.params["neat_add_param_prob"])
        remove_param_prob = config.get("neat_remove_param_prob", self.params["neat_remove_param_prob"])
        compat_threshold = config.get("neat_compatibility_threshold", self.params["neat_compatibility_threshold"])
        min_params = config.get("neat_min_params", self.params["neat_min_params"])
        survival_rate = config.get("neat_survival_rate", self.params["neat_survival_rate"])
        interspecies_mate_rate = config.get("neat_interspecies_mate_rate", self.params["neat_interspecies_mate_rate"])
        neat_elitism = config.get("neat_elitism", self.params["neat_elitism"])
        mutpb = config.get("mutpb", self.params.get("mutpb", 0.2))

        # Initial parameters
        initial_params = config.get("neat_initial_params")
        if not initial_params:
            initial_params = all_params[:min(min_params, len(all_params))]

        # ── Stage schedule ───────────────────────────────────
        _raw_stages = config.get("optimization_stages", None)
        if _raw_stages is None:
            _raw_stages = _build_default_stages(all_params, n_generations)

        _stage_schedule = []
        _gen_cursor = 0
        for _si, _s in enumerate(_raw_stages):
            _s_params = ([p for p in _s["params"] if p in full_bounds]
                         if _s["params"] != "all" else list(all_params))
            _s_gens = _s.get("generations", max(3, n_generations // len(_raw_stages)))
            _stage_entry = {
                "name": _s["name"],
                "stage_idx": _si,
                "active_params": _s_params,
                "frozen_params": set(all_params) - set(_s_params),
                "start_gen": _gen_cursor,
                "end_gen": _gen_cursor + _s_gens,
            }
            if "patience" in _s:
                _stage_entry["patience"] = _s["patience"]
            _stage_schedule.append(_stage_entry)
            _gen_cursor += _s_gens
        _total_stage_gens = _gen_cursor
        _staged_mode = len(_stage_schedule) > 1
        _current_stage = _stage_schedule[0]

        # Extract default values for frozen params from config
        _param_defaults = {}
        for p in all_params:
            if p in config:
                v = config[p]
                if p == "use_log1p_features":
                    v = 1 if v else 0
                elif p in ("positional_encoding", "use_temporal_features",
                           "add_window_stats", "add_multi_scale_returns"):
                    v = 1 if v else 0
                elif p == "activation" and isinstance(v, str):
                    v = ACTIVATION_INDEX_TO_NAME.index(v) if v in ACTIVATION_INDEX_TO_NAME else 0
                elif p in ("hod_encoding", "dow_encoding", "moy_encoding") and isinstance(v, str):
                    v = ENCODING_INDEX_TO_NAME.index(v) if v in ENCODING_INDEX_TO_NAME else 0
                elif p == "loss_type" and isinstance(v, str):
                    v = LOSS_TYPE_INDEX_TO_NAME.index(v) if v in LOSS_TYPE_INDEX_TO_NAME else 0
                try:
                    _param_defaults[p] = float(v)
                except (TypeError, ValueError):
                    _param_defaults[p] = float((full_bounds[p][0] + full_bounds[p][1]) / 2)
            else:
                low, high = full_bounds[p]
                _param_defaults[p] = float((low + high) / 2)

        # ── Innovation tracking ──────────────────────────────
        innovation_tracker = InnovationTracker()
        # Pre-assign innovation numbers for all params (deterministic ordering)
        for p in all_params:
            innovation_tracker.get_innovation(p)

        # ── Initialize tracking state ────────────────────────
        self.eval_counter = 0
        self.total_eval_counter = 0
        self.current_gen = 0
        # Fitness direction: binary/direction targets use higher-is-better
        _is_binary = config.get("target_plugin") in ("binary_target", "direction_target")
        _hib = config.get("higher_is_better", _is_binary)
        self._higher_is_better = _hib
        _worst_fitness = float("-inf") if _hib else float("inf")
        def _fitness_better(a, b):
            return a > b if _hib else a < b
        def _best_of(iterable, default=None):
            return max(iterable, default=default if default is not None else _worst_fitness) if _hib \
                else min(iterable, default=default if default is not None else _worst_fitness)
        def _best_genome_key(g):
            """Sort key: best fitness first."""
            if g.fitness is None or not np.isfinite(g.fitness):
                return _worst_fitness
            return -g.fitness if _hib else g.fitness
        self.best_fitness_so_far = _worst_fitness
        self.patience_counter = 0
        self.best_val_mae_so_far = None
        self.best_naive_mae_so_far = None
        self.best_test_mae_so_far = None
        self.best_test_naive_mae_so_far = None
        self.best_train_mae_so_far = None
        self.best_train_naive_mae_so_far = None
        self.best_params_so_far = {}
        self.best_at_gen_start = _worst_fitness

        # NEAT-specific tracking (exposed to dashboard)
        self.neat_species_count = 0
        self.neat_avg_complexity = 0
        self.neat_max_complexity = 0
        self.neat_min_complexity = 0
        self.neat_species_details = []

        # ── Callbacks ────────────────────────────────────────
        _opt_callbacks = config.get("optimization_callbacks", {})

        # ── Seed initialization (per-node diversity) ─────────
        _base_seed = config.get("random_seed", 42)
        _seed_offset = config.get("node_seed_offset", 0)
        _effective_seed = _base_seed + _seed_offset
        if config.get("deterministic_training", False):
            random.seed(_effective_seed)
            np.random.seed(_effective_seed)
            print(f"[NEAT] Seeded RNG: base={_base_seed} + offset={_seed_offset} = {_effective_seed}")

        # ── Create initial population ────────────────────────
        def _create_genome(params_list):
            g = NeatGenome()
            for p in params_list:
                inn = innovation_tracker.get_innovation(p)
                low, high = full_bounds[p]
                if isinstance(low, int) and isinstance(high, int):
                    val = random.randint(low, high)
                else:
                    val = random.uniform(low, high)
                g.genes[inn] = NeatGene(inn, p, val)
            return g

        def _create_stage_genome(active_params_set, frozen_set, best_values, randomize_active=True):
            """Create genome with ALL params; frozen at best_values, active randomized or perturbed."""
            g = NeatGenome()
            for p in all_params:
                inn = innovation_tracker.get_innovation(p)
                low, high = full_bounds[p]
                if p in frozen_set:
                    val = best_values.get(p, _param_defaults[p])
                elif randomize_active:
                    if isinstance(low, int) and isinstance(high, int):
                        val = random.randint(low, high)
                    else:
                        val = random.uniform(low, high)
                else:
                    # Perturbed copy of best value (for elite seeds)
                    val = best_values.get(p, _param_defaults[p])
                    if isinstance(low, int) and isinstance(high, int):
                        span = high - low
                        if span > 1:
                            val = max(low, min(high, int(round(val + random.gauss(0, span * 0.15)))))
                    else:
                        val = max(low, min(high, val + random.gauss(0, (high - low) * 0.15)))
                g.genes[inn] = NeatGene(inn, p, float(val))
            return g

        def _build_stage_population(active_params_set, frozen_set, best_values):
            """Build population for a new stage: 25% perturbed elites + 75% random."""
            pop = []
            n_elite_seeds = max(2, population_size // 4)
            for _ in range(n_elite_seeds):
                pop.append(_create_stage_genome(active_params_set, frozen_set, best_values, randomize_active=False))
            for _ in range(population_size - n_elite_seeds):
                pop.append(_create_stage_genome(active_params_set, frozen_set, best_values, randomize_active=True))
            return pop

        if _staged_mode:
            # Stage 1: all params present, stage 1 active params randomized, rest at defaults
            population = _build_stage_population(
                set(_current_stage["active_params"]),
                _current_stage["frozen_params"],
                _param_defaults,
            )
        else:
            population = [_create_genome(initial_params) for _ in range(population_size)]

        species_list = []
        best_genome = None
        stats_history = []

        # ── Candidate history CSV ────────────────────────────
        _candidate_csv_path = config.get(
            "optimization_candidate_history",
            "optimization_candidate_history.csv",
        )
        _candidate_csv_path = _resolve_repo_path(_candidate_csv_path) or _candidate_csv_path
        _csv_columns = (
            ["total_eval", "generation", "candidate_in_gen", "stage_name",
             "species_id", "complexity", "is_champion"]
            + sorted(all_params)
            + ["fitness", "train_mcc", "train_f1", "val_mcc", "val_f1",
               "test_mcc", "test_f1", "champion_fitness"]
        )
        # Write header (overwrite if exists — resume=false means fresh start)
        try:
            os.makedirs(os.path.dirname(os.path.abspath(_candidate_csv_path)), exist_ok=True)
            with open(_candidate_csv_path, "w", newline="") as _cf:
                csv.writer(_cf).writerow(_csv_columns)
            print(f"[NEAT] Candidate history CSV: {_candidate_csv_path}")
        except Exception as _csv_err:
            print(f"[NEAT] Warning: could not create candidate CSV: {_csv_err}")
            _candidate_csv_path = None

        print(f"\n{'='*80}")
        print(f"[NEAT] NEAT-style Optimization Starting")
        print(f"[NEAT] Population: {population_size} | Generations: {_total_stage_gens} | Patience: {patience}")
        if _staged_mode:
            print(f"[NEAT] STAGED MODE: {len(_stage_schedule)} stages")
            for _ss in _stage_schedule:
                _sp = _ss.get('patience', patience)
                print(f"  Stage {_ss['stage_idx']+1} '{_ss['name']}': "
                      f"{len(_ss['active_params'])} params, {_ss['end_gen'] - _ss['start_gen']} gens "
                      f"(gens {_ss['start_gen']}-{_ss['end_gen']-1}), patience={_sp}")
        else:
            print(f"[NEAT] Initial parameters ({len(initial_params)}): {initial_params}")
        print(f"[NEAT] All available parameters ({len(all_params)}): {all_params}")
        print(f"[NEAT] Compatibility threshold: {compat_threshold}")
        print(f"[NEAT] Add param prob: {add_param_prob} | Remove param prob: {remove_param_prob}")
        print(f"{'='*80}\n")

        # ── Resume logic ────────────────────────────────────
        resume_enabled = config.get("optimization_resume", False)
        resume_path = _resolve_repo_path(config.get("optimization_resume_file"))
        params_path = _resolve_repo_path(config.get("optimization_parameters_file"))
        start_gen = 0

        # When resume is disabled, remove stale checkpoint/params files so they
        # cannot accidentally be picked up by any downstream code path.
        if not resume_enabled:
            for _stale in (resume_path, params_path):
                if _stale and os.path.exists(_stale):
                    try:
                        os.remove(_stale)
                        print(f"[NEAT] Removed stale file (resume=false): {_stale}")
                    except OSError:
                        pass

        if resume_enabled and resume_path and os.path.exists(resume_path):
            try:
                with open(resume_path, "r") as f:
                    checkpoint = json.load(f)
                # Restore innovation tracker
                if "innovation_tracker" in checkpoint:
                    innovation_tracker = InnovationTracker.from_serializable(checkpoint["innovation_tracker"])
                # Restore population
                if "population" in checkpoint:
                    population = [NeatGenome.from_serializable(gd) for gd in checkpoint["population"]]
                    print(f"[NEAT RESUME] Loaded {len(population)} genomes from checkpoint")
                # Restore optimizer state
                opt_state = checkpoint.get("optimizer_state", {})
                if opt_state.get("best_fitness_so_far") is not None:
                    self.best_fitness_so_far = float(opt_state["best_fitness_so_far"])
                if opt_state.get("best_val_mae_so_far") is not None:
                    self.best_val_mae_so_far = float(opt_state["best_val_mae_so_far"])
                if opt_state.get("best_naive_mae_so_far") is not None:
                    self.best_naive_mae_so_far = float(opt_state["best_naive_mae_so_far"])
                if opt_state.get("best_test_mae_so_far") is not None:
                    self.best_test_mae_so_far = float(opt_state["best_test_mae_so_far"])
                if opt_state.get("best_test_naive_mae_so_far") is not None:
                    self.best_test_naive_mae_so_far = float(opt_state["best_test_naive_mae_so_far"])
                if opt_state.get("best_train_mae_so_far") is not None:
                    self.best_train_mae_so_far = float(opt_state["best_train_mae_so_far"])
                if opt_state.get("best_train_naive_mae_so_far") is not None:
                    self.best_train_naive_mae_so_far = float(opt_state["best_train_naive_mae_so_far"])
                if opt_state.get("best_params_so_far"):
                    self.best_params_so_far = opt_state["best_params_so_far"]
                if opt_state.get("total_eval_counter") is not None:
                    self.total_eval_counter = int(opt_state["total_eval_counter"])
                if opt_state.get("patience_counter") is not None:
                    self.patience_counter = int(opt_state["patience_counter"])
                # Fallback: load best_params from optimization_parameters_file
                # for checkpoints saved before best_params_so_far was persisted
                if not self.best_params_so_far and params_path and os.path.exists(params_path):
                    try:
                        with open(params_path, "r") as pf:
                            self.best_params_so_far = json.load(pf)
                        print(f"[NEAT RESUME] Loaded best_params from {params_path}")
                    except Exception:
                        pass
                self.best_at_gen_start = float(self.best_fitness_so_far)
                start_gen = checkpoint.get("generation", 0) + 1
                print(f"[NEAT RESUME] Resuming from generation {start_gen}, "
                      f"best_fitness={self.best_fitness_so_far:.6f}, patience={self.patience_counter}")
            except Exception as e:
                print(f"[NEAT RESUME] Failed to load checkpoint: {e}")
                start_gen = 0
        elif resume_enabled and params_path and os.path.exists(params_path):
            # Load champion params into population[0]
            try:
                with open(params_path, "r") as f:
                    champ_params = json.load(f)
                champ_genome = NeatGenome()
                for p, v in champ_params.items():
                    if p in full_bounds:
                        inn = innovation_tracker.get_innovation(p)
                        # Convert special types back to numeric
                        if p == "use_log1p_features":
                            v = 1 if v else 0
                        elif p == "positional_encoding":
                            v = 1 if v else 0
                        elif p == "use_temporal_features":
                            v = 1 if v else 0
                        elif p == "activation" and isinstance(v, str):
                            v = ACTIVATION_INDEX_TO_NAME.index(v) if v in ACTIVATION_INDEX_TO_NAME else 0
                        elif p in ("hod_encoding", "dow_encoding", "moy_encoding") and isinstance(v, str):
                            v = ENCODING_INDEX_TO_NAME.index(v) if v in ENCODING_INDEX_TO_NAME else 0
                        elif p == "loss_type" and isinstance(v, str):
                            v = LOSS_TYPE_INDEX_TO_NAME.index(v) if v in LOSS_TYPE_INDEX_TO_NAME else 0
                        champ_genome.genes[inn] = NeatGene(inn, p, float(v))
                if champ_genome.genes:
                    population[0] = champ_genome
                    print(f"[NEAT RESUME] Injected champion with {len(champ_genome.genes)} params into population[0]")
            except Exception as e:
                print(f"[NEAT RESUME] Failed to load champion: {e}")

        end_gen = start_gen + n_generations

        # ── Evaluation function ──────────────────────────────
        def eval_genome(genome, gen):
            """Evaluate a genome using subprocess worker."""
            self.eval_counter += 1
            self.total_eval_counter += 1

            hyper_dict = genome.to_hyper_dict(param_types)
            genome.hyper_dict = hyper_dict

            print(f"\n--- [NEAT] Evaluating Candidate {self.eval_counter}/{population_size} | "
                  f"Gen {gen}/{_total_stage_gens} | Active Params: {genome.complexity}/{len(all_params)} | "
                  f"Species: {genome.species_id or '?'} | Total Evals: {self.total_eval_counter} ---")
            print(f"Active: {genome.active_params}")
            print(f"Params: {hyper_dict}")

            new_config = config.copy()
            new_config.update(hyper_dict)

            # Subprocess isolation
            for k in ("memory_log_file", "optimizer_resource_log_file", "batch_memory_log_file"):
                if new_config.get(k):
                    new_config[k] = _resolve_repo_path(new_config.get(k))
            new_config.setdefault(
                "memory_log_tag",
                f"neat_gen{gen}_cand{int(self.eval_counter)}",
            )

            with tempfile.TemporaryDirectory(prefix="neat_cand_") as td:
                in_path = os.path.join(td, "input.json")
                out_path = os.path.join(td, "output.json")
                model_path = os.path.join(td, "candidate_model.keras")
                new_config["_doin_model_save_path"] = model_path
                with open(in_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "gen": gen,
                        "cand": int(self.eval_counter),
                        "config": _json_sanitize(new_config),
                        "hyper": hyper_dict,
                    }, f)

                env = os.environ.copy()
                env.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "1")
                env.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
                env.setdefault("PYTHONUNBUFFERED", "1")

                cmd = [sys.executable, "-u", "-m",
                       "optimizer_plugins.candidate_worker",
                       "--input", in_path, "--output", out_path]

                try:
                    master_fd, slave_fd = pty.openpty()
                    p = subprocess.Popen(
                        cmd, env=env, stdin=slave_fd, stdout=slave_fd,
                        stderr=slave_fd, close_fds=True,
                    )
                    os.close(slave_fd)

                    while True:
                        r, _, _ = select.select([master_fd], [], [], 0.2)
                        if master_fd in r:
                            try:
                                data = os.read(master_fd, 4096)
                            except OSError:
                                data = b""
                            if not data:
                                break
                            try:
                                sys.stdout.buffer.write(data)
                                sys.stdout.buffer.flush()
                            except Exception:
                                pass
                        if p.poll() is not None:
                            continue

                    returncode = p.wait()
                    try:
                        os.close(master_fd)
                    except Exception:
                        pass
                except Exception as e:
                    print(f"CRITICAL: Worker spawn failed: {e}")
                    genome.fitness = float("inf")
                    return float("inf")

                if returncode != 0:
                    print(f"CRITICAL: Worker failed (returncode={returncode})")
                    genome.fitness = float("inf")
                    return float("inf")

                try:
                    with open(out_path, "r", encoding="utf-8") as f:
                        payload = json.load(f)
                except Exception:
                    genome.fitness = float("inf")
                    return float("inf")

                fitness = float(payload.get("fitness", float("inf")))
                genome.fitness = fitness
                genome.val_mae = float(payload.get("val_mae", float("inf")))
                genome.naive_mae = float(payload.get("naive_mae", float("inf")))
                genome.train_mae = float(payload.get("train_mae", float("inf")))
                genome.train_naive_mae = float(payload.get("train_naive_mae", float("inf")))
                genome.test_mae = float(payload.get("test_mae", float("inf")))
                genome.test_naive_mae = float(payload.get("test_naive_mae", float("inf")))
                genome.model_summary = payload.get("model_summary", "")

                # Check for new champion
                is_new_champion = False
                _is_new_best = (fitness > float(self.best_fitness_so_far)) if _hib else (fitness < float(self.best_fitness_so_far))
                if np.isfinite(fitness) and _is_new_best:
                    self.best_fitness_so_far = float(fitness)
                    self.best_val_mae_so_far = genome.val_mae if np.isfinite(genome.val_mae) else self.best_val_mae_so_far
                    self.best_naive_mae_so_far = genome.naive_mae if np.isfinite(genome.naive_mae) else self.best_naive_mae_so_far
                    self.best_test_mae_so_far = genome.test_mae if np.isfinite(genome.test_mae) else self.best_test_mae_so_far
                    self.best_test_naive_mae_so_far = genome.test_naive_mae if np.isfinite(genome.test_naive_mae) else self.best_test_naive_mae_so_far
                    self.best_train_mae_so_far = genome.train_mae if np.isfinite(genome.train_mae) else self.best_train_mae_so_far
                    self.best_train_naive_mae_so_far = genome.train_naive_mae if np.isfinite(genome.train_naive_mae) else self.best_train_naive_mae_so_far
                    is_new_champion = True
                    self.best_params_so_far = hyper_dict.copy()

                    # Save champion
                    pf = config.get("optimization_parameters_file", "optimization_parameters.json")
                    resolved_pf = _resolve_repo_path(pf) or pf
                    try:
                        _atomic_json_dump(resolved_pf, hyper_dict)
                        print(f"  [NEAT CHAMPION] Parameters saved to {resolved_pf}")
                    except Exception as e:
                        print(f"  [NEAT CHAMPION] Failed to save: {e}")

                    # Callback: on_new_champion
                    _cb_new_champ = _opt_callbacks.get("on_new_champion")
                    if _cb_new_champ:
                        try:
                            _model_b64 = None
                            if os.path.exists(model_path):
                                import base64
                                with open(model_path, "rb") as mf:
                                    _model_b64 = base64.b64encode(mf.read()).decode("ascii")
                            _champ_metrics = {
                                "fitness": fitness,
                                "val_mae": genome.val_mae, "val_naive_mae": genome.naive_mae,
                                "train_mae": genome.train_mae, "train_naive_mae": genome.train_naive_mae,
                                "test_mae": genome.test_mae, "test_naive_mae": genome.test_naive_mae,
                                "_model_b64": _model_b64,
                            }
                            _champ_stage = {
                                "stage": _current_stage["stage_idx"] + 1, "total_stages": len(_stage_schedule),
                                "stage_name": _current_stage["name"],
                                "n_generations_total": _total_stage_gens,
                                "generation": gen,
                                "candidate": int(self.eval_counter),
                                "total_candidates_evaluated": int(self.total_eval_counter),
                                "neat_species_count": self.neat_species_count,
                                "neat_complexity": genome.complexity,
                            }
                            _cb_new_champ(hyper_dict, fitness, _champ_metrics, gen, _champ_stage)
                        except Exception as _cb_err:
                            print(f"  [NEAT] Champion broadcast error: {_cb_err}")

                # Print result summary
                print(f"\n{'='*80}")
                print(f"[NEAT] CANDIDATE RESULT | Gen {gen}/{_total_stage_gens} | "
                      f"Candidate {self.eval_counter}/{population_size} | "
                      f"Complexity: {genome.complexity} params | Total Evals: {self.total_eval_counter}")
                print(f"Active Parameters: {', '.join(genome.active_params)}")
                print(f"{'-'*80}")
                _is_binary = config.get("target_plugin") in ("binary_target", "direction_target")
                _lbl1 = "Accuracy" if _is_binary else "MAE"
                _lbl2 = "F1" if _is_binary else "Naive"
                print(f"  TRAINING   -> {_lbl1}: {genome.train_mae:.6f} | {_lbl2}: {genome.train_naive_mae:.6f}")
                print(f"  VALIDATION -> {_lbl1}: {genome.val_mae:.6f} | {_lbl2}: {genome.naive_mae:.6f}")
                print(f"  TEST       -> {_lbl1}: {genome.test_mae:.6f} | {_lbl2}: {genome.test_naive_mae:.6f}")
                print(f"  FITNESS: {fitness:.6f}{'  *** NEW CHAMPION ***' if is_new_champion else ''}")
                print(f"  Champion fitness: {float(self.best_fitness_so_far):.6f} | "
                      f"Patience: {self.patience_counter}/{patience}")
                print(f"{'='*80}")

                # ── Append to candidate history CSV ──────────
                if _candidate_csv_path:
                    try:
                        _row = {
                            "total_eval": int(self.total_eval_counter),
                            "generation": gen,
                            "candidate_in_gen": int(self.eval_counter),
                            "stage_name": _current_stage["name"],
                            "species_id": genome.species_id or "",
                            "complexity": genome.complexity,
                            "is_champion": 1 if is_new_champion else 0,
                            "fitness": f"{fitness:.8f}",
                            "train_mcc": f"{genome.train_mae:.6f}",
                            "train_f1": f"{genome.train_naive_mae:.6f}",
                            "val_mcc": f"{genome.val_mae:.6f}",
                            "val_f1": f"{genome.naive_mae:.6f}",
                            "test_mcc": f"{genome.test_mae:.6f}",
                            "test_f1": f"{genome.test_naive_mae:.6f}",
                            "champion_fitness": f"{float(self.best_fitness_so_far):.8f}",
                        }
                        for _p in sorted(all_params):
                            _row[_p] = hyper_dict.get(_p, "")
                        with open(_candidate_csv_path, "a", newline="") as _cf:
                            w = csv.DictWriter(_cf, fieldnames=_csv_columns)
                            w.writerow(_row)
                    except Exception as _csv_err:
                        print(f"  [NEAT] CSV write error: {_csv_err}")

                # ── Callback: on_candidate_evaluated ─────────
                _cb_candidate = _opt_callbacks.get("on_candidate_evaluated")
                if _cb_candidate:
                    try:
                        _cand_info = {
                            "total_eval": int(self.total_eval_counter),
                            "generation": gen,
                            "candidate_in_gen": int(self.eval_counter),
                            "stage": _current_stage["stage_idx"] + 1,
                            "total_stages": len(_stage_schedule),
                            "stage_name": _current_stage["name"],
                            "gen_in_stage": gen - _current_stage["start_gen"],
                            "n_generations_stage": _current_stage["end_gen"] - _current_stage["start_gen"],
                            "n_generations_total": _total_stage_gens,
                            "population_size": population_size,
                            "species_id": genome.species_id or "",
                            "complexity": genome.complexity,
                            "is_champion": is_new_champion,
                            "parameters": hyper_dict.copy(),
                            "champion_parameters": self.best_params_so_far.copy() if self.best_params_so_far else {},
                            "fitness": fitness,
                            "champion_fitness": float(self.best_fitness_so_far),
                            "train_mae": genome.train_mae,
                            "train_naive_mae": genome.train_naive_mae,
                            "val_mae": genome.val_mae,
                            "val_naive_mae": genome.naive_mae,
                            "test_mae": genome.test_mae,
                            "test_naive_mae": genome.test_naive_mae,
                            "no_improve_counter": self.patience_counter,
                            "optimization_patience": patience,
                            "neat_species_count": self.neat_species_count,
                            "neat_avg_complexity": self.neat_avg_complexity,
                        }
                        _cb_candidate(_cand_info)
                    except Exception as _cb_err:
                        print(f"  [NEAT] Candidate evaluated callback error: {_cb_err}")

                return fitness

        # ── Re-broadcast existing champion from checkpoint ────
        # After a chain reset the NEAT checkpoint may already hold a champion
        # that the chain has never seen.  Fire the callback immediately so the
        # chain records it and can start generating blocks.
        if (np.isfinite(self.best_fitness_so_far)
                and self.best_params_so_far
                and self.best_fitness_so_far < float("inf")):
            _cb_resume = _opt_callbacks.get("on_new_champion")
            if _cb_resume:
                try:
                    _resume_metrics = {
                        "fitness": self.best_fitness_so_far,
                        "val_mae": self.best_val_mae_so_far or 0.0,
                        "val_naive_mae": self.best_naive_mae_so_far or 0.0,
                        "train_mae": self.best_train_mae_so_far or 0.0,
                        "train_naive_mae": self.best_train_naive_mae_so_far or 0.0,
                        "test_mae": self.best_test_mae_so_far or 0.0,
                        "test_naive_mae": self.best_test_naive_mae_so_far or 0.0,
                        "_model_b64": None,
                    }
                    _resume_stage = {
                        "stage": _current_stage["stage_idx"] + 1, "total_stages": len(_stage_schedule),
                        "stage_name": _current_stage["name"],
                        "n_generations_total": _total_stage_gens,
                        "generation": start_gen,
                        "candidate": 0,
                        "total_candidates_evaluated": int(self.total_eval_counter),
                    }
                    print(f"[NEAT] Re-broadcasting checkpoint champion "
                          f"(fitness={self.best_fitness_so_far:.6f}) to chain")
                    _cb_resume(self.best_params_so_far, self.best_fitness_so_far,
                               _resume_metrics, start_gen, _resume_stage)
                    print(f"[NEAT] Checkpoint champion broadcast complete")
                except Exception as _e:
                    print(f"[NEAT] Checkpoint champion broadcast error: {_e}")

        # ── Main generation loop ─────────────────────────────
        start_opt = time.time()

        # Evaluate initial population
        print(f"\n[NEAT] Evaluating initial population ({population_size} genomes)...")
        self.current_gen = start_gen
        self.eval_counter = 0
        no_improve_counter = int(self.patience_counter)
        _force_advance_flag = False

        for genome in population:
            if genome.fitness is None:
                eval_genome(genome, start_gen)

                # Between-candidates callback
                _cb_between = _opt_callbacks.get("on_between_candidates")
                if _cb_between:
                    try:
                        _bc_stage = {
                            "stage": _current_stage["stage_idx"] + 1, "total_stages": len(_stage_schedule),
                            "stage_name": _current_stage["name"],
                            "n_generations_total": _total_stage_gens,
                            "n_generations_stage": _current_stage["end_gen"] - _current_stage["start_gen"],
                            "gen_in_stage": start_gen - _current_stage["start_gen"],
                            "generation": start_gen,
                            "candidate_num": int(self.eval_counter),
                            "total_candidates": population_size,
                            "total_candidates_evaluated": int(self.total_eval_counter),
                            "no_improve_counter": no_improve_counter,
                            "patience": patience,
                            "fitness": genome.fitness,
                            "val_mae": genome.val_mae,
                            "train_mae": genome.train_mae,
                            "val_naive_mae": genome.naive_mae,
                            "train_naive_mae": genome.train_naive_mae,
                            "champion_fitness": float(self.best_fitness_so_far) if self.best_fitness_so_far != float("inf") else None,
                            "candidate_params": genome.hyper_dict,
                            "model_summary": genome.model_summary,
                            "neat_species_count": self.neat_species_count,
                            "neat_complexity": genome.complexity,
                        }
                        _result = _cb_between(start_gen, int(self.eval_counter), _bc_stage)
                        if isinstance(_result, dict) and _result.get("_force_stage_advance"):
                            _force_advance_flag = True
                            break
                    except Exception as _cb_err:
                        print(f"  [NEAT] Between-candidates callback error: {_cb_err}")

        # Find best genome
        best_genome = min(population, key=_best_genome_key)
        self.best_at_gen_start = float(self.best_fitness_so_far)
        no_improve_counter = int(self.patience_counter)

        # If initial population was evaluated, that counted as generation start_gen (0).
        # The reproductive loop should start from the next generation.
        _initial_pop_evaluated = (self.eval_counter > 0)
        if _initial_pop_evaluated:
            _loop_start = start_gen + 1
            end_gen = start_gen + _total_stage_gens
        else:
            _loop_start = start_gen

        # Apply per-stage patience for the initial stage
        if _staged_mode and "patience" in _current_stage:
            patience = _current_stage["patience"]

        # Generation loop
        for gen in range(_loop_start, end_gen):
            if _force_advance_flag:
                print(f"[NEAT] Force advance detected — stopping optimization")
                break

            # ── Stage transition check ───────────────────────
            _stage_just_transitioned = False
            if _staged_mode:
                for _ss in _stage_schedule:
                    if _ss["start_gen"] <= gen < _ss["end_gen"]:
                        if _ss["stage_idx"] != _current_stage["stage_idx"]:
                            _prev_stage = _current_stage
                            _current_stage = _ss
                            _stage_just_transitioned = True
                            # Extract best numeric values for seeding
                            _best_numeric = {g.param_name: g.value for g in best_genome.genes.values()} if best_genome else {}
                            _seed_values = {**_param_defaults, **_best_numeric}
                            # Rebuild population for new stage
                            population = _build_stage_population(
                                set(_current_stage["active_params"]),
                                _current_stage["frozen_params"],
                                _seed_values,
                            )
                            species_list = []
                            no_improve_counter = 0
                            self.patience_counter = 0
                            # Update patience for the new stage (per-stage override)
                            if "patience" in _current_stage:
                                patience = _current_stage["patience"]
                            print(f"\n{'#'*80}")
                            print(f"[NEAT] *** STAGE TRANSITION: "
                                  f"'{_prev_stage['name']}' → '{_current_stage['name']}' ***")
                            print(f"[NEAT] Active params ({len(_current_stage['active_params'])}): "
                                  f"{_current_stage['active_params']}")
                            print(f"[NEAT] Frozen params ({len(_current_stage['frozen_params'])}): "
                                  f"{sorted(_current_stage['frozen_params'])}")
                            print(f"[NEAT] Patience: {patience}")
                            print(f"{'#'*80}")
                        break

            gen_start_time = time.time()
            self.current_gen = gen
            self.eval_counter = 0
            print(f"\n{'='*80}")
            _stage_label = f" [{_current_stage['name']}]" if _staged_mode else ""
            _gen_in_stage = gen - _current_stage["start_gen"]
            _gens_this_stage = _current_stage["end_gen"] - _current_stage["start_gen"]
            print(f"[NEAT] Generation {gen}/{_total_stage_gens} (stage {_gen_in_stage}/{_gens_this_stage}){_stage_label}")
            print(f"{'='*80}")

            best_at_gen_start = float(self.best_fitness_so_far)
            self.best_at_gen_start = best_at_gen_start

            # ── Callback: on_generation_start (migration IN) ──
            _cb_gen_start = _opt_callbacks.get("on_generation_start")
            if _cb_gen_start:
                try:
                    _stage_info = {
                        "stage": _current_stage["stage_idx"] + 1, "total_stages": len(_stage_schedule),
                        "stage_name": _current_stage["name"],
                        "n_generations_total": _total_stage_gens,
                        "n_generations_stage": _current_stage["end_gen"] - _current_stage["start_gen"],
                        "gen_in_stage": gen - _current_stage["start_gen"],
                        "meta_mode": False,
                        "total_candidates_evaluated": int(self.total_eval_counter),
                        "population_size": population_size,
                        "neat_species_count": self.neat_species_count,
                        "neat_avg_complexity": self.neat_avg_complexity,
                    }
                    _migrant_params = _cb_gen_start(population, None, None, gen, _stage_info)
                    if isinstance(_migrant_params, dict) and _migrant_params.get("_force_stage_advance"):
                        print(f"  [NEAT] Network signalled stage advance — ending optimization")
                        break
                    if _migrant_params and isinstance(_migrant_params, dict) and not _migrant_params.get("_force_stage_advance"):
                        # Inject network champion as a NeatGenome
                        migrant_genome = NeatGenome()
                        for p, v in _migrant_params.items():
                            if p in full_bounds:
                                inn = innovation_tracker.get_innovation(p)
                                if p == "use_log1p_features":
                                    v = 1 if v else 0
                                elif p == "positional_encoding":
                                    v = 1 if v else 0
                                elif p == "use_temporal_features":
                                    v = 1 if v else 0
                                elif p == "activation" and isinstance(v, str):
                                    v = ACTIVATION_INDEX_TO_NAME.index(v) if v in ACTIVATION_INDEX_TO_NAME else 0
                                elif p in ("hod_encoding", "dow_encoding", "moy_encoding") and isinstance(v, str):
                                    v = ENCODING_INDEX_TO_NAME.index(v) if v in ENCODING_INDEX_TO_NAME else 0
                                elif p == "loss_type" and isinstance(v, str):
                                    v = LOSS_TYPE_INDEX_TO_NAME.index(v) if v in LOSS_TYPE_INDEX_TO_NAME else 0
                                migrant_genome.genes[inn] = NeatGene(inn, p, float(v))
                        if migrant_genome.genes:
                            # Dedup: reject migrants too similar to existing population
                            _min_dist = min(
                                compatibility_distance(migrant_genome, g, full_bounds)
                                for g in population
                            )
                            if _min_dist < 0.1:
                                print(f"  [NEAT MIGRATION] Skipped — near-duplicate of existing genome (dist={_min_dist:.4f})")
                            else:
                                # Replace worst individual
                                worst_idx = max(range(len(population)),
                                                key=lambda i: _best_genome_key(population[i]))
                                population[worst_idx] = migrant_genome
                                print(f"  [NEAT MIGRATION] Injected network champion ({len(migrant_genome.genes)} params, dist={_min_dist:.4f})")
                except Exception as _cb_err:
                    print(f"  [NEAT] gen_start callback error: {_cb_err}")

            # On stage transitions: population was rebuilt by _build_stage_population,
            # skip speciation+reproduction and go directly to evaluation.
            if not _stage_just_transitioned:

                # ── Speciation ───────────────────────────────────
                speciate(population, species_list, full_bounds, compat_threshold)
                adjust_fitness(species_list, _hib)

                # Update NEAT tracking stats
                self.neat_species_count = len(species_list)
                complexities = [g.complexity for g in population]
                self.neat_avg_complexity = sum(complexities) / len(complexities) if complexities else 0
                self.neat_max_complexity = max(complexities) if complexities else 0
                self.neat_min_complexity = min(complexities) if complexities else 0
                self.neat_species_details = [
                    {"id": sp.id, "size": sp.size,
                     "best_fitness": _best_of((g.fitness for g in sp.members if g.fitness is not None), default=_worst_fitness),
                     "avg_complexity": sum(g.complexity for g in sp.members) / max(sp.size, 1)}
                    for sp in species_list
                ]

                print(f"[NEAT] Species: {self.neat_species_count} | "
                      f"Complexity: avg={self.neat_avg_complexity:.1f} min={self.neat_min_complexity} max={self.neat_max_complexity}")
                for sp in species_list:
                    best_f = _best_of((g.fitness for g in sp.members if g.fitness is not None), default=_worst_fitness)
                    print(f"  Species {sp.id}: size={sp.size}, best_fitness={best_f:.6f}")

                # ── Reproduction ─────────────────────────────────
                # Adaptive mutation: boost rates when stagnating
                _adaptive_boost = 1.0
                if no_improve_counter >= patience // 2:
                    _adaptive_boost = 2.0
                    print(f"  [NEAT] Adaptive mutation boost: 2x (stagnation {no_improve_counter}/{patience})")
                _eff_mutpb = min(mutpb * _adaptive_boost, 0.8)
                _eff_add_prob = min(add_param_prob * _adaptive_boost, 0.5)
                _eff_sigma_scale = min(0.15 * _adaptive_boost, 0.4)
                _frozen = _current_stage["frozen_params"] if _staged_mode else set()

                # Calculate offspring allocation per species
                species_scores = []
                for sp in species_list:
                    finite_adj = [g.adjusted_fitness for g in sp.members
                                  if g.adjusted_fitness is not None and np.isfinite(g.adjusted_fitness)]
                    if finite_adj:
                        mean_adj = sum(finite_adj) / len(finite_adj)
                        if _hib:
                            species_scores.append(max(mean_adj, 1e-10))
                        else:
                            species_scores.append(1.0 / max(mean_adj, 1e-10))
                    else:
                        species_scores.append(1.0)
                total_score = sum(species_scores) or 1.0

                new_population = []

                for sp_idx, sp in enumerate(species_list):
                    sp.members.sort(key=_best_genome_key)

                    # Elitism: keep best individuals from each species (dedup)
                    for elite in sp.members[:neat_elitism]:
                        is_dup = False
                        for existing in new_population:
                            if compatibility_distance(elite, existing, full_bounds) < 0.1:
                                is_dup = True
                                break
                        if not is_dup:
                            new_population.append(elite.deep_copy())

                    n_offspring = max(0, int(round(
                        population_size * species_scores[sp_idx] / total_score
                    )) - neat_elitism)

                    survival_count = max(1, int(len(sp.members) * survival_rate))
                    survivors = sp.members[:survival_count]

                    for _ in range(n_offspring):
                        if len(survivors) < 2 or random.random() < 0.25:
                            parent = random.choice(survivors)
                            child = parent.deep_copy()
                        else:
                            if random.random() < interspecies_mate_rate and len(species_list) > 1:
                                other_sp = random.choice([s for s in species_list if s.id != sp.id])
                                p2 = random.choice(other_sp.members)
                            else:
                                p2 = random.choice(survivors)
                            p1 = random.choice(survivors)
                            child = neat_crossover(p1, p2)

                        # Mutations — skip structural add/remove in staged mode
                        if not _staged_mode:
                            mutate_add_param(child, all_params, full_bounds, innovation_tracker, _eff_add_prob)
                            mutate_remove_param(child, min_params, remove_param_prob)
                        mutate_values(child, full_bounds, _eff_mutpb, _eff_sigma_scale,
                                      frozen_params=_frozen)
                        clamp_genome(child, full_bounds)
                        child.fitness = None
                        new_population.append(child)

                # Ensure population size is maintained
                while len(new_population) < population_size:
                    if _staged_mode:
                        new_population.append(_create_stage_genome(
                            set(_current_stage["active_params"]),
                            _current_stage["frozen_params"],
                            _param_defaults,
                            randomize_active=True,
                        ))
                    else:
                        new_population.append(_create_genome(initial_params))
                new_population = new_population[:population_size]

                population = new_population

            # ── Evaluate unevaluated genomes ─────────────────
            _force_advance_flag = False
            for genome in population:
                if genome.fitness is None:
                    eval_genome(genome, gen)

                    # Between-candidates callback
                    _cb_between = _opt_callbacks.get("on_between_candidates")
                    if _cb_between:
                        try:
                            _bc_stage = {
                                "stage": _current_stage["stage_idx"] + 1, "total_stages": len(_stage_schedule),
                                "stage_name": _current_stage["name"],
                                "n_generations_total": _total_stage_gens,
                                "n_generations_stage": _current_stage["end_gen"] - _current_stage["start_gen"],
                                "gen_in_stage": gen - _current_stage["start_gen"],
                                "generation": gen,
                                "candidate_num": int(self.eval_counter),
                                "total_candidates": sum(1 for g in population if g.fitness is None) + int(self.eval_counter),
                                "total_candidates_evaluated": int(self.total_eval_counter),
                                "no_improve_counter": no_improve_counter,
                                "patience": patience,
                                "fitness": genome.fitness,
                                "val_mae": genome.val_mae,
                                "train_mae": genome.train_mae,
                                "val_naive_mae": genome.naive_mae,
                                "train_naive_mae": genome.train_naive_mae,
                                "champion_fitness": float(self.best_fitness_so_far) if self.best_fitness_so_far != float("inf") else None,
                                "candidate_params": genome.hyper_dict,
                                "model_summary": genome.model_summary,
                                "neat_species_count": self.neat_species_count,
                                "neat_complexity": genome.complexity,
                            }
                            _result = _cb_between(gen, int(self.eval_counter), _bc_stage)
                            if isinstance(_result, dict) and _result.get("_force_stage_advance"):
                                print(f"  [NEAT] Force stage advance between candidates")
                                _force_advance_flag = True
                                break
                        except Exception as _cb_err:
                            print(f"  [NEAT] Between-candidates callback error: {_cb_err}")

            if _force_advance_flag:
                print(f"[NEAT] Breaking generation loop for force advance")
                break

            # ── Update best genome ───────────────────────────
            gen_best = min(population, key=_best_genome_key)
            if gen_best.fitness is not None and (best_genome is None or _fitness_better(gen_best.fitness, best_genome.fitness or _worst_fitness)):
                best_genome = gen_best.deep_copy()
                best_genome.fitness = gen_best.fitness
                best_genome.hyper_dict = gen_best.hyper_dict

            # ── Patience check ───────────────────────────────
            current_best_fitness = _best_of(
                (g.fitness for g in population if g.fitness is not None),
                default=_worst_fitness,
            )
            if _fitness_better(current_best_fitness, best_at_gen_start):
                no_improve_counter = 0
                self.patience_counter = 0
                _cmp_sym = ">" if _hib else "<"
                print(f"  [NEAT PATIENCE] RESET — new best {current_best_fitness:.6f} {_cmp_sym} gen_start {best_at_gen_start:.6f}")
            else:
                no_improve_counter += 1
                self.patience_counter = no_improve_counter
                print(f"  [NEAT PATIENCE] No improvement: {no_improve_counter}/{patience} "
                      f"(best_gen={current_best_fitness:.6f}, gen_start={best_at_gen_start:.6f}, "
                      f"global_best={float(self.best_fitness_so_far):.6f})")

            # ── Statistics ───────────────────────────────────
            gen_end_time = time.time()
            gen_duration = gen_end_time - gen_start_time
            valid_fitnesses = [g.fitness for g in population if g.fitness is not None and np.isfinite(g.fitness)]
            avg_fitness = sum(valid_fitnesses) / len(valid_fitnesses) if valid_fitnesses else _worst_fitness

            stats_history.append({
                "generation": gen,
                "duration": gen_duration,
                "avg_fitness": avg_fitness,
                "best_fitness_gen": current_best_fitness,
                "champion_fitness_global": float(self.best_fitness_so_far),
                "champion_validation_mae_global": self.best_val_mae_so_far,
                "champion_validation_naive_mae_global": self.best_naive_mae_so_far,
                "species_count": self.neat_species_count,
                "avg_complexity": self.neat_avg_complexity,
            })

            # Save statistics
            stats_data = {
                "optimizer_type": "neat",
                "total_time_elapsed": gen_end_time - start_opt,
                "candidates_evaluated_so_far": int(self.total_eval_counter),
                "champion_fitness": float(self.best_fitness_so_far) if self.best_fitness_so_far is not None else None,
                "champion_validation_mae": float(self.best_val_mae_so_far) if self.best_val_mae_so_far is not None else None,
                "champion_validation_naive_mae": float(self.best_naive_mae_so_far) if self.best_naive_mae_so_far is not None else None,
                "champion_test_mae": float(self.best_test_mae_so_far) if self.best_test_mae_so_far is not None else None,
                "champion_test_naive_mae": float(self.best_test_naive_mae_so_far) if self.best_test_naive_mae_so_far is not None else None,
                "champion_train_mae": float(self.best_train_mae_so_far) if self.best_train_mae_so_far is not None else None,
                "champion_train_naive_mae": float(self.best_train_naive_mae_so_far) if self.best_train_naive_mae_so_far is not None else None,
                "neat_species_count": self.neat_species_count,
                "neat_avg_complexity": self.neat_avg_complexity,
                "neat_max_complexity": self.neat_max_complexity,
                "neat_min_complexity": self.neat_min_complexity,
                "neat_species_details": self.neat_species_details,
                "history": stats_history,
            }
            stats_file = config.get("optimization_statistics", "optimization_stats.json")
            resolved_stats_file = _resolve_repo_path(stats_file) or stats_file
            try:
                _atomic_json_dump(resolved_stats_file, stats_data)
            except Exception as e:
                print(f"  [NEAT] Failed to save statistics: {e}")

            # Save resume checkpoint
            if resume_path:
                checkpoint = {
                    "generation": gen,
                    "population": [g.to_serializable() for g in population],
                    "innovation_tracker": innovation_tracker.to_serializable(),
                    "optimizer_state": {
                        "best_fitness_so_far": float(self.best_fitness_so_far) if self.best_fitness_so_far != float("inf") else None,
                        "best_val_mae_so_far": float(self.best_val_mae_so_far) if self.best_val_mae_so_far is not None else None,
                        "best_naive_mae_so_far": float(self.best_naive_mae_so_far) if self.best_naive_mae_so_far is not None else None,
                        "best_test_mae_so_far": float(self.best_test_mae_so_far) if self.best_test_mae_so_far is not None else None,
                        "best_test_naive_mae_so_far": float(self.best_test_naive_mae_so_far) if self.best_test_naive_mae_so_far is not None else None,
                        "best_train_mae_so_far": float(self.best_train_mae_so_far) if self.best_train_mae_so_far is not None else None,
                        "best_train_naive_mae_so_far": float(self.best_train_naive_mae_so_far) if self.best_train_naive_mae_so_far is not None else None,
                        "best_params_so_far": self.best_params_so_far if self.best_params_so_far else None,
                        "total_eval_counter": int(self.total_eval_counter),
                        "patience_counter": int(self.patience_counter),
                    },
                }
                try:
                    _atomic_json_dump(resume_path, checkpoint)
                except Exception:
                    pass

            # ── Callback: on_generation_end ──────────────────
            _cb_gen_end = _opt_callbacks.get("on_generation_end")
            if _cb_gen_end:
                try:
                    _gen_end_info = {
                        "stage": _current_stage["stage_idx"] + 1, "total_stages": len(_stage_schedule),
                        "stage_name": _current_stage["name"],
                        "n_generations_total": _total_stage_gens,
                        "n_generations_stage": _current_stage["end_gen"] - _current_stage["start_gen"],
                        "gen_in_stage": gen - _current_stage["start_gen"],
                        "meta_mode": False,
                        "generation": gen,
                        "total_candidates_evaluated": int(self.total_eval_counter),
                        "population_size": population_size,
                        "no_improve_counter": no_improve_counter,
                        "patience": patience,
                        "champion_fitness": float(self.best_fitness_so_far) if self.best_fitness_so_far is not None else None,
                        "champion_val_mae": float(self.best_val_mae_so_far) if self.best_val_mae_so_far is not None else None,
                        "champion_naive_mae": float(self.best_naive_mae_so_far) if self.best_naive_mae_so_far is not None else None,
                        "champion_test_mae": float(self.best_test_mae_so_far) if self.best_test_mae_so_far is not None else None,
                        "champion_test_naive_mae": float(self.best_test_naive_mae_so_far) if self.best_test_naive_mae_so_far is not None else None,
                        "champion_train_mae": float(self.best_train_mae_so_far) if self.best_train_mae_so_far is not None else None,
                        "champion_train_naive_mae": float(self.best_train_naive_mae_so_far) if self.best_train_naive_mae_so_far is not None else None,
                        "champion_parameters": self.best_params_so_far.copy() if self.best_params_so_far else {},
                        "avg_fitness": avg_fitness,
                        "best_fitness_gen": current_best_fitness,
                        "neat_species_count": self.neat_species_count,
                        "neat_avg_complexity": self.neat_avg_complexity,
                        "neat_species_details": self.neat_species_details,
                    }
                    _cb_gen_end(population, best_genome, None, gen, _gen_end_info, stats_data)
                except Exception as _cb_err:
                    print(f"  [NEAT] Generation end callback error: {_cb_err}")

            # Early stopping / stage advancement on patience exhaustion
            if no_improve_counter >= patience:
                if _staged_mode and _current_stage["stage_idx"] < len(_stage_schedule) - 1:
                    # Not the last stage — advance to the next stage
                    _next_idx = _current_stage["stage_idx"] + 1
                    _next_stage = _stage_schedule[_next_idx]
                    print(f"\n[NEAT] Patience exhausted ({no_improve_counter}/{patience}) "
                          f"— advancing from stage '{_current_stage['name']}' to '{_next_stage['name']}'")
                    # Jump the generation counter to the next stage's start
                    # The stage transition logic at the top of the loop will handle the rest
                    self.current_gen = _next_stage["start_gen"]
                    # We need to continue the for-loop but skip to the next stage's start_gen.
                    # Since Python for-loops don't support changing the counter, we break
                    # and let the outer stage-handling mechanism re-enter.
                    # Instead, we directly do the stage transition here:
                    _prev_stage = _current_stage
                    _current_stage = _next_stage
                    _best_numeric = {g.param_name: g.value for g in best_genome.genes.values()} if best_genome else {}
                    _seed_values = {**_param_defaults, **_best_numeric}
                    population = _build_stage_population(
                        set(_current_stage["active_params"]),
                        _current_stage["frozen_params"],
                        _seed_values,
                    )
                    species_list = []
                    no_improve_counter = 0
                    self.patience_counter = 0
                    # Update patience for the new stage (per-stage override)
                    if "patience" in _current_stage:
                        patience = _current_stage["patience"]
                    print(f"\n{'#'*80}")
                    print(f"[NEAT] *** STAGE TRANSITION (patience): "
                          f"'{_prev_stage['name']}' → '{_current_stage['name']}' ***")
                    print(f"[NEAT] Active params ({len(_current_stage['active_params'])}): "
                          f"{_current_stage['active_params']}")
                    print(f"[NEAT] Frozen params ({len(_current_stage['frozen_params'])}): "
                          f"{sorted(_current_stage['frozen_params'])}")
                    print(f"[NEAT] Patience: {patience}")
                    print(f"{'#'*80}")
                else:
                    # Last stage (or non-staged mode) — truly stop
                    print(f"\n[NEAT] Early stopping triggered after {gen + 1} generations (patience={patience})")
                    break

        # ── Extract best result ──────────────────────────────
        end_opt = time.time()
        print(f"\n[NEAT] Optimization completed in {end_opt - start_opt:.2f} seconds")
        print(f"[NEAT] Total evaluations: {self.total_eval_counter}")
        print(f"[NEAT] Final species count: {self.neat_species_count}")

        if best_genome is None or best_genome.fitness is None:
            best_genome = min(population, key=lambda g: g.fitness if g.fitness is not None else float("inf"))

        best_hyper = best_genome.to_hyper_dict(param_types)
        print(f"[NEAT] Best hyperparameters ({len(best_hyper)} params): {best_hyper}")
        print(f"[NEAT] Best fitness: {self.best_fitness_so_far:.6f}")

        return best_hyper

    # ── Shared-Population Helpers ────────────────────────────
    # These methods support the distributed shared-population mode
    # where multiple nodes share one population via blockchain.

    @staticmethod
    def _parse_bounds_and_types(config):
        """Extract full_bounds, all_params, param_types from config."""
        full_bounds = config.get("hyperparameter_bounds") or config.get("param_bounds", {})
        all_params = list(full_bounds.keys())
        param_types = {}
        for key in all_params:
            low, up = full_bounds[key]
            param_types[key] = "int" if isinstance(low, int) and isinstance(up, int) else "float"
        return full_bounds, all_params, param_types

    @staticmethod
    def _build_innovation_tracker(all_params):
        """Create an InnovationTracker pre-assigned for all params."""
        tracker = InnovationTracker()
        for p in all_params:
            tracker.get_innovation(p)
        return tracker

    @staticmethod
    def _extract_param_defaults(all_params, full_bounds, config):
        """Extract default parameter values from config."""
        defaults = {}
        for p in all_params:
            if p in config:
                v = config[p]
                if p == "use_log1p_features":
                    v = 1 if v else 0
                elif p in ("positional_encoding", "use_temporal_features",
                           "add_window_stats", "add_multi_scale_returns"):
                    v = 1 if v else 0
                elif p == "activation" and isinstance(v, str):
                    v = ACTIVATION_INDEX_TO_NAME.index(v) if v in ACTIVATION_INDEX_TO_NAME else 0
                elif p in ("hod_encoding", "dow_encoding", "moy_encoding") and isinstance(v, str):
                    v = ENCODING_INDEX_TO_NAME.index(v) if v in ENCODING_INDEX_TO_NAME else 0
                elif p == "loss_type" and isinstance(v, str):
                    v = LOSS_TYPE_INDEX_TO_NAME.index(v) if v in LOSS_TYPE_INDEX_TO_NAME else 0
                try:
                    defaults[p] = float(v)
                except (TypeError, ValueError):
                    defaults[p] = float((full_bounds[p][0] + full_bounds[p][1]) / 2)
            else:
                low, high = full_bounds[p]
                defaults[p] = float((low + high) / 2)
        return defaults

    @staticmethod
    def create_shared_population(population_size, config, seed=42):
        """Create an initial population for shared-population mode.

        Returns a list of serializable genome dicts ready for blockchain storage.
        Uses a deterministic seed so all nodes create the *same* population
        from the same genesis block seed.
        """
        random.seed(seed)
        np.random.seed(seed)

        full_bounds, all_params, param_types = Plugin._parse_bounds_and_types(config)
        innovation_tracker = Plugin._build_innovation_tracker(all_params)
        _param_defaults = Plugin._extract_param_defaults(all_params, full_bounds, config)

        # Build stage schedule
        _raw_stages = config.get("optimization_stages", None)
        if _raw_stages is None:
            _raw_stages = _build_default_stages(all_params,
                                                 config.get("n_generations", 10))
        _stage_schedule = []
        _gen_cursor = 0
        for _si, _s in enumerate(_raw_stages):
            _s_params = ([p for p in _s["params"] if p in full_bounds]
                         if _s["params"] != "all" else list(all_params))
            _s_gens = _s.get("generations",
                             max(3, config.get("n_generations", 10) // len(_raw_stages)))
            _stage_entry = {
                "name": _s["name"],
                "stage_idx": _si,
                "active_params": _s_params,
                "frozen_params": list(set(all_params) - set(_s_params)),
                "start_gen": _gen_cursor,
                "end_gen": _gen_cursor + _s_gens,
            }
            # Preserve per-stage patience if defined
            if "patience" in _s:
                _stage_entry["patience"] = _s["patience"]
            _stage_schedule.append(_stage_entry)
            _gen_cursor += _s_gens
        _staged_mode = len(_stage_schedule) > 1
        _current_stage = _stage_schedule[0]

        # Create genomes
        def _mk_genome(params_list):
            g = NeatGenome()
            for p in params_list:
                inn = innovation_tracker.get_innovation(p)
                low, high = full_bounds[p]
                if isinstance(low, int) and isinstance(high, int):
                    val = random.randint(low, high)
                else:
                    val = random.uniform(low, high)
                g.genes[inn] = NeatGene(inn, p, float(val))
            return g

        def _mk_stage_genome(active_set, frozen_set, best_values, randomize_active=True):
            g = NeatGenome()
            for p in all_params:
                inn = innovation_tracker.get_innovation(p)
                low, high = full_bounds[p]
                if p in frozen_set:
                    val = best_values.get(p, _param_defaults[p])
                elif randomize_active:
                    if isinstance(low, int) and isinstance(high, int):
                        val = random.randint(low, high)
                    else:
                        val = random.uniform(low, high)
                else:
                    val = best_values.get(p, _param_defaults[p])
                    if isinstance(low, int) and isinstance(high, int):
                        span = high - low
                        if span > 1:
                            val = max(low, min(high, int(round(val + random.gauss(0, span * 0.15)))))
                    else:
                        val = max(low, min(high, val + random.gauss(0, (high - low) * 0.15)))
                g.genes[inn] = NeatGene(inn, p, float(val))
            return g

        if _staged_mode:
            n_elite = max(2, population_size // 4)
            population = []
            active_set = set(_current_stage["active_params"])
            frozen_set = set(_current_stage.get("frozen_params", []))
            for _ in range(n_elite):
                population.append(_mk_stage_genome(active_set, frozen_set,
                                                   _param_defaults, randomize_active=False))
            for _ in range(population_size - n_elite):
                population.append(_mk_stage_genome(active_set, frozen_set,
                                                   _param_defaults, randomize_active=True))
        else:
            min_params = config.get("neat_min_params", 6)
            initial_params = config.get("neat_initial_params")
            if not initial_params:
                initial_params = all_params[:min(min_params, len(all_params))]
            population = [_mk_genome(initial_params) for _ in range(population_size)]

        result = {
            "population": [g.to_serializable() for g in population],
            "innovation_tracker": innovation_tracker.to_serializable(),
            "stage_schedule": _stage_schedule,
            "param_defaults": _param_defaults,
            "config_snapshot": {
                "population_size": population_size,
                "all_params": all_params,
                "param_types": param_types,
                "full_bounds": {k: list(v) for k, v in full_bounds.items()},
            },
        }
        return result

    @staticmethod
    def reproduce_shared(evaluated_pop_serialized, generation, seed, config,
                         innovation_tracker_data, stage_schedule,
                         param_defaults, current_stage_idx=0,
                         no_improve_count=0):
        """Deterministic reproduction for shared-population mode.

        Given a fully-evaluated population (list of serializable genome dicts
        with fitness values), produce the next generation using the same
        evolutionary operators as the island-mode GA.

        All nodes calling this with the same inputs + seed will produce the
        *identical* next population — this is essential for blockchain consensus.

        Returns dict with next population + updated state.
        """
        random.seed(seed)
        np.random.seed(seed)

        full_bounds, all_params, param_types = Plugin._parse_bounds_and_types(config)
        innovation_tracker = InnovationTracker.from_serializable(innovation_tracker_data)

        # Fitness direction
        _is_binary = config.get("target_plugin") in ("binary_target", "direction_target")
        _hib = config.get("higher_is_better", _is_binary)
        _worst_fitness = float("-inf") if _hib else float("inf")
        def _best_key(g):
            if g.fitness is None or not np.isfinite(g.fitness):
                return _worst_fitness
            return -g.fitness if _hib else g.fitness

        # Restore genomes
        population = [NeatGenome.from_serializable(gd) for gd in evaluated_pop_serialized]
        population_size = len(population)

        # Config params
        patience = config.get("optimization_patience", 6)
        compat_threshold = config.get("neat_compatibility_threshold", 2.0)
        min_params = config.get("neat_min_params", 6)
        survival_rate = config.get("neat_survival_rate", 0.5)
        interspecies_mate_rate = config.get("neat_interspecies_mate_rate", 0.01)
        neat_elitism = config.get("neat_elitism", 1)
        mutpb = config.get("mutpb", 0.2)
        add_param_prob = config.get("neat_add_param_prob", 0.35)
        remove_param_prob = config.get("neat_remove_param_prob", 0.05)

        _staged_mode = len(stage_schedule) > 1
        _current_stage = stage_schedule[current_stage_idx]
        _frozen = set(_current_stage.get("frozen_params", [])) if _staged_mode else set()

        # Per-stage patience overrides global patience
        if "patience" in _current_stage:
            patience = _current_stage["patience"]

        # Find best genome
        valid_pop = [g for g in population if g.fitness is not None and np.isfinite(g.fitness)]
        if not valid_pop:
            # All failed — regenerate randomly
            result_pop = Plugin.create_shared_population(population_size, config, seed=seed + 1)
            return {
                "population": result_pop["population"],
                "generation": generation + 1,
                "best_fitness": _worst_fitness,
                "stage_idx": current_stage_idx,
                "no_improve_count": no_improve_count + 1,
                "stage_advanced": False,
                "patience": patience,
            }

        best_genome = min(valid_pop, key=_best_key)
        best_fitness = best_genome.fitness

        # Check patience / stage advancement
        stage_advanced = False
        new_stage_idx = current_stage_idx
        if no_improve_count >= patience:
            if _staged_mode and current_stage_idx < len(stage_schedule) - 1:
                new_stage_idx = current_stage_idx + 1
                stage_advanced = True
                no_improve_count = 0
                _current_stage = stage_schedule[new_stage_idx]
                _frozen = set(_current_stage.get("frozen_params", []))
                # Update patience for the new stage (per-stage override)
                if "patience" in _current_stage:
                    patience = _current_stage["patience"]
                print(f"[SHARED NEAT] Stage advance: {stage_schedule[current_stage_idx]['name']} "
                      f"→ {_current_stage['name']} (patience exhausted, new patience={patience})")
            else:
                # Final stage patience exhausted — signal convergence
                return {
                    "population": [g.to_serializable() for g in population],
                    "generation": generation + 1,
                    "best_fitness": best_fitness,
                    "stage_idx": current_stage_idx,
                    "no_improve_count": no_improve_count,
                    "converged": True,
                    "stage_advanced": False,
                    "patience": patience,
                }

        # If stage advanced, rebuild population for new stage
        if stage_advanced:
            _best_numeric = {g.param_name: g.value for g in best_genome.genes.values()}
            _seed_values = {**param_defaults, **_best_numeric}
            active_set = set(_current_stage["active_params"])
            frozen_set = set(_current_stage.get("frozen_params", []))

            new_population = []
            n_elite = max(2, population_size // 4)
            for i in range(population_size):
                g = NeatGenome()
                for p in all_params:
                    inn = innovation_tracker.get_innovation(p)
                    low, high = full_bounds[p]
                    if p in frozen_set:
                        val = _seed_values.get(p, param_defaults.get(p, (low + high) / 2))
                    elif i < n_elite:
                        val = _seed_values.get(p, param_defaults.get(p, (low + high) / 2))
                        if isinstance(low, int) and isinstance(high, int):
                            span = high - low
                            if span > 1:
                                val = max(low, min(high, int(round(val + random.gauss(0, span * 0.15)))))
                        else:
                            val = max(low, min(high, val + random.gauss(0, (high - low) * 0.15)))
                    else:
                        if isinstance(low, int) and isinstance(high, int):
                            val = random.randint(low, high)
                        else:
                            val = random.uniform(low, high)
                    g.genes[inn] = NeatGene(inn, p, float(val))
                new_population.append(g)

            return {
                "population": [g.to_serializable() for g in new_population],
                "generation": generation + 1,
                "best_fitness": best_fitness,
                "stage_idx": new_stage_idx,
                "no_improve_count": 0,
                "stage_advanced": True,
                "patience": patience,
            }

        # Normal reproduction (no stage change)
        species_list = []
        speciate(population, species_list, full_bounds, compat_threshold)
        adjust_fitness(species_list, _hib)

        # Adaptive mutation boost on stagnation
        _adaptive_boost = 2.0 if no_improve_count >= patience // 2 else 1.0
        _eff_mutpb = min(mutpb * _adaptive_boost, 0.8)
        _eff_add_prob = min(add_param_prob * _adaptive_boost, 0.5)
        _eff_sigma_scale = min(0.15 * _adaptive_boost, 0.4)

        # Offspring allocation per species
        species_scores = []
        for sp in species_list:
            finite_adj = [g.adjusted_fitness for g in sp.members
                          if g.adjusted_fitness is not None and np.isfinite(g.adjusted_fitness)]
            if finite_adj:
                mean_adj = sum(finite_adj) / len(finite_adj)
                if _hib:
                    species_scores.append(max(mean_adj, 1e-10))
                else:
                    species_scores.append(1.0 / max(mean_adj, 1e-10))
            else:
                species_scores.append(1.0)
        total_score = sum(species_scores) or 1.0

        new_population = []
        for sp_idx, sp in enumerate(species_list):
            sp.members.sort(key=_best_key)

            # Elitism
            for elite in sp.members[:neat_elitism]:
                is_dup = False
                for existing in new_population:
                    if compatibility_distance(elite, existing, full_bounds) < 0.1:
                        is_dup = True
                        break
                if not is_dup:
                    e = elite.deep_copy()
                    e.fitness = None  # Reset for re-evaluation
                    new_population.append(e)

            n_offspring = max(0, int(round(
                population_size * species_scores[sp_idx] / total_score
            )) - neat_elitism)

            survival_count = max(1, int(len(sp.members) * survival_rate))
            survivors = sp.members[:survival_count]

            for _ in range(n_offspring):
                if len(survivors) < 2 or random.random() < 0.25:
                    parent = random.choice(survivors)
                    child = parent.deep_copy()
                else:
                    if random.random() < interspecies_mate_rate and len(species_list) > 1:
                        other_sp = random.choice([s for s in species_list if s.id != sp.id])
                        p2 = random.choice(other_sp.members)
                    else:
                        p2 = random.choice(survivors)
                    p1 = random.choice(survivors)
                    child = neat_crossover(p1, p2)

                if not _staged_mode:
                    mutate_add_param(child, all_params, full_bounds, innovation_tracker, _eff_add_prob)
                    mutate_remove_param(child, min_params, remove_param_prob)
                mutate_values(child, full_bounds, _eff_mutpb, _eff_sigma_scale,
                              frozen_params=_frozen)
                clamp_genome(child, full_bounds)
                child.fitness = None
                new_population.append(child)

        # Ensure population size
        while len(new_population) < population_size:
            if _staged_mode:
                g = NeatGenome()
                active_set = set(_current_stage["active_params"])
                frozen_set_local = set(_current_stage.get("frozen_params", []))
                for p in all_params:
                    inn = innovation_tracker.get_innovation(p)
                    low, high = full_bounds[p]
                    if p in frozen_set_local:
                        val = param_defaults.get(p, (low + high) / 2)
                    else:
                        val = random.randint(low, high) if isinstance(low, int) and isinstance(high, int) else random.uniform(low, high)
                    g.genes[inn] = NeatGene(inn, p, float(val))
                new_population.append(g)
            else:
                init_params = config.get("neat_initial_params")
                if not init_params:
                    init_params = all_params[:min(min_params, len(all_params))]
                g = NeatGenome()
                for p in init_params:
                    inn = innovation_tracker.get_innovation(p)
                    low, high = full_bounds[p]
                    val = random.randint(low, high) if isinstance(low, int) and isinstance(high, int) else random.uniform(low, high)
                    g.genes[inn] = NeatGene(inn, p, float(val))
                new_population.append(g)
        new_population = new_population[:population_size]

        return {
            "population": [g.to_serializable() for g in new_population],
            "innovation_tracker": innovation_tracker.to_serializable(),
            "generation": generation + 1,
            "best_fitness": best_fitness,
            "stage_idx": current_stage_idx,
            "no_improve_count": no_improve_count,
            "stage_advanced": False,
            "patience": patience,
        }

    @staticmethod
    def evaluate_single_genome(genome_serialized, gen, config):
        """Evaluate a single genome using the subprocess candidate_worker.

        This is the shared-population equivalent of the inner eval_genome()
        function.  Returns dict with fitness and all metrics.
        """
        # Determine worst fitness based on metric_type
        _metric_type = config.get("metric_type", "regression")
        _worst = float("-inf") if _metric_type == "binary" else float("inf")

        full_bounds, all_params, param_types = Plugin._parse_bounds_and_types(config)
        innovation_tracker = Plugin._build_innovation_tracker(all_params)
        genome = NeatGenome.from_serializable(genome_serialized)
        hyper_dict = genome.to_hyper_dict(param_types)

        eval_config = config.copy()
        eval_config.update(hyper_dict)
        # Remove non-serializable keys
        eval_config.pop("optimization_callbacks", None)
        eval_config.pop("_non_serializable_keys", None)

        for k in ("memory_log_file", "optimizer_resource_log_file", "batch_memory_log_file"):
            if eval_config.get(k):
                eval_config[k] = _resolve_repo_path(eval_config.get(k))
        eval_config.setdefault("memory_log_tag", f"shared_gen{gen}")

        with tempfile.TemporaryDirectory(prefix="shared_cand_") as td:
            in_path = os.path.join(td, "input.json")
            out_path = os.path.join(td, "output.json")
            model_path = os.path.join(td, "candidate_model.keras")
            eval_config["_doin_model_save_path"] = model_path
            with open(in_path, "w", encoding="utf-8") as f:
                json.dump({
                    "gen": gen,
                    "cand": 0,
                    "config": _json_sanitize(eval_config),
                    "hyper": hyper_dict,
                }, f)

            env = os.environ.copy()
            env.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "1")
            env.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
            env.setdefault("PYTHONUNBUFFERED", "1")

            cmd = [sys.executable, "-u", "-m",
                   "optimizer_plugins.candidate_worker",
                   "--input", in_path, "--output", out_path]

            try:
                master_fd, slave_fd = pty.openpty()
                p = subprocess.Popen(
                    cmd, env=env, stdin=slave_fd, stdout=slave_fd,
                    stderr=slave_fd, close_fds=True,
                )
                os.close(slave_fd)

                while True:
                    r, _, _ = select.select([master_fd], [], [], 0.2)
                    if master_fd in r:
                        try:
                            data = os.read(master_fd, 4096)
                        except OSError:
                            data = b""
                        if not data:
                            break
                        try:
                            sys.stdout.buffer.write(data)
                            sys.stdout.buffer.flush()
                        except Exception:
                            pass
                    if p.poll() is not None:
                        break

                returncode = p.wait()
                try:
                    os.close(master_fd)
                except Exception:
                    pass
            except Exception as e:
                print(f"[SHARED NEAT] Worker spawn failed: {e}")
                return {"fitness": _worst, "error": str(e)}

            if returncode != 0:
                print(f"[SHARED NEAT] Worker failed (rc={returncode})")
                return {"fitness": _worst, "error": f"worker_rc_{returncode}"}

            try:
                with open(out_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception:
                return {"fitness": _worst, "error": "output_parse_error"}

            fitness = float(payload.get("fitness", _worst))
            model_b64 = None
            if os.path.exists(model_path):
                import base64
                with open(model_path, "rb") as mf:
                    model_b64 = base64.b64encode(mf.read()).decode("ascii")

            return {
                "fitness": fitness,
                "val_mae": float(payload.get("val_mae", float("inf"))),
                "val_naive_mae": float(payload.get("naive_mae", float("inf"))),
                "train_mae": float(payload.get("train_mae", float("inf"))),
                "train_naive_mae": float(payload.get("train_naive_mae", float("inf"))),
                "test_mae": float(payload.get("test_mae", float("inf"))),
                "test_naive_mae": float(payload.get("test_naive_mae", float("inf"))),
                "hyper_dict": hyper_dict,
                "_model_b64": model_b64,
            }
