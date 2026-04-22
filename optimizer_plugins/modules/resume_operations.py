"""Resume and checkpoint operations for optimization."""
import json
import os


def load_resume_checkpoint(resume_path, population, hyper_keys, full_bounds, incremental_enabled, config=None):
    """
    Load population state from resume checkpoint with stage awareness.
    
    Args:
        resume_path: Path to resume JSON file
        population: DEAP population to load into
        hyper_keys: Current list of active parameter names from config
        full_bounds: Dict of all parameter bounds
        incremental_enabled: Whether incremental optimization is active
        config: Configuration dict (for detecting new parameters)
    
    Returns:
        (start_gen: int, loaded_count: int, loaded_indices: set, actual_genome_size: int,
         resumed_stage: int, resumed_params: list, optimizer_state: dict)
    """
    loaded_indices = set()
    start_gen = 0
    loaded_count = 0
    actual_genome_size = len(hyper_keys)
    resumed_stage = 1
    resumed_params = hyper_keys.copy()
    optimizer_state = {}
    
    try:
        print(f"\n[RESUME] Found resume file at: {resume_path}")
        print(f"[RESUME] Attempting to load population state...")
        
        with open(resume_path, 'r') as f:
            resume_data = json.load(f)
        
        print(f"[RESUME] Successfully loaded JSON from resume file")
        
        saved_pop = resume_data.get("population", [])
        start_gen = resume_data.get("generation", -1) + 1
        saved_active_params = resume_data.get("active_parameters", [])
        resumed_stage = resume_data.get("current_stage", 1)
        saved_meta_mode = resume_data.get("meta_mode", False)
        
        print(f"[RESUME] Resume data: generation={start_gen-1}, population_size={len(saved_pop)}, stage={resumed_stage}, meta_mode={saved_meta_mode}")
        
        saved_fitnesses = resume_data.get("fitnesses", [])
        saved_ind_metrics = resume_data.get("individual_metrics", [])
        optimizer_state = resume_data.get("optimizer_state", {})

        if not saved_pop:
            print(f"[RESUME] WARN: Resume file found but contained no population data.")
            return start_gen, 0, loaded_indices, actual_genome_size, resumed_stage, resumed_params, optimizer_state
        
        # Detect genome size from saved population
        saved_genome_size = len(saved_pop[0]) if saved_pop else 0
        actual_genome_size = saved_genome_size
        
        print(f"[RESUME] Saved genome size: {saved_genome_size}, Current config expects: {len(hyper_keys)}")
        print(f"[RESUME] Saved stage: {resumed_stage}, Meta-mode: {saved_meta_mode}")
        print(f"[RESUME] Saved active parameters: {saved_active_params}")
        
        # Detect NEW parameters in config that weren't in resume
        if saved_active_params:
            new_params_in_config = [p for p in hyper_keys if p not in saved_active_params]
            if new_params_in_config:
                print(f"[RESUME] Config has {len(new_params_in_config)} NEW parameters not in resume:")
                print(f"[RESUME]   New parameters: {new_params_in_config}")
                if incremental_enabled:
                    print(f"[RESUME] Will add new parameters incrementally starting from stage {resumed_stage}")
                    # Use saved parameters initially, will add new ones per-stage
                    resumed_params = saved_active_params
                    actual_genome_size = len(saved_active_params)
        
        # Load individuals
        invalid_count = 0
        expanded_count = 0
        
        for i in range(min(len(population), len(saved_pop))):
            saved_ind = saved_pop[i]
            
            # Handle genome size mismatch
            if len(saved_ind) < len(hyper_keys):
                # Config has MORE parameters than saved
                new_param_count = len(hyper_keys) - len(saved_ind)
                print(f"[RESUME] Individual {i}: Config has {new_param_count} new parameters")
                
                if incremental_enabled:
                    # Don't expand - will use incremental addition
                    print(f"[RESUME] Incremental mode: Will add new parameters progressively")
                else:
                    # Expand immediately at midpoint
                    print(f"[RESUME] Non-incremental mode: Adding new parameters at midpoint")
                    new_params = hyper_keys[len(saved_ind):]
                    for param_name in new_params:
                        low, up = full_bounds[param_name]
                        midpoint = (low + up) / 2.0
                        if isinstance(low, int) and isinstance(up, int):
                            midpoint = int(round(midpoint))
                        saved_ind.append(midpoint)
                    expanded_count += 1
            
            elif len(saved_ind) > len(hyper_keys):
                # Config has FEWER parameters - truncate
                print(f"[RESUME] WARN: Individual {i} truncated from {len(saved_ind)} to {len(hyper_keys)} genes")
                saved_ind = saved_ind[:len(hyper_keys)]
            
            # Validate bounds (only for genes we're loading)
            valid = True
            genes_to_load = min(len(saved_ind), len(hyper_keys))
            
            for k in range(genes_to_load):
                val = saved_ind[k]
                param_name = hyper_keys[k] if k < len(hyper_keys) else f"param_{k}"
                if param_name in full_bounds:
                    low, up = full_bounds[param_name]
                    if not (low <= val <= up):
                        print(f"[RESUME] WARN: Individual {i} parameter '{param_name}' value {val} out of bounds [{low}, {up}]")
                        valid = False
                        invalid_count += 1
                        break
            
            # Load into population
            if valid:
                for k in range(genes_to_load):
                    population[i][k] = saved_ind[k]
                # Restore fitness if available
                if i < len(saved_fitnesses) and saved_fitnesses[i] is not None:
                    population[i].fitness.values = tuple(saved_fitnesses[i])
                # Restore per-individual metrics
                if i < len(saved_ind_metrics) and saved_ind_metrics[i]:
                    m = saved_ind_metrics[i]
                    for attr_name in ("val_mae", "naive_mae", "train_mae", "train_naive_mae", "test_mae", "test_naive_mae"):
                        if m.get(attr_name) is not None:
                            setattr(population[i], attr_name, m[attr_name])
                loaded_count += 1
                loaded_indices.add(i)
        
        fitness_restored = sum(1 for i in range(min(len(population), len(saved_fitnesses)))
                               if i in loaded_indices and saved_fitnesses[i] is not None
                               and population[i].fitness.valid)
        print(f"[RESUME] SUCCESS: Loaded {loaded_count} individuals ({fitness_restored} with restored fitness)")
        if expanded_count > 0:
            print(f"[RESUME] Expanded {expanded_count} individuals with new parameters")
        if invalid_count > 0:
            print(f"[RESUME] Skipped {invalid_count} invalid individuals")
        if optimizer_state:
            print(f"[RESUME] Optimizer state restored: best_fitness={optimizer_state.get('best_fitness_so_far')}, "
                  f"total_evals={optimizer_state.get('total_eval_counter')}, patience={optimizer_state.get('patience_counter')}")
        print(f"[RESUME] Resuming from generation {start_gen}")
        
    except Exception as e:
        print(f"[RESUME] ERROR: Failed to load resume file: {e}")
        import traceback
        traceback.print_exc()
        print(f"[RESUME] Continuing with fresh population")
    
    return start_gen, loaded_count, loaded_indices, actual_genome_size, resumed_stage, resumed_params, optimizer_state


def load_champion_parameters(params_path, hyper_keys, full_bounds):
    """
    Load champion parameters from file.
    
    Args:
        params_path: Path to champion parameters JSON
        hyper_keys: List of parameter names to load
        full_bounds: Dict of parameter bounds
    
    Returns:
        List of parameter values or None if failed
    """
    try:
        print(f"\n[CHAMPION LOAD] Found champion parameters file at: {params_path}")
        
        with open(params_path, 'r') as f:
            champ_dict = json.load(f)
        
        champ_ind = []
        for key in hyper_keys:
            if key not in champ_dict:
                print(f"[CHAMPION LOAD] WARN: Champion missing key '{key}'")
                return None
            
            val = champ_dict[key]
            
            # Handle special encodings
            if key == "use_log1p_features":
                val = 1 if val == ["typical_price"] else 0
            elif key == "positional_encoding":
                val = 1 if val else 0
            
            champ_ind.append(val)
        
        print(f"[CHAMPION LOAD] SUCCESS: Loaded champion with {len(champ_ind)} parameters")
        return champ_ind
        
    except Exception as e:
        print(f"[CHAMPION LOAD] ERROR: Failed to load champion: {e}")
        return None


def save_resume_checkpoint(resume_path, generation, population, current_stage=None, active_parameters=None, meta_mode=False, optimizer_state=None):
    """
    Save current population state to resume file with stage tracking.
    
    Args:
        resume_path: Path to save resume JSON
        generation: Current generation number
        population: DEAP population
        current_stage: Current stage number (for incremental/meta mode)
        active_parameters: List of currently active parameter names
        meta_mode: Whether in meta-optimization mode
        optimizer_state: Dict with best_fitness_so_far, MAE metrics, counters, etc.
    """
    try:
        # Save individual fitness values and per-individual metrics
        pop_fitnesses = []
        pop_metrics = []
        for ind in population:
            if ind.fitness.valid:
                pop_fitnesses.append(list(ind.fitness.values))
            else:
                pop_fitnesses.append(None)
            pop_metrics.append({
                "val_mae": getattr(ind, "val_mae", None),
                "naive_mae": getattr(ind, "naive_mae", None),
                "train_mae": getattr(ind, "train_mae", None),
                "train_naive_mae": getattr(ind, "train_naive_mae", None),
                "test_mae": getattr(ind, "test_mae", None),
                "test_naive_mae": getattr(ind, "test_naive_mae", None),
            })

        resume_payload = {
            "generation": generation,
            "population": [list(ind) for ind in population],
            "fitnesses": pop_fitnesses,
            "individual_metrics": pop_metrics,
            "genome_size": len(population[0]) if population else 0,
            "active_parameters": active_parameters if active_parameters else [],
            "current_stage": current_stage if current_stage is not None else 1,
            "meta_mode": meta_mode,
        }
        if optimizer_state:
            resume_payload["optimizer_state"] = optimizer_state
        
        # Atomic save
        temp_path = resume_path + ".tmp"
        with open(temp_path, 'w') as f:
            json.dump(resume_payload, f, indent=2)
        os.replace(temp_path, resume_path)
        
        print(f"  Resume state saved to {resume_path}")
        if meta_mode and current_stage:
            print(f"  Meta-mode stage {current_stage} with {len(active_parameters)} active parameters")
    except Exception as e:
        print(f"  Failed to save resume state: {e}")
