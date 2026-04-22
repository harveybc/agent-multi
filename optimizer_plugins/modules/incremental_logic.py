"""Incremental optimization logic - progressive parameter addition."""
from .meta_stages import (
    get_active_parameters_for_stage,
    get_new_parameters_in_stage,
    get_stage_info,
    get_total_stages
)


def setup_incremental_optimization(full_bounds, config):
    """
    Setup incremental optimization parameters.
    
    Supports two modes:
    1. Standard incremental: Adds parameters in fixed-size chunks
    2. Meta-optimization: Follows 8-stage predefined parameter groups
    
    Args:
        full_bounds: Dict of all hyperparameter bounds
        config: Configuration dict
    
    Returns:
        (enabled: bool, increment_size: int, all_params: list, initial_params: list, 
         total_stages: int, meta_mode: bool)
    """
    enabled = config.get("optimization_incremental", False)
    meta_mode = config.get("optimization_meta_mode", False)
    
    if meta_mode:
        # META-OPTIMIZATION MODE: Use config-defined stages
        print(f"\n[META-OPTIMIZATION] Hierarchical staged parameter deployment enabled")
        all_params = list(full_bounds.keys())
        
        # Stage 1: Get initial parameters from config
        initial_params = get_active_parameters_for_stage(1, config)
        total_stages = get_total_stages(config)
        increment_size = 0  # Not used in meta mode
        
        stage_name, stage_desc = get_stage_info(1, config)
        print(f"[META-OPTIMIZATION] Stage 1/{total_stages}: {stage_name}")
        print(f"[META-OPTIMIZATION] Description: {stage_desc}")
        print(f"[META-OPTIMIZATION] Starting parameters: {initial_params}")
        
        return enabled, increment_size, all_params, initial_params, total_stages, meta_mode
    
    elif enabled:
        # STANDARD INCREMENTAL MODE
        increment_size = config.get("optimization_increment_size", 2)
        all_params = list(full_bounds.keys())
        
        initial_count = min(increment_size, len(all_params))
        initial_params = all_params[:initial_count]
        total_stages = (len(all_params) + increment_size - 1) // increment_size
        
        print(f"\n[INCREMENTAL] Incremental optimization enabled:")
        print(f"[INCREMENTAL] Total parameters: {len(all_params)}")
        print(f"[INCREMENTAL] Increment size: {increment_size} parameters per stage")
        print(f"[INCREMENTAL] Total stages: {total_stages}")
        print(f"[INCREMENTAL] All parameters: {all_params}")
        print(f"[INCREMENTAL] Starting with: {initial_params}")
        
        return enabled, increment_size, all_params, initial_params, total_stages, False
    
    else:
        # NO INCREMENTAL MODE
        all_params = list(full_bounds.keys())
        return False, 0, all_params, all_params, 1, False


def should_add_more_parameters(current_params, all_params, incremental_enabled):
    """
    Check if there are more parameters to add.
    
    Args:
        current_params: List of currently active parameters
        all_params: List of all parameters
        incremental_enabled: Whether incremental is active
    
    Returns:
        bool: True if more parameters should be added
    """
    if not incremental_enabled:
        return False
    return len(current_params) < len(all_params)


def get_next_parameter_batch(current_params, all_params, increment_size, current_stage=None, meta_mode=False, config=None):
    """
    Get the next batch of parameters to add.
    
    Args:
        current_params: List of currently active parameters
        all_params: List of all parameters
        increment_size: Number of parameters to add per stage (standard mode)
        current_stage: Current stage number (meta mode)
        meta_mode: Whether using meta-optimization mode
        config: Configuration dict (required for meta mode)
    
    Returns:
        (new_params: list, updated_params: list): New params to add and full updated list
    """
    if meta_mode and current_stage is not None:
        # META MODE: Get parameters for next stage
        if config is None:
            raise ValueError("get_next_parameter_batch in meta_mode requires config parameter")
        
        next_stage = current_stage + 1
        new_params = get_new_parameters_in_stage(next_stage, config)
        updated_params = get_active_parameters_for_stage(next_stage, config)
        
        stage_name, stage_desc = get_stage_info(next_stage, config)
        print(f"[META-OPTIMIZATION] Advancing to Stage {next_stage}/{get_total_stages(config)}: {stage_name}")
        print(f"[META-OPTIMIZATION] Description: {stage_desc}")
        print(f"[META-OPTIMIZATION] Adding parameters: {new_params}")
        
        return new_params, updated_params
    else:
        # STANDARD MODE: Add fixed-size chunks
        current_count = len(current_params)
        next_count = min(current_count + increment_size, len(all_params))
        
        new_params = all_params[current_count:next_count]
        updated_params = all_params[:next_count]
        
        return new_params, updated_params


def adjust_params_for_resume(hyper_keys, all_params, actual_genome_size, incremental_enabled):
    """
    Adjust active parameters based on resumed genome size.
    
    When resuming with incremental optimization, we want to continue from where we left off.
    If the loaded genome has fewer genes than config expects, we should start with that size
    and incrementally add the rest.
    
    Args:
        hyper_keys: Current list of expected parameters from config
        all_params: All parameters from full_bounds
        actual_genome_size: Actual size of loaded genome from resume file
        incremental_enabled: Whether incremental optimization is active
    
    Returns:
        Adjusted list of parameters matching loaded genome size (if incremental)
    """
    if not incremental_enabled:
        return hyper_keys
    
    if actual_genome_size < len(hyper_keys):
        # Resume has fewer parameters - start from that point
        adjusted = all_params[:actual_genome_size]
        print(f"[INCREMENTAL] Adjusted active parameters to match resumed genome: {len(hyper_keys)} → {actual_genome_size}")
        print(f"[INCREMENTAL] Will incrementally add remaining {len(hyper_keys) - actual_genome_size} parameters")
        return adjusted
    
    return hyper_keys
