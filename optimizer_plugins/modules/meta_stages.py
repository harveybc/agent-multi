"""Meta-optimization stage definitions - configurable from JSON config."""


def get_active_parameters_for_stage(stage_number, config):
    """
    Get cumulative list of all parameters active up to and including specified stage.
    
    Args:
        stage_number: Stage number (1-based)
        config: Configuration dict with 'optimization_stages' key
    
    Returns:
        List of parameter names (cumulative from stage 1 to stage_number)
    """
    stages = config.get("optimization_stages", [])
    if not stages:
        raise ValueError("No optimization_stages defined in config")
    
    if stage_number < 1 or stage_number > len(stages):
        raise ValueError(f"Stage {stage_number} out of range (1-{len(stages)})")
    
    # Cumulative parameters from all stages up to stage_number
    active_params = []
    for stage in stages[:stage_number]:
        active_params.extend(stage.get("parameters", []))
    
    return active_params


def get_new_parameters_in_stage(stage_number, config):
    """
    Get only the NEW parameters introduced in specified stage.
    
    Args:
        stage_number: Stage number (1-based)
        config: Configuration dict with 'optimization_stages' key
    
    Returns:
        List of parameter names introduced in this stage
    """
    stages = config.get("optimization_stages", [])
    if not stages:
        raise ValueError("No optimization_stages defined in config")
    
    if stage_number < 1 or stage_number > len(stages):
        raise ValueError(f"Stage {stage_number} out of range (1-{len(stages)})")
    
    stage = stages[stage_number - 1]  # Convert to 0-based index
    return stage.get("parameters", [])


def get_stage_info(stage_number, config):
    """
    Get name and description for a specific stage.
    
    Args:
        stage_number: Stage number (1-based)
        config: Configuration dict with 'optimization_stages' key
    
    Returns:
        (name: str, description: str)
    """
    stages = config.get("optimization_stages", [])
    if not stages:
        return ("Unknown", "No stages defined")
    
    if stage_number < 1 or stage_number > len(stages):
        return ("Invalid", f"Stage {stage_number} out of range")
    
    stage = stages[stage_number - 1]
    return (stage.get("name", "Unnamed"), stage.get("description", "No description"))


def get_total_stages(config):
    """
    Get total number of optimization stages.
    
    Args:
        config: Configuration dict with 'optimization_stages' key
    
    Returns:
        Number of stages defined
    """
    stages = config.get("optimization_stages", [])
    return len(stages)


def get_all_meta_parameters(config):
    """
    Get all parameters across all stages (flattened list).
    
    Args:
        config: Configuration dict with 'optimization_stages' key
    
    Returns:
        List of all parameter names
    """
    stages = config.get("optimization_stages", [])
    all_params = []
    for stage in stages:
        all_params.extend(stage.get("parameters", []))
    return all_params


def validate_stage_configuration(config):
    """
    Validate that optimization_stages configuration is well-formed.
    
    Args:
        config: Configuration dict
    
    Returns:
        (valid: bool, errors: list)
    """
    errors = []
    stages = config.get("optimization_stages", [])
    
    if not stages:
        errors.append("No optimization_stages defined in config")
        return (False, errors)
    
    # Check stage numbering
    for i, stage in enumerate(stages, 1):
        if stage.get("stage") != i:
            errors.append(f"Stage numbering mismatch: expected {i}, got {stage.get('stage')}")
        
        if not stage.get("parameters"):
            errors.append(f"Stage {i} has no parameters defined")
        
        # Check for duplicate parameters
        params = stage.get("parameters", [])
        if len(params) != len(set(params)):
            errors.append(f"Stage {i} has duplicate parameters")
    
    # Check bounds exist for all parameters
    bounds = config.get("hyperparameter_bounds", {})
    all_params = get_all_meta_parameters(config)
    
    for param in all_params:
        if param not in bounds:
            errors.append(f"Parameter '{param}' missing from hyperparameter_bounds")
    
    return (len(errors) == 0, errors)
