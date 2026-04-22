"""Genome expansion and manipulation operations."""
import random

# Activation string to integer mapping (must match optimizer's ACTIVATION_INDEX_TO_NAME)
ACTIVATION_NAME_TO_INDEX = {
    "relu": 0,
    "elu": 1,
    "selu": 2,
    "tanh": 3,
    "sigmoid": 4,
    "swish": 5,
    "gelu": 6,
    "leaky_relu": 7
}


def expand_genome_with_new_params(individual, new_params, full_bounds, config):
    """
    Expand an individual's genome by adding new parameters.
    
    Args:
        individual: DEAP individual to expand
        new_params: List of parameter names to add
        full_bounds: Dict of all parameter bounds
        config: Configuration dict (to get values if available)
    
    Returns:
        Expanded individual
    """
    for param_name in new_params:
        low, up = full_bounds[param_name]
        
        # Try to use config value if available
        if param_name in config:
            val = config[param_name]
            # Handle special categorical/boolean encodings
            if param_name == "use_log1p_features":
                val = 1 if val == ["typical_price"] else 0
            elif param_name == "positional_encoding":
                val = 1 if val else 0
            elif param_name == "activation":
                # Convert string activation back to integer for GA
                if isinstance(val, str):
                    val = ACTIVATION_NAME_TO_INDEX.get(val, 0)
            individual.append(val)
        else:
            # Initialize at midpoint of bounds (safe neutral value)
            midpoint = (low + up) / 2.0
            if isinstance(low, int) and isinstance(up, int):
                midpoint = int(round(midpoint))
            individual.append(midpoint)
    
    # Reset fitness for re-evaluation
    if hasattr(individual, 'fitness') and hasattr(individual.fitness, 'values'):
        del individual.fitness.values
    
    return individual


def expand_population_with_new_params(population, new_params, full_bounds, config):
    """
    Expand entire population by adding new parameters to all individuals.
    
    Args:
        population: List of DEAP individuals
        new_params: List of parameter names to add
        full_bounds: Dict of all parameter bounds
        config: Configuration dict
    
    Returns:
        Number of individuals expanded
    """
    count = 0
    for ind in population:
        expand_genome_with_new_params(ind, new_params, full_bounds, config)
        count += 1
    return count


def validate_genome_bounds(individual, hyper_keys, full_bounds):
    """
    Validate that all genes in individual are within bounds.
    
    Args:
        individual: DEAP individual to validate
        hyper_keys: List of parameter names
        full_bounds: Dict of parameter bounds
    
    Returns:
        (valid: bool, error_message: str or None)
    """
    for k, param_name in enumerate(hyper_keys):
        if k >= len(individual):
            return False, f"Individual has {len(individual)} genes but needs {len(hyper_keys)}"
        
        val = individual[k]
        low, up = full_bounds[param_name]
        
        if not (low <= val <= up):
            return False, f"Parameter '{param_name}' value {val} out of bounds [{low}, {up}]"
    
    return True, None
