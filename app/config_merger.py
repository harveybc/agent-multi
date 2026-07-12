import sys
from app.config import DEFAULT_VALUES

def process_unknown_args(unknown_args):
    """
    Parses a list of unknown strings into a dictionary.
    Assumes even-length list of [--key, value, --key, value].
    If a key is a flag without a value (though argparse unknown_args
    usually contains the next token), this might need adjustment.
    """
    if not unknown_args:
        return {}
    
    res = {}
    i = 0
    while i < len(unknown_args):
        arg = unknown_args[i]
        if arg.startswith("--"):
            key = arg.lstrip("-")
            if i + 1 < len(unknown_args) and not unknown_args[i+1].startswith("--"):
                res[key] = convert_type(unknown_args[i+1])
                i += 2
            else:
                # Key without value (boolean flag)
                res[key] = True
                i += 1
        else:
            # Not starting with --, skip or handle as positional
            i += 1
    return res

def convert_type(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
        if normalized in {"none", "null"}:
            return None
    try:
        if "." in value:
            return float(value)
        return int(value)
    except (ValueError, TypeError):
        return value

def merge_config(defaults, plugin_params1, plugin_params2, file_config, cli_args, unknown_args):
    # Base defaults
    merged_config = defaults.copy()
    
    # Plugin provided defaults (pipeline should usually be lowest)
    merged_config.update(plugin_params1)
    merged_config.update(plugin_params2)
    
    # File configuration
    merged_config.update(file_config)
    
    # CLI arguments (argparse known)
    for k, v in cli_args.items():
        if v is not None:
            merged_config[k] = v
            
    # Unknown CLI arguments
    for k, v in unknown_args.items():
        merged_config[k] = convert_type(v)
        
    return merged_config
