"""Agent plugins for agent-multi.

Each plugin implements this informal contract:

    class Plugin:
        plugin_params: dict           # default hyperparameters
        def __init__(self, config=None): ...
        def set_params(self, **kwargs): ...
        def build(self, env, config): return model
        def train(self, model, config): return model
        def predict(self, model, obs): return action
        def save(self, model, path): ...
        def load(self, path, env): return model
        def fitness(self, summary, config): return float
        # Optional:
        def hparam_schema(self): return [(name, low, high, type), ...]
"""
