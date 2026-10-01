"""Name the missing mechanism instead of failing on an import line.

Each RL01-RL08 test asks for a mechanism by (module, attribute). While the
mechanism does not exist the test fails RED with a sentence that names it;
once it exists the test exercises its behaviour. A test that fails on
``ModuleNotFoundError`` tells the reader nothing about what is missing.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GYM_FX_ROOT = ROOT.parent / "gym-fx-g-rl-20261001"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def require(test_id: str, module: str, attribute: str, mechanism: str):
    """Return ``module.attribute`` or fail naming the mechanism it implements."""
    try:
        mod = importlib.import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name and (module == exc.name or module.startswith(exc.name + ".")):
            pytest.fail(
                f"{test_id} missing mechanism: {mechanism} "
                f"(expected {module}.{attribute}; module {module!r} does not exist)"
            )
        raise
    obj = getattr(mod, attribute, None)
    if obj is None:
        pytest.fail(
            f"{test_id} missing mechanism: {mechanism} "
            f"(expected {module}.{attribute}; attribute absent)"
        )
    return obj


def assert_checkout_resolution(test_id: str) -> None:
    """The .runtime editable finders shadow ``agent_plugins``; refuse that."""
    import agent_plugins
    import env_plugins
    import pipeline_plugins

    for mod in (agent_plugins, env_plugins, pipeline_plugins):
        path = Path(mod.__file__).resolve()
        assert ROOT in path.parents, (
            f"{test_id}: {mod.__name__} resolved to {path}, not this checkout "
            f"{ROOT}; a live .runtime checkout is being tested instead"
        )
