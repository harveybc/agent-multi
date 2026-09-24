"""Test this checkout, not whatever copy happens to be installed in the interpreter running pytest.

The provider is also installed into the operator's chat environment. Without this, `pytest m5phet_policy/tests` from that
interpreter imports the installed copy, and a change made here would be reported as green without ever being executed.

The question envelope (`m5phet.questions`) is newer than some installed copies of m5phet. When the installed package has
no `questions` module, an m5phet source tree is looked for -- `M5PHET_SRC`, else the sibling `m5phet-chat` worktree -- and
put ahead of the installed copy. Nothing is stubbed: without either, the contract tests skip and say so.
"""

import importlib.util
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _m5phet_source_with_questions():
    candidates = [os.environ.get("M5PHET_SRC"),
                  Path(__file__).resolve().parents[3] / ".worktrees" / "m5phet-chat" / "src",
                  Path(__file__).resolve().parents[3] / "m5phet" / "src"]
    for candidate in candidates:
        if candidate and Path(candidate, "m5phet", "questions.py").is_file():
            return str(candidate)
    return None


_installed = importlib.util.find_spec("m5phet")
if _installed is None or not Path(_installed.origin).with_name("questions.py").is_file():
    _source = _m5phet_source_with_questions()
    if _source:
        for _name in [n for n in sys.modules if n.split(".")[0] == "m5phet"]:
            del sys.modules[_name]
        sys.path.insert(0, _source)
