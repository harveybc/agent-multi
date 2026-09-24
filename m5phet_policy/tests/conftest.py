"""Test this checkout, not whatever copy happens to be installed in the interpreter running pytest.

The provider is also installed into the operator's chat environment. Without this, `pytest m5phet_policy/tests` from that
interpreter imports the installed copy, and a change made here would be reported as green without ever being executed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
