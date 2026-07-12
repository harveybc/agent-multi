#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$ROOT/.." && pwd)}"
BASE_PYTHON="${BASE_PYTHON:-$HOME/anaconda3/envs/tensorflow/bin/python}"
VENV="${AGENT_NAUTILUS_VENV:-$HOME/.venvs/agent-multi-nautilus}"

command -v uv >/dev/null 2>&1 || {
  echo "uv is required: https://docs.astral.sh/uv/" >&2
  exit 1
}
[[ -x "$BASE_PYTHON" ]] || {
  echo "BASE_PYTHON is not executable: $BASE_PYTHON" >&2
  exit 1
}

if [[ ! -x "$VENV/bin/python" ]]; then
  uv venv "$VENV" --python "$BASE_PYTHON" --system-site-packages
fi

# Keep NumPy on the TensorFlow-compatible line while meeting Nautilus's data
# dependencies inside the overlay environment.
uv pip install --python "$VENV/bin/python" \
  "nautilus_trader==1.230.0" \
  "numpy==2.1.3" \
  "pandas==2.3.3" \
  "pyarrow==25.0.0" \
  "pytest>=8,<9" \
  psutil

uv pip install --python "$VENV/bin/python" --no-deps \
  -e "$REPO_ROOT/trading-contracts" \
  -e "$REPO_ROOT/gym-fx" \
  -e "$ROOT"

"$VENV/bin/python" - <<'PY'
import nautilus_trader
import stable_baselines3
import tensorflow
import torch

print("nautilus_trader", nautilus_trader.__version__)
print("stable_baselines3", stable_baselines3.__version__)
print("tensorflow", tensorflow.__version__)
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
PY

echo "Combined agent environment ready: $VENV/bin/python"
