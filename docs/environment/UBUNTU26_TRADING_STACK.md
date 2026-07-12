# Ubuntu 26 Trading Stack

Canonical environment for `predictor`, `gym-fx`, `agent-multi`, `heuristic-strategy`, and the DOIN repositories.

## Runtime contract

- Conda environment: `trading-stack`
- Python: `3.12.13`
- pip: `26.1.2`
- NVIDIA driver tested: `580.159.03`
- TensorFlow: `2.21.0` with `tf-keras 2.21.0` and `tensorflow-probability 0.25.0`
- PyTorch: `2.13.0+cu130`
- NautilusTrader: `1.230.0`
- Stable-Baselines3: `2.9.0`
- Gymnasium: `1.3.0`
- NumPy: `2.5.1`
- pandas: `3.0.3`
- Protobuf: `7.35.1`

The CUDA 12 packages are TensorFlow's runtime and the CUDA 13 packages are PyTorch's runtime. Both are intentional and must remain installed together. The system CUDA toolkit is not used as the Python dependency source.

## Recreate on Ubuntu 26

From a checkout containing this file and sibling repositories:

```bash
conda env remove -n trading-stack -y
conda env create -f agent-multi/docs/environment/ubuntu26-trading-stack.yml
conda run -n trading-stack python -m pip install -r agent-multi/docs/environment/trading-stack-pip-lock-2026-07-11.txt
conda run -n trading-stack python -m pip install -e trading-contracts -e doin-core -e doin-plugins -e doin-node
conda run -n trading-stack python -m pip install -e predictor -e gym-fx -e agent-multi -e heuristic-strategy
```

The lock file contains third-party packages only. Local repositories are installed separately as editable packages so the same environment uses the current checkout after a Git update.

Before starting a node or experiment, verify the machine driver and dependency graph:

```bash
nvidia-smi
conda run -n trading-stack python -m pip check
conda run -n trading-stack python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
conda run -n trading-stack python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Do not activate or recreate the old `tensorflow` environment for this work. It contained an incompatible TensorFlow 2.19 / OR-Tools 9.15 / Protobuf 5 combination and is not part of the canonical stack.

## Acceptance evidence on omega

- `pip check`: no broken requirements.
- TensorFlow GPU operation: passed.
- PyTorch CUDA operation: passed.
- `agent-multi` unit tests: 152 passed.
- `gym-fx` tests: 41 passed, with only Nautilus deprecation warnings.
- `doin-node`: 307 passed; 3 repository-level failures remain documented separately because they are contradictory or pre-existing logic tests, not environment failures.
- Predictor plugin smoke imports: passed for TCN, TFT, Transformer, binary TCN, direction TCN, pipeline, preprocessor, and target plugins.
