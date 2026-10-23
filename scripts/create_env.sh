#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ENV_NAME=${1:-chinamobile-ser-dace-accr}
CONDA_BIN=${CONDA_BIN:-conda}

"$CONDA_BIN" create -y -n "$ENV_NAME" python=3.10 pip
"$CONDA_BIN" run -n "$ENV_NAME" python -m pip install --upgrade pip
"$CONDA_BIN" run -n "$ENV_NAME" python -m pip install \
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
"$CONDA_BIN" run -n "$ENV_NAME" python -m pip install -r "$ROOT/requirements.txt"
"$CONDA_BIN" run -n "$ENV_NAME" python "$ROOT/scripts/verify_install.py" \
  --skip-base-model

echo "Environment created: $ENV_NAME"
echo "Activate with: conda activate $ENV_NAME"
