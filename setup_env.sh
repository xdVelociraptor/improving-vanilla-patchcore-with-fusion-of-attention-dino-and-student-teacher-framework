#!/usr/bin/env bash
# Build the pinned NVIDIA/CUDA environment on Linux (requires matching wheels).
#   bash setup_env.sh        builds .venv-linux from requirements.lock.txt (downloads; needs Python 3.12)
# For macOS or a CPU environment, follow the portable setup in README.md instead.
set -euo pipefail
cd "$(dirname "$0")"
PYTHON=${PYTHON:-python3.12}
VENV=.venv-linux
if [ ! -x "$VENV/bin/python" ]; then "$PYTHON" -m venv "$VENV"; fi
"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install -r requirements.lock.txt --extra-index-url https://download.pytorch.org/whl/cu128
"$VENV/bin/python" -m pip install -e . --no-deps --no-build-isolation
"$VENV/bin/python" -m hybridad.cli --help
