#!/usr/bin/env bash
# Create the Python environment for this repo and pre-download XLS-R-300M.
#
#   bash scripts/setup_env.sh
#
# Overridable env vars:
#   PYTHON_BIN   python used to create the venv           (default: python3.12, else python3)
#   VENV_DIR     venv location                             (default: <repo>/.venv)
#   TORCH_INDEX  PyTorch wheel index (CUDA build)          (default: cu128; driver 595 supports up to CUDA 13.2)
#   HF_HOME      Hugging Face cache                        (default: ~/.cache/huggingface)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

if [ -z "${PYTHON_BIN:-}" ]; then
    if command -v python3.12 >/dev/null 2>&1; then PYTHON_BIN=python3.12; else PYTHON_BIN=python3; fi
fi
VENV_DIR="${VENV_DIR:-$REPO_DIR/.venv}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"

echo "=== Repo:   $REPO_DIR"
echo "=== Python: $($PYTHON_BIN --version) ($PYTHON_BIN)"
echo "=== Venv:   $VENV_DIR"

# 1. venv
if [ ! -x "$VENV_DIR/bin/python" ]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel setuptools

# 2. PyTorch (CUDA build) first, then the rest
pip install torch torchaudio --index-url "$TORCH_INDEX"
pip install -r requirements.txt

# 3. Sanity checks: CUDA, bf16, audio decoding (mp3 / NIST SPHERE need libsndfile >= 1.1)
python - <<'EOF'
import torch, soundfile, transformers
print(f"torch {torch.__version__} | CUDA build {torch.version.cuda} | transformers {transformers.__version__}")
assert torch.cuda.is_available(), "CUDA not available - check the driver / TORCH_INDEX"
print(f"GPU: {torch.cuda.get_device_name(0)} | bf16: {torch.cuda.is_bf16_supported()}")
print(f"libsndfile {soundfile.__libsndfile_version__} | formats: "
      f"{sorted(k for k in soundfile.available_formats() if k in ('WAV', 'FLAC', 'MP3', 'NIST'))}")
EOF

# 4. Pre-download the pretrained backbone (~1.2 GB) into the HF cache
python - <<'EOF'
from huggingface_hub import snapshot_download
path = snapshot_download("facebook/wav2vec2-xls-r-300m")
print("XLS-R-300M cached at", path)
EOF

# 5. Unit tests (CPU, ~10 s)
python -m pytest -q

echo
echo "Done. Activate with:  source $VENV_DIR/bin/activate"
