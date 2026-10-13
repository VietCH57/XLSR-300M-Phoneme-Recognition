#!/usr/bin/env bash
# Full pipeline on the 4090 server: (1) pre-convert audio, (2) train, (3) test inference.
#
#   bash scripts/run_train_server.sh                         # foreground
#   nohup bash scripts/run_train_server.sh > run.out 2>&1 &  # background (or run inside tmux)
#
# Overridable env vars:
#   CONFIG     (default configs/server_4090.yaml)
#   RUN_NAME   (default server_4090_<date>) -> output in runs/$RUN_NAME
#   SKIP_PREPARE=1   skip audio pre-conversion
#   EXTRA_ARGS       extra CLI overrides for train.py, e.g. EXTRA_ARGS="--domains native_en l2a el_easy"
#   RESUME=1         resume runs/$RUN_NAME/last (RUN_NAME must be given)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
# shellcheck disable=SC1091
source "${VENV_DIR:-$REPO_DIR/.venv}/bin/activate"

DATA_DIR="/home/mdd/lamnh/MDD-2026/data/stage2_combined/server"
TRAIN_CSV="$DATA_DIR/fullnative_el2a_train_ctc.csv"
DEV_CSV="$DATA_DIR/fullnative_el2a_dev_ctc.csv"
TEST_CSV="$DATA_DIR/fullnative_el2a_test_ctc.csv"
CONFIG="${CONFIG:-configs/server_4090.yaml}"
RUN_NAME="${RUN_NAME:-server_4090_$(date +%Y%m%d_%H%M)}"
OUT_DIR="runs/$RUN_NAME"
CACHE_DIR="$REPO_DIR/cache"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # less fragmentation with variable-length batches
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1                                  # loader workers decode audio in parallel already

for f in "$TRAIN_CSV" "$DEV_CSV" "$TEST_CSV" "$CONFIG"; do
    [ -f "$f" ] || { echo "ERROR: missing $f" >&2; exit 1; }
done
echo "=== Run: $OUT_DIR | config: $CONFIG | free disk: $(df -h "$REPO_DIR" | tail -1 | awk '{print $4}')"

# 1. Decode + resample every file once to 16 kHz FLAC (resumable; already converted files are skipped)
if [ "${SKIP_PREPARE:-0}" != "1" ]; then
    python scripts/prepare_audio.py --cache_dir "$CACHE_DIR" --num_workers 8 \
        --csv "$TRAIN_CSV" "$DEV_CSV" "$TEST_CSV"
fi

# 2. Train
if [ "${RESUME:-0}" = "1" ]; then
    python scripts/train.py --config "$OUT_DIR/config.yaml" --resume_from "$OUT_DIR/last" \
        --cache_dir "$CACHE_DIR" ${EXTRA_ARGS:-}
else
    # shellcheck disable=SC2086
    python scripts/train.py --config "$CONFIG" --output_dir "$OUT_DIR" --cache_dir "$CACHE_DIR" \
        --train_csv "$TRAIN_CSV" --dev_csv "$DEV_CSV" ${EXTRA_ARGS:-}
fi

# 3. Test with the best checkpoint (PER per domain in runs/$RUN_NAME/best/infer/)
python scripts/infer.py --ckpt "$OUT_DIR/best" --csv "$TEST_CSV" --cache_dir "$CACHE_DIR" \
    --gpu_memory_fraction 0.5 --max_batch_seconds 300

echo "=== Finished. Logs: $OUT_DIR/train.log | test report: $OUT_DIR/best/infer/"
