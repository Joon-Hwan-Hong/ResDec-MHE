#!/usr/bin/env bash
# Diff-test 5-fold: train all 5 folds with the diff_test_no_reg_with_flag.yaml
# config (return_attention_in_training=true + reg disabled = SDPA path with
# attention_weights returned). Sequential on GPU 0 to avoid GPU contention
# with the gradshap_smoothgrad job on GPU 1.
set -uo pipefail
WORKTREE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../" && pwd)"
cd "$WORKTREE_ROOT"
PYTHONPATH=. export PYTHONPATH

OUT_DIR="${OUT_DIR:-outputs/redesign/p5_diff_test}"
mkdir -p "$OUT_DIR"

for f in 0 1 2 3 4; do
    summary="$OUT_DIR/fold$f/summary.json"
    if [[ -f "$summary" ]]; then
        echo "[diff_test_5fold] fold $f already done — skipping"
        continue
    fi
    log="/tmp/diff_test_fold${f}.log"
    echo "[diff_test_5fold] launching fold $f → $log"
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/resdec_mhe/training/train.py \
        --config configs/resdec_mhe/diff_test_no_reg_with_flag.yaml \
        --fold "$f" \
        --output-dir "$OUT_DIR" \
        > "$log" 2>&1
    ec=$?
    if [[ $ec -eq 0 ]]; then
        echo "[diff_test_5fold] fold $f DONE"
    else
        echo "[diff_test_5fold] fold $f FAILED exit=$ec (continuing)"
    fi
done

echo "[diff_test_5fold] all folds attempted; summary:"
for f in 0 1 2 3 4; do
    if [[ -f "$OUT_DIR/fold$f/summary.json" ]]; then
        r2=$(uv run python -c "import json; print(json.load(open('$OUT_DIR/fold$f/summary.json'))['val_results'][0]['val/r2'])")
        echo "  fold $f: r2=$r2"
    else
        echo "  fold $f: MISSING"
    fi
done
