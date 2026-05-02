#!/usr/bin/env bash
# Diff-test 5-fold: train all 5 folds with the diff_test_no_reg_with_flag.yaml
# config (return_attention_in_training=true + reg disabled = SDPA path with
# attention_weights returned). Sequential on GPU 0 to avoid GPU contention
# with the gradshap_smoothgrad job on GPU 1.
set -uo pipefail
WORKTREE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../" && pwd)"
cd "$WORKTREE_ROOT"
PYTHONPATH=. export PYTHONPATH

OUT_DIR="${OUT_DIR:-outputs/canonical/p5_diff_test}"
mkdir -p "$OUT_DIR"
# Co-locate logs with sweep outputs (B-DT2: /tmp does not survive reboot;
# matches the layout in run_sae_sweep_smaller_m.sh:142-144).
LOG_DIR="$OUT_DIR/_diff_test_logs"
mkdir -p "$LOG_DIR"

for f in 0 1 2 3 4; do
    summary="$OUT_DIR/fold$f/summary.json"
    if [[ -f "$summary" ]]; then
        echo "[diff_test_5fold] fold $f already done — skipping"
        continue
    fi
    log="$LOG_DIR/diff_test_fold${f}.log"
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
    summary_path="$OUT_DIR/fold$f/summary.json"
    if [[ -f "$summary_path" ]]; then
        # Pass path via env so the inline command stays quote-safe regardless
        # of $OUT_DIR contents (single-quote / dollar-sign / spaces).
        r2=$(SUMMARY_PATH="$summary_path" uv run python -c "import json, os; print(json.load(open(os.environ['SUMMARY_PATH']))['val_results'][0]['val/r2'])")
        echo "  fold $f: r2=$r2"
    else
        echo "  fold $f: MISSING"
    fi
done
