#!/bin/bash
# Persistent benchmark agent loop — runs in a tmux session.
#
# Handles:
#   1. Waits for in-flight jobs (classical baselines, mixmil data prep)
#   2. Runs remaining benchmarks via orchestrator (scPhase, MixMIL, ABMIL, SetTransformer)
#   3. Invokes Claude Code agent for analysis after each
#   4. Final cross-benchmark comparison
#
# Usage:
#   tmux new-session -d -s benchmarks 'bash scripts/benchmark_agent_loop.sh'
#   tmux attach -t benchmarks

set -euo pipefail
cd /host/milan/tank/Joon/proj_ml_snrna

ANALYSIS_DIR="outputs/benchmark_analysis"
mkdir -p "$ANALYSIS_DIR" outputs/logs

echo "=========================================="
echo "  Benchmark Pipeline — $(date)"
echo "=========================================="

# ---- Phase 0: Wait for in-flight jobs ----
echo ""
echo "Phase 0: Waiting for in-flight jobs..."

# Wait for classical baselines
CLASSICAL_PID=$(ps aux | grep "run_baselines.py" | grep python3 | grep -v grep | awk '{print $2}' | head -1)
if [ -n "$CLASSICAL_PID" ]; then
    echo "  Classical baselines running (PID $CLASSICAL_PID) — waiting..."
    while kill -0 "$CLASSICAL_PID" 2>/dev/null; do
        PROGRESS=$(grep -c "R2=" outputs/logs/baselines_classical.log 2>/dev/null || echo "0")
        echo "    $(date +%H:%M:%S) — $PROGRESS experiments completed"
        sleep 120
    done
    echo "  Classical baselines finished at $(date)"
else
    echo "  No classical baselines running"
fi

# Wait for mixmil data prep
PREP_PID=$(ps aux | grep "prepare_data.py.*mixmil" | grep python | grep -v grep | awk '{print $2}' | head -1)
if [ -n "$PREP_PID" ]; then
    echo "  MixMIL data prep running (PID $PREP_PID) — waiting..."
    while kill -0 "$PREP_PID" 2>/dev/null; do
        echo "    $(date +%H:%M:%S) — still training scVI..."
        sleep 120
    done
    echo "  MixMIL data prep finished at $(date)"
    if [ -f baselines/shared/mixmil_input.h5ad ]; then
        echo "  mixmil_input.h5ad created successfully"
    else
        echo "  WARNING: mixmil_input.h5ad not found — MIL baselines will be blocked"
    fi
else
    echo "  No data prep running"
fi

echo ""
echo "Phase 0 complete — $(date)"

# ---- Phase 1: Run remaining benchmarks ----
echo ""
echo "=========================================="
echo "  Phase 1: Running remaining benchmarks"
echo "=========================================="

# The orchestrator skips already-completed (status.json seeded)
# and auto-runs data prep for blocked benchmarks
uv run python -u scripts/run_benchmarks.py \
    --device cuda:1 \
    --skip cloudpred cloudpred_pertype perceiver_io gpio classical \
    2>&1 | tee outputs/logs/orchestrator_phase1.log

echo ""
echo "Phase 1 complete — $(date)"

# ---- Phase 2: Final cross-benchmark analysis ----
echo ""
echo "=========================================="
echo "  Phase 2: Final analysis"
echo "=========================================="

claude -p "You are analyzing benchmark results for a cognitive resilience prediction project.

Working directory: /host/milan/tank/Joon/proj_ml_snrna
Our model: R²=0.323±0.067 (HPO7 production, 5-fold CV, 516 subjects)

TASKS:
1. Read outputs/benchmark_status.json for all completed benchmarks.
2. Read each results CSV: outputs/baselines/*/results.csv and outputs/baseline_results.csv
3. Update docs/results/2026-03-30-baseline-benchmarks.md comprehensively:
   - Per-fold tables for each completed baseline
   - Final ranking table sorted by R²
   - 3-4 sentence interpretation of what components matter
4. Write outputs/benchmark_analysis/final_summary.txt

Be thorough with the doc, concise in the summary." \
    --output-format text > "${ANALYSIS_DIR}/final_analysis.log" 2>&1 || true

echo ""
echo "=========================================="
echo "  All done — $(date)"
echo "  Results: docs/results/2026-03-30-baseline-benchmarks.md"
echo "  Status:  outputs/benchmark_status.json"
echo "=========================================="
