#!/bin/bash
# Phase B Runner - run G5, G6, G7+G8 sequentially without stopping
set -e
echo "[$(date)] Starting Phase B sequential runner"

echo "[$(date)] === Running G5 (Eviction) ==="
/root/miniconda3/bin/python /root/autodl-tmp/kvcache-lab/gpu-experiments/run_g5_eviction.py 2>&1 | tee /root/autodl-tmp/experiment_g5.log
echo "[$(date)] G5 DONE"

echo "[$(date)] === Running G6 (Full OS Ablation) ==="
/root/miniconda3/bin/python /root/autodl-tmp/kvcache-lab/gpu-experiments/run_g6_ablation.py 2>&1 | tee /root/autodl-tmp/experiment_g6.log
echo "[$(date)] G6 DONE"

echo "[$(date)] === Running G7+G8 (Concurrency + Quality) ==="
/root/miniconda3/bin/python /root/autodl-tmp/kvcache-lab/gpu-experiments/run_g7g8_combined.py 2>&1 | tee /root/autodl-tmp/experiment_g7g8.log
echo "[$(date)] G7+G8 DONE"

echo "[$(date)] === ALL PHASE B EXPERIMENTS COMPLETE ==="
