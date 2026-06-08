#!/usr/bin/env bash
# SpectrumKV GPU Experiments Runner
# Usage: bash run_gpu_experiments.sh [experiment_group]
#   experiment_group: exp1-8, p0, supp, niah, all (default: all)

set -e
cd "$(dirname "$0")/gpu-experiments"

GROUP="${1:-all}"

echo "=== SpectrumKV GPU Experiments: $GROUP ==="

case "$GROUP" in
    exp1|e1)
        python3 spectrumkv_scripts/exp1_ppl_fine_budget.py
        ;;
    exp2|e2)
        python3 spectrumkv_scripts/exp2_niah_fine_depth.py
        ;;
    exp3|e3)
        python3 spectrumkv_scripts/exp3_quant_error.py
        ;;
    exp4|e4)
        python3 spectrumkv_scripts/exp4_layer_budget.py
        ;;
    exp5-8|e5-8)
        python3 exp7_exp8_combined.py
        ;;
    p0)
        python3 p0_experiments.py
        ;;
    niah)
        python3 niah_depth_scan.py
        ;;
    supp)
        python3 supp_experiments.py
        ;;
    all)
        echo "Running all experiments sequentially..."
        for exp in exp1 exp2 exp3 exp4; do
            echo "--- $exp ---"
            bash run_gpu_experiments.sh "$exp"
        done
        echo "--- exp5-8 ---"
        python3 exp7_exp8_combined.py
        echo "--- p0 ---"
        python3 p0_experiments.py
        ;;
    *)
        echo "Unknown experiment group: $GROUP"
        echo "Available: exp1, exp2, exp3, exp4, exp5-8, p0, niah, supp, all"
        exit 1
        ;;
esac

echo ""
echo "=== Experiment $GROUP completed ==="
