#!/usr/bin/env bash
# run_formal_study.sh - Full Eco-Reasoning Formal Study Pipeline
# Usage: bash run_formal_study.sh (inside activated venv)

set -e
echo "============================================================"
echo "  STEP 1: Retrain System 1 on Log Returns"
echo "============================================================"
python retrain_system1.py

echo ""
echo "============================================================"
echo "  STEP 2: System 1 Accuracy Benchmark (Log Returns)"
echo "============================================================"
python system1_benchmarker.py

echo ""
echo "============================================================"
echo "  STEP 3: Full Formal Study (Tasks 2, 3, 4)"
echo "============================================================"
python eco_formal_study.py
