#!/bin/bash
# Bacco GPU Launcher
# ==================
# Performance: ~40 FPS (GPU accelerato su Jetson Thor)

echo "🍎 Avvio Bacco in modalità GPU..."
echo ""

# IMPORTANTE: Usa le librerie CUDA di sistema JetPack invece di quelle pip
# Questo risolve conflitti CUBLAS tra pip nvidia-* packages e JetPack CUDA 13.0
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0
export CUDA_MODULE_LOADING=LAZY

python3.10 main.py "$@"

