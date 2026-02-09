#!/usr/bin/env python3
"""
Script Diagnostico GPU per Jetson Thor
======================================
Verifica installazione CUDA e PyTorch
"""

import sys

print("=" * 60)
print("🔍 DIAGNOSTICA GPU JETSON THOR")
print("=" * 60)

# 1. Verifica PyTorch
print("\n[1/5] Verifica PyTorch...")
try:
    import torch
    print(f"✅ PyTorch installato: {torch.__version__}")
except ImportError:
    print("❌ PyTorch NON installato!")
    sys.exit(1)

# 2. Verifica CUDA in PyTorch
print("\n[2/5] Verifica CUDA in PyTorch...")
cuda_available = torch.cuda.is_available()
print(f"CUDA disponibile: {cuda_available}")

if cuda_available:
    print(f"✅ CUDA Version: {torch.version.cuda}")
    print(f"✅ GPU Trovata: {torch.cuda.get_device_name(0)}")
    print(f"✅ Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("❌ CUDA NON disponibile in PyTorch!")
    print("\n🔧 POSSIBILI CAUSE:")
    print("   1. PyTorch installato senza supporto CUDA")
    print("   2. Driver NVIDIA non correttamente configurato")
    print("   3. CUDA toolkit non trovato")

# 3. Verifica CUDA Runtime
print("\n[3/5] Verifica CUDA Runtime...")
try:
    import subprocess
    result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ CUDA Toolkit installato")
        for line in result.stdout.split('\n'):
            if 'release' in line.lower():
                print(f"   {line.strip()}")
    else:
        print("❌ nvcc non trovato (CUDA toolkit non installato?)")
except FileNotFoundError:
    print("❌ nvcc non trovato nel PATH")

# 4. Verifica nvidia-smi
print("\n[4/5] Verifica nvidia-smi...")
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ nvidia-smi funzionante")
        # Estrae info GPU
        for line in result.stdout.split('\n'):
            if 'NVIDIA' in line or 'CUDA Version' in line:
                print(f"   {line.strip()}")
    else:
        print("❌ nvidia-smi fallito")
except FileNotFoundError:
    print("❌ nvidia-smi non trovato")

# 5. Test Tensor su GPU
print("\n[5/5] Test allocazione Tensor su GPU...")
if cuda_available:
    try:
        tensor = torch.randn(100, 100).cuda()
        print(f"✅ Tensor allocato su GPU: {tensor.device}")
        del tensor
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"❌ Errore allocazione: {e}")
else:
    print("⏭️  Skipped (CUDA non disponibile)")

# VERDETTO FINALE
print("\n" + "=" * 60)
print("📊 VERDETTO FINALE")
print("=" * 60)

if cuda_available:
    print("✅ GPU PRONTA! Il sistema può usare CUDA.")
    print("\n💡 Suggerimento:")
    print("   Riavvia il terminale e rilancia main.py")
else:
    print("❌ GPU NON FUNZIONANTE")
    print("\n🔧 SOLUZIONI:")
    print("\n--- Soluzione 1: Reinstalla PyTorch per Jetson ---")
    print("pip uninstall torch torchvision torchaudio -y")
    print("pip install torch torchvision torchaudio")
    print("\n--- Soluzione 2: Usa PyTorch wheel NVIDIA per Jetson ---")
    print("Visita: https://forums.developer.nvidia.com/t/pytorch-for-jetson/")
    print("Scarica il wheel per la tua versione JetPack e:")
    print("pip install <nome_wheel>.whl")
    print("\n--- Soluzione 3: Verifica JetPack ---")
    print("sudo apt update")
    print("sudo apt install nvidia-jetpack")

print("=" * 60 + "\n")