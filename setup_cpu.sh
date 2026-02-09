#!/bin/bash
# Bacco CPU-Only Setup for JetPack 7.1
# Performance: ~4 FPS (CPU only, nessuna GPU)
# Uso: bash setup_cpu.sh

set -e

echo "╔════════════════════════════════════════════════════════╗"
echo "║                                                        ║"
echo "║     🍎 Bacco v2.0 - CPU-Only Setup (JetPack 7.1)      ║"
echo "║                                                        ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

# Check Python 3.10
if ! command -v python3.10 &> /dev/null; then
    echo "❌ Python 3.10 non trovato!"
    echo "Installazione Python 3.10..."
    sudo apt-get update
    sudo apt-get install -y python3.10 python3.10-dev python3-pip
fi

echo "✅ Python 3.10 trovato: $(python3.10 --version)"
echo ""

# Rimuovi PyTorch vecchi
echo "[1/4] Pulizia installazioni precedenti..."
python3.10 -m pip uninstall torch torchvision torchaudio -y 2>/dev/null || true
echo ""

# Installa dipendenze
echo "[2/4] Installazione dipendenze da requirements.txt..."
python3.10 -m pip install --user -r requirements.txt
echo ""

# Test importazioni
echo "[3/4] Test importazioni..."
python3.10 << 'PYEOF'
import sys
errors = []

try:
    import torch
    print(f"✅ PyTorch {torch.__version__} (CPU-only)")
    if torch.cuda.is_available():
        print("   ⚠️  CUDA disponibile ma non dovrebbe (CPU-only config)")
    else:
        print("   ℹ️  CUDA non disponibile (come atteso per CPU-only)")
except Exception as e:
    errors.append(f"❌ PyTorch: {e}")

try:
    import cv2
    print(f"✅ OpenCV {cv2.__version__}")
except Exception as e:
    errors.append(f"❌ OpenCV: {e}")

try:
    from ultralytics import YOLO
    print(f"✅ Ultralytics OK")
except Exception as e:
    errors.append(f"❌ Ultralytics: {e}")

try:
    import supervision
    print(f"✅ Supervision OK")
except Exception as e:
    errors.append(f"❌ Supervision: {e}")

if errors:
    print("\n❌ ERRORI:")
    for err in errors:
        print(f"   {err}")
    sys.exit(1)
else:
    print("\n✅ Tutte le librerie importate correttamente!")
PYEOF

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Setup fallito! Controlla errori sopra."
    exit 1
fi

echo ""

# Crea launcher
echo "[4/4] Creazione launcher..."
cat > run_bacco_cpu.sh << 'EOF'
#!/bin/bash
# Bacco CPU-Only Launcher

echo "🍎 Avvio Bacco in modalità CPU-only..."
echo "⚠️  Performance: ~4 FPS (per GPU vedere README.md)"
echo ""

python3.10 main.py "$@"
EOF

chmod +x run_bacco_cpu.sh

echo ""
echo "╔════════════════════════════════════════════════════════╗"
echo "║                                                        ║"
echo "║            ✅ SETUP COMPLETATO! ✅                     ║"
echo "║                                                        ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""
echo "📋 Come usare Bacco:"
echo ""
echo "   ./run_bacco_cpu.sh test_video.mp4"
echo ""
echo "   oppure:"
echo ""
echo "   python3.10 main.py test_video.mp4"
echo ""
echo "⚠️  NOTA: Modalità CPU-only - Performance ~4 FPS"
echo "   Per GPU (30 FPS): Vedi README.md 'Opzione A: PyTorch da Sorgente'"
echo ""