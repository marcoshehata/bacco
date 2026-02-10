#!/bin/bash
# Bacco GPU Setup for JetPack 7.1 (Jetson Thor)
# ==============================================
# Performance: ~40 FPS (GPU accelerated on Jetson Thor)
# Uso: bash setup_gpu.sh

set -e

echo "╔════════════════════════════════════════════════════════╗"
echo "║                                                        ║"
echo "║    🍎 Bacco v2.0 - GPU Setup (JetPack 7.1 Thor)       ║"
echo "║                                                        ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

# Colori
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ==============================================================================
# STEP 1: Verifica Sistema
# ==============================================================================
echo -e "${BLUE}[STEP 1/6]${NC} Verifica sistema..."

# Check Python 3.10
if ! command -v python3.10 &> /dev/null; then
    echo -e "${RED}❌ Python 3.10 non trovato!${NC}"
    echo "Installazione Python 3.10..."
    sudo apt-get update
    sudo apt-get install -y python3.10 python3.10-dev python3.10-venv python3-pip
fi
echo -e "${GREEN}✅${NC} Python 3.10: $(python3.10 --version)"

# Check nvidia-smi
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}❌ nvidia-smi non trovato! Driver NVIDIA non installato.${NC}"
    exit 1
fi
echo -e "${GREEN}✅${NC} NVIDIA Driver: OK"

# Check CUDA version
CUDA_VERSION=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9.]+" | head -1)
echo -e "${GREEN}✅${NC} CUDA Version: $CUDA_VERSION"

# Check JetPack
JETPACK_VERSION=$(dpkg-query --show nvidia-jetpack 2>/dev/null | awk '{print $2}' || echo "unknown")
echo -e "${GREEN}✅${NC} JetPack: $JETPACK_VERSION"
echo ""

# ==============================================================================
# STEP 2: Pulizia Installazioni Precedenti
# ==============================================================================
echo -e "${BLUE}[STEP 2/6]${NC} Pulizia installazioni PyTorch precedenti..."
python3.10 -m pip uninstall torch torchvision torchaudio -y 2>/dev/null || true
echo ""

# ==============================================================================
# STEP 3: Installazione cuSPARSELt (Dipendenza PyTorch)
# ==============================================================================
echo -e "${BLUE}[STEP 3/6]${NC} Verifica/installazione cuSPARSELt..."

# Check if cuSPARSELt is already installed
if [ -f "/usr/local/cuda/lib64/libcusparseLt.so" ] || [ -f "/usr/lib/aarch64-linux-gnu/libcusparseLt.so" ]; then
    echo -e "${GREEN}✅${NC} cuSPARSELt già installato"
else
    echo "📥 Download cuSPARSELt..."
    CUSPARSELT_NAME="libcusparse_lt-linux-aarch64-0.6.3.2-archive"
    TMP_DIR=$(mktemp -d)
    cd "$TMP_DIR"
    
    curl --retry 3 -OLs "https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-aarch64/${CUSPARSELT_NAME}.tar.xz" || {
        echo -e "${YELLOW}⚠️  Download fallito, continuo senza cuSPARSELt${NC}"
        cd "$SCRIPT_DIR"
        rm -rf "$TMP_DIR"
    }
    
    if [ -f "${CUSPARSELT_NAME}.tar.xz" ]; then
        tar xf "${CUSPARSELT_NAME}.tar.xz"
        sudo cp -a ${CUSPARSELT_NAME}/include/* /usr/local/cuda/include/ 2>/dev/null || true
        sudo cp -a ${CUSPARSELT_NAME}/lib/* /usr/local/cuda/lib64/ 2>/dev/null || true
        sudo ldconfig
        echo -e "${GREEN}✅${NC} cuSPARSELt installato"
    fi
    
    cd "$SCRIPT_DIR"
    rm -rf "$TMP_DIR"
fi
echo ""

# ==============================================================================
# STEP 4: Installazione PyTorch per JetPack 7.1 con CUDA 13.0
# ==============================================================================
echo -e "${BLUE}[STEP 4/6]${NC} Installazione PyTorch con CUDA 13.0..."

# PyTorch con CUDA 13.0 da pytorch.org wheel index
# Fonte: https://download.pytorch.org/whl/cu130
echo "📥 Download PyTorch+CUDA 13.0 da pytorch.org..."
python3.10 -m pip install --user torch torchvision --index-url https://download.pytorch.org/whl/cu130 || {
    echo -e "${RED}❌ Installazione PyTorch fallita!${NC}"
    echo ""
    echo -e "${YELLOW}SOLUZIONE ALTERNATIVA:${NC}"
    echo "1. Verifica connessione internet"
    echo "2. Riprova: pip install --user torch --index-url https://download.pytorch.org/whl/cu130"
    echo "3. Forum: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048"
    echo ""
    exit 1
}

echo -e "${GREEN}✅${NC} PyTorch+CUDA installato"
echo ""

# IMPORTANTE: Rimuovi le librerie CUDA pip-bundled che confliggono con JetPack
# Il wheel pytorch.org/whl/cu130 include nvidia-cublas, nvidia-cudnn, etc. che
# causano CUBLAS_STATUS_INVALID_VALUE su Jetson Thor.
# Soluzione: rimuovi queste librerie pip e usa quelle JetPack in /usr/local/cuda/lib64
echo "🔧 Rimozione librerie CUDA pip (uso system JetPack CUDA 13.0)..."
python3.10 -m pip uninstall nvidia-cublas nvidia-cudnn-cu13 nvidia-cufft nvidia-curand nvidia-cusolver nvidia-cusparse -y 2>/dev/null || true
echo -e "${GREEN}✅${NC} Configurazione CUDA sistema completata"
echo ""

# ==============================================================================
# STEP 5: Installazione Dipendenze Bacco
# ==============================================================================
echo -e "${BLUE}[STEP 5/6]${NC} Installazione dipendenze Bacco..."

# Installa da requirements_gpu.txt (senza torch)
if [ -f "requirements_gpu.txt" ]; then
    python3.10 -m pip install --user -r requirements_gpu.txt
else
    # Fallback: installa manualmente
    python3.10 -m pip install --user opencv-python==4.10.0.84
    python3.10 -m pip install --user ultralytics==8.3.36
    python3.10 -m pip install --user supervision==0.24.0
    python3.10 -m pip install --user numpy==1.26.4 Pillow==10.4.0 PyYAML==6.0.2 scipy==1.13.1 tqdm==4.66.5
fi

echo -e "${GREEN}✅${NC} Dipendenze installate"
echo ""

# ==============================================================================
# STEP 6: Verifica Installazione
# ==============================================================================
echo -e "${BLUE}[STEP 6/6]${NC} Verifica installazione..."

python3.10 << 'PYEOF'
import sys

print("Test importazioni...")
errors = []

# Test PyTorch
try:
    import torch
    print(f"✅ PyTorch {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"✅ CUDA Disponibile!")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
        
        # Test allocazione
        x = torch.randn(100, 100).cuda()
        print(f"   Test Tensor GPU: OK ({x.device})")
        del x
        torch.cuda.empty_cache()
    else:
        print("❌ CUDA NON disponibile!")
        errors.append("CUDA non funzionante")
except Exception as e:
    errors.append(f"PyTorch: {e}")

# Test OpenCV
try:
    import cv2
    print(f"✅ OpenCV {cv2.__version__}")
except Exception as e:
    errors.append(f"OpenCV: {e}")

# Test Ultralytics
try:
    from ultralytics import YOLO
    print(f"✅ Ultralytics OK")
except Exception as e:
    errors.append(f"Ultralytics: {e}")

# Test Supervision
try:
    import supervision
    print(f"✅ Supervision OK")
except Exception as e:
    errors.append(f"Supervision: {e}")

if errors:
    print("\n❌ ERRORI:")
    for err in errors:
        print(f"   {err}")
    sys.exit(1)
else:
    print("\n🎉 TUTTO OK!")
PYEOF

RESULT=$?

echo ""

if [ $RESULT -eq 0 ]; then
    echo "╔════════════════════════════════════════════════════════╗"
    echo "║                                                        ║"
    echo "║          ✅ SETUP GPU COMPLETATO! ✅                   ║"
    echo "║                                                        ║"
    echo "╚════════════════════════════════════════════════════════╝"
    echo ""
    echo "📋 Come usare Bacco:"
    echo ""
    echo "   ./run_bacco_gpu.sh test_video.mp4"
    echo ""
    echo "   oppure:"
    echo ""
    echo "   python3.10 main.py test_video.mp4"
    echo ""
    echo "🚀 Performance attese: ~40 FPS (GPU accelerato su Jetson Thor)"
else
    echo "╔════════════════════════════════════════════════════════╗"
    echo "║                                                        ║"
    echo "║              ❌ SETUP FALLITO! ❌                      ║"
    echo "║                                                        ║"
    echo "╚════════════════════════════════════════════════════════╝"
    echo ""
    echo "🔧 Troubleshooting:"
    echo "   1. python3.10 check_gpu.py    # Diagnostica dettagliata"
    echo "   2. nvidia-smi                 # Verifica driver"
    echo "   3. Consulta: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048"
    exit 1
fi
