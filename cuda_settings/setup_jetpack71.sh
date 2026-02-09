#!/bin/bash
# Setup Completo Bacco per JetPack 7.1 (Jetson Thor)
# ===================================================

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════╗"
echo "║                                                    ║"
echo "║   🍎 BACCO - Setup per JetPack 7.1 (Thor)  🍎    ║"
echo "║                                                    ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""

# Colori
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Verifica se siamo in venv
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo -e "${RED}❌ Virtual environment non attivo!${NC}"
    echo "Esegui prima: source venv/bin/activate"
    exit 1
fi

echo -e "${GREEN}✅ Virtual environment attivo${NC}"
echo ""

# Verifica JetPack
echo "📋 Verifica JetPack..."
JETPACK_VERSION=$(dpkg-query --show nvidia-jetpack 2>/dev/null | awk '{print $2}' || echo "unknown")
echo -e "${BLUE}JetPack Version: $JETPACK_VERSION${NC}"

if [[ "$JETPACK_VERSION" != 7.1* ]]; then
    echo -e "${YELLOW}⚠️  Questo script è ottimizzato per JetPack 7.1${NC}"
    echo -e "${YELLOW}   La tua versione è: $JETPACK_VERSION${NC}"
    echo ""
    read -p "Vuoi continuare comunque? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 1/4: Rimozione PyTorch Esistente"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "🗑️  Rimuovo installazioni PyTorch esistenti..."
pip uninstall torch torchvision torchaudio -y 2>/dev/null || true

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 2/4: Download PyTorch per JetPack 7.1"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

WHEEL_DIR="$HOME/pytorch_wheels"
mkdir -p "$WHEEL_DIR"
cd "$WHEEL_DIR"

# URL del wheel PyTorch per JetPack 7.1
TORCH_WHEEL_URL="https://developer.download.nvidia.com/compute/redist/jp/v71/pytorch/torch-2.6.0a0+b31f58d-cp312-cp312-linux_aarch64.whl"
TORCH_WHEEL="torch-2.6.0a0+b31f58d-cp312-cp312-linux_aarch64.whl"

if [ -f "$TORCH_WHEEL" ]; then
    echo -e "${GREEN}✅ Wheel già scaricato: $TORCH_WHEEL${NC}"
else
    echo "📥 Download PyTorch wheel per JetPack 7.1..."
    echo "   URL: $TORCH_WHEEL_URL"
    wget -q --show-progress "$TORCH_WHEEL_URL" || {
        echo -e "${RED}❌ Download fallito!${NC}"
        echo ""
        echo "Soluzione alternativa:"
        echo "1. Visita: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048"
        echo "2. Trova la sezione 'JetPack 7.1'"
        echo "3. Scarica manualmente il wheel torch-2.6.0*"
        echo "4. Mettilo in: $WHEEL_DIR/"
        echo "5. Rilancia questo script"
        exit 1
    }
    echo -e "${GREEN}✅ Download completato${NC}"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 3/4: Installazione PyTorch"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "📦 Installo PyTorch..."
pip install "$TORCH_WHEEL"

echo ""
echo "📦 Installo torchvision..."
pip install torchvision==0.19.0

echo -e "${GREEN}✅ PyTorch installato con successo${NC}"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 4/4: Installazione Dipendenze Bacco"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Torna alla directory Bacco
cd "$VIRTUAL_ENV/.."

echo "📦 Installo requirements.txt..."
pip install -r requirements.txt

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 5/5: Verifica Installazione"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "🧪 Test PyTorch + CUDA..."
python3 << 'PYTHON_EOF'
import sys
import torch

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA disponibile: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ CUDA Version: {torch.version.cuda}")
    print(f"✅ Compute Capability: {torch.cuda.get_device_capability(0)}")
    
    # Test allocazione
    try:
        x = torch.randn(100, 100).cuda()
        print(f"✅ Test allocazione GPU: OK")
    except Exception as e:
        print(f"❌ Test allocazione GPU: FAILED - {e}")
else:
    print("❌ CUDA non disponibile!")
    sys.exit(1)
PYTHON_EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "╔════════════════════════════════════════════════════╗"
    echo "║                                                    ║"
    echo "║          ✅ SETUP COMPLETATO CON SUCCESSO ✅      ║"
    echo "║                                                    ║"
    echo "╚════════════════════════════════════════════════════╝"
    echo ""
    echo "🎯 Prossimi passi:"
    echo "   1. python main.py test_video.mp4"
    echo "   2. Controlla che FPS sia ~30 (non ~4)"
    echo ""
else
    echo ""
    echo -e "${RED}╔════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║                                                    ║${NC}"
    echo -e "${RED}║            ❌ SETUP FALLITO - GPU NON OK ❌       ║${NC}"
    echo -e "${RED}║                                                    ║${NC}"
    echo -e "${RED}╚════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "🔧 Troubleshooting:"
    echo "   1. Verifica driver: nvidia-smi"
    echo "   2. Controlla JetPack: dpkg-query --show nvidia-jetpack"
    echo "   3. Consulta: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048"
    exit 1
fi