#!/bin/bash
# Fix CUDA per Jetson Thor - Bacco Project
# =========================================

echo "🔧 FIX CUDA PER JETSON THOR"
echo "============================"
echo ""

# Colori
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Verifica se siamo in venv
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo -e "${RED}❌ Virtual environment non attivo!${NC}"
    echo "Esegui prima: source venv/bin/activate"
    exit 1
fi

echo -e "${GREEN}✅ Virtual environment attivo: $VIRTUAL_ENV${NC}"
echo ""

# Verifica versione CUDA sistema
echo "📋 Verifica CUDA sistema..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=driver_version --format=csv,noheader
    CUDA_VERSION=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9.]+")
    echo -e "${GREEN}✅ CUDA Version: $CUDA_VERSION${NC}"
else
    echo -e "${RED}❌ nvidia-smi non trovato${NC}"
    exit 1
fi

echo ""
echo "🗑️  Rimuovo vecchia installazione PyTorch..."
pip uninstall torch torchvision torchaudio -y

echo ""
echo "📦 Installo PyTorch con CUDA support..."

# Determina URL in base a CUDA version
if [[ "$CUDA_VERSION" == 13.* ]]; then
    echo "Uso torch index per CUDA 12.1 (compatibile con 13.0)"
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
elif [[ "$CUDA_VERSION" == 12.* ]]; then
    echo "Uso torch index per CUDA 12.1"
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
elif [[ "$CUDA_VERSION" == 11.* ]]; then
    echo "Uso torch index per CUDA 11.8"
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
else
    echo -e "${YELLOW}⚠️  CUDA version non riconosciuta, uso default${NC}"
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
fi

echo ""
echo "🧪 Test installazione..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "✅ FATTO!"
echo ""
echo "💡 Prossimo passo:"
echo "   python check_gpu.py    # Verifica completa"
echo "   python main.py video.mp4    # Testa con video"