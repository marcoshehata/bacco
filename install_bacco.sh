#!/bin/bash
echo "🚀 INSTALLAZIONE PULITA BACCO + PYTORCH"
echo "========================================"

cd ~/Desktop/fiera_rimini_2026/ready_to_use/Bacco

# STEP 1: Verifica Python 3.10
echo ""
echo "[STEP 1/6] Verifica Python 3.10..."
if ! command -v python3.10 &> /dev/null; then
    echo "❌ Python 3.10 non trovato. Installazione..."
    sudo apt-get update
    sudo apt-get install -y python3.10 python3.10-venv python3.10-dev
else
    echo "✅ Python 3.10 trovato: $(python3.10 --version)"
fi

# STEP 2: Installa cuSPARSELt (OBBLIGATORIO per PyTorch 2.5+)
echo ""
echo "[STEP 2/6] Installazione cuSPARSELt..."
cat > install_cusparselt.sh << 'EOF'
#!/bin/bash
set -e
mkdir -p tmp_cusparselt && cd tmp_cusparselt
CUSPARSELT_NAME="libcusparse_lt-linux-aarch64-0.6.3.2-archive"
echo "Download cuSPARSELt..."
curl --retry 3 -OLs https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-aarch64/${CUSPARSELT_NAME}.tar.xz
echo "Estrazione..."
tar xf ${CUSPARSELT_NAME}.tar.xz
echo "Installazione librerie..."
sudo cp -a ${CUSPARSELT_NAME}/include/* /usr/local/cuda/include/
sudo cp -a ${CUSPARSELT_NAME}/lib/* /usr/local/cuda/lib64/
cd ..
rm -rf tmp_cusparselt
sudo ldconfig
echo "✅ cuSPARSELt installato"
EOF

chmod +x install_cusparselt.sh
sudo bash ./install_cusparselt.sh

# STEP 3: Installa PyTorch con CUDA
echo ""
echo "[STEP 3/6] Download e installazione PyTorch..."
python3.10 -m pip install --user --no-cache https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl

# STEP 4: Verifica CUDA
echo ""
echo "[STEP 4/6] Test CUDA..."
CUDA_RESULT=$(python3.10 -c "import torch; print('True' if torch.cuda.is_available() else 'False')" 2>&1)

if [[ "$CUDA_RESULT" == *"True"* ]]; then
    echo "✅ CUDA FUNZIONA!"
    python3.10 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
else
    echo "❌ CUDA NON FUNZIONA!"
    echo "Output: $CUDA_RESULT"
    echo ""
    echo "Interrompo installazione. Controlla errori sopra."
    exit 1
fi

# STEP 5: Installa dipendenze Bacco
echo ""
echo "[STEP 5/6] Installazione dipendenze Bacco..."
python3.10 -m pip install --user opencv-python==4.10.0.84
python3.10 -m pip install --user ultralytics==8.3.36
python3.10 -m pip install --user supervision==0.24.0
python3.10 -m pip install --user numpy==1.26.4
python3.10 -m pip install --user Pillow==10.4.0
python3.10 -m pip install --user PyYAML==6.0.2
python3.10 -m pip install --user scipy==1.13.1
python3.10 -m pip install --user tqdm==4.66.5

# STEP 6: Test importazioni
echo ""
echo "[STEP 6/6] Test importazioni..."
python3.10 << 'PYEOF'
import sys
print("Test importazioni...")
try:
    import torch
    print(f"✅ torch {torch.__version__}")
except Exception as e:
    print(f"❌ torch: {e}")
    sys.exit(1)

try:
    import cv2
    print(f"✅ opencv {cv2.__version__}")
except Exception as e:
    print(f"❌ opencv: {e}")
    sys.exit(1)

try:
    import ultralytics
    print(f"✅ ultralytics OK")
except Exception as e:
    print(f"❌ ultralytics: {e}")
    sys.exit(1)

try:
    import supervision
    print(f"✅ supervision OK")
except Exception as e:
    print(f"❌ supervision: {e}")
    sys.exit(1)

print("")
print("🎉 TUTTE LE IMPORTAZIONI FUNZIONANO!")
PYEOF

# STEP 7: Crea script launcher
echo ""
echo "Creazione script launcher..."
cat > run_bacco.sh << 'EOF'
#!/bin/bash
python3.10 main.py "$@"
EOF
chmod +x run_bacco.sh

echo ""
echo "╔════════════════════════════════════════════════════╗"
echo "║                                                    ║"
echo "║        ✅ INSTALLAZIONE COMPLETATA! ✅            ║"
echo "║                                                    ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""
echo "📋 Come usare Bacco:"
echo "   ./run_bacco.sh test_video.mp4"
echo ""
echo "   oppure:"
echo ""
echo "   python3.10 main.py test_video.mp4"
echo ""