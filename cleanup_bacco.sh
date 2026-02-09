#!/bin/bash
echo "🧹 PULIZIA COMPLETA SISTEMA BACCO"
echo "=================================="

cd ~/Desktop/fiera_rimini_2026/ready_to_use/Bacco

# 1. RIMUOVI TUTTI I VIRTUAL ENVIRONMENTS
echo "1. Rimuovo virtual environments..."
rm -rf venv/
rm -rf venv311/
rm -rf env/
rm -rf .venv/

# 2. RIMUOVI INSTALLAZIONI PYTHON PRECEDENTI (tutte le versioni)
echo "2. Rimuovo installazioni PyTorch/dipendenze precedenti..."
python3.10 -m pip uninstall torch torchvision torchaudio -y 2>/dev/null || true
python3.11 -m pip uninstall torch torchvision torchaudio -y 2>/dev/null || true
python3.12 -m pip uninstall torch torchvision torchaudio -y 2>/dev/null || true
python3 -m pip uninstall torch torchvision torchaudio -y 2>/dev/null || true

python3.10 -m pip uninstall numpy opencv-python opencv-contrib-python ultralytics supervision Pillow PyYAML scipy tqdm -y 2>/dev/null || true
python3.11 -m pip uninstall numpy opencv-python opencv-contrib-python ultralytics supervision Pillow PyYAML scipy tqdm -y 2>/dev/null || true

# 3. RIMUOVI CONTAINER DOCKER INUTILI
echo "3. Rimuovo container Docker..."
sudo docker stop $(sudo docker ps -aq) 2>/dev/null || true
sudo docker rm $(sudo docker ps -aq) 2>/dev/null || true
sudo docker rmi nvcr.io/nvidia/l4t-ml:r36.2.0-py3 2>/dev/null || true
sudo docker rmi dustynv/pytorch:2.1-r36.2.0 2>/dev/null || true
sudo docker system prune -a -f

# 4. RIMUOVI JETSON-CONTAINERS
echo "4. Rimuovo jetson-containers..."
rm -rf jetson-containers/

# 5. RIMUOVI TORCHVISION SOURCE
echo "5. Rimuovo torchvision source..."
rm -rf torchvision/

# 6. RIMUOVI WHEEL SCARICATI
echo "6. Rimuovo wheel scaricati..."
rm -f *.whl
rm -f install_cusparselt.sh

# 7. PULISCI CACHE PIP
echo "7. Pulisco cache pip..."
python3.10 -m pip cache purge 2>/dev/null || true

# 8. PULISCI SYSTEM
echo "8. Pulizia sistema..."
sudo apt autoremove -y
sudo apt clean

echo ""
echo "✅ PULIZIA COMPLETATA!"
echo ""
echo "Spazio liberato:"
df -h | grep -E "Filesystem|/$"
echo ""