# 🍎 Bacco - Apple Detection System v2.0

Sistema di rilevamento e tracciamento automatico delle mele su video, ottimizzato per NVIDIA Jetson Thor.

## ✨ Caratteristiche

- **Object Detection**: YOLOWorld-Small con prompts personalizzati
- **Multi-Object Tracking**: ByteTrack per tracking stabile nel tempo
- **Class Merging + NMS**: Eliminazione automatica double counting da multi-prompt
- **CLAHE Enhancement**: Migliora detection mele parziali/in ombra
- **Maschere Ellittiche**: Copertura precisa delle mele basata su bounding box
- **ID Persistenti**: Registry unico per conteggio accurato nel tempo
- **Temporal Smoothing**: Riduzione jitter coordinate (media mobile 5 frame)
- **Colori Fissi**: Rosso per bbox/ellisse, Bordeaux per traiettorie
- **Visualizzazione Real-time**: 
  - Bounding boxes rosse
  - Ellissi semitrasparenti rosse
  - Traiettorie bordeaux
  - HUD con statistiche live
- **Resize Adattivo**: Gestione automatica di qualsiasi risoluzione (verticale/orizzontale)
- **GPU Optimization**: Supporto CUDA con diagnostica automatica
- **Performance**: Target 20+ FPS su Jetson Thor con GPU

---

## 📋 Requisiti

- **Hardware**: NVIDIA Jetson Thor (o GPU CUDA 13.0+)
- **Python**: 3.10+
- **CUDA**: 13.0
- **Sistema**: Linux (Ubuntu)

---

## 🚀 Installazione

### Metodo 1: Setup Automatico GPU (Consigliato per Jetson Thor)

```bash
cd Bacco
bash setup_gpu.sh
```

Lo script esegue automaticamente:
1. Verifica Python 3.10 e CUDA 13.0
2. Installa PyTorch 2.10.0+cu130 da PyTorch.org
3. Rimuove le librerie CUDA pip che confliggono con JetPack
4. Installa le dipendenze Bacco
5. Verifica che tutto funzioni

### Metodo 2: Setup Manuale (se lo script fallisce)

```bash
# 1. Installa PyTorch con CUDA 13.0
pip install --user torch torchvision --index-url https://download.pytorch.org/whl/cu130

# 2. IMPORTANTE: Rimuovi le librerie CUDA pip che confliggono con JetPack
pip uninstall nvidia-cublas nvidia-cudnn-cu13 nvidia-cufft nvidia-curand nvidia-cusolver nvidia-cusparse -y

# 3. Installa dipendenze Bacco
pip install -r requirements_gpu.txt

# 4. Verifica GPU
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
python check_gpu.py
```

> **⚠️ IMPORTANTE**: I wheel PyTorch da pytorch.org includono librerie CUDA (nvidia-cublas, ecc.) che confliggono con quelle di JetPack. La rimozione al passo 2 è obbligatoria per evitare errori `CUBLAS_STATUS_INVALID_VALUE`.

### Verificare Setup GPU

```bash
python check_gpu.py
```

Dovresti vedere:
```
✅ PyTorch installato: 2.10.0+cu130
✅ CUDA disponibile: True
✅ GPU Trovata: NVIDIA Thor
✅ Tensor allocato su GPU: cuda:0
```

---

## 🎬 Utilizzo

### Metodo 1: Launcher GPU (Consigliato)

```bash
./run_bacco_gpu.sh test_video.mp4
```

Il launcher imposta automaticamente `LD_LIBRARY_PATH` per usare le librerie CUDA di sistema.

### Metodo 2: Da Terminale direttamente

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
python3.10 main.py test_video.mp4
```

### Metodo 3: Input Interattivo

```bash
./run_bacco_gpu.sh
# Il programma chiederà il path del video
```

---

## ⚙️ Configurazione

Tutte le configurazioni sono in `main.py` nella classe `Config`:

```python
class Config:
    # Detection
    YOLO_CONFIDENCE = 0.03  # Threshold detection (0.01-1.0) - ABBASSATO per mele parziali
    YOLO_PROMPTS = ["apple", "red apple", "green apple"]  # Multi-prompt per recall
    NMS_THRESHOLD = 0.4     # NUOVO: NMS per eliminare duplicati
    
    # Enhancement
    USE_CLAHE = True        # NUOVO: Abilita CLAHE per mele in ombra
    CLAHE_CLIP_LIMIT = 2.0
    CLAHE_TILE_SIZE = (8, 8)
    
    # Tracking
    TRACK_BUFFER = 60       # Frame buffer prima di perdere ID
    MATCH_THRESH = 0.7      # Soglia IoU per matching
    
    # Resize
    RESIZE_STRATEGY = "adaptive"  # 'adaptive', 'native', 'fixed'
    TARGET_SIZE = 1280      # Max dimensione lato lungo
    MIN_SIZE = 640          # Min dimensione per resize
    
    # Visualizzazione (COLORI FISSI)
    BBOX_COLOR = (0, 0, 255)         # Rosso BGR
    ELLIPSE_COLOR = (0, 0, 255)      # Rosso BGR
    TRAJECTORY_COLOR = (0, 0, 139)   # Bordeaux BGR
    MAX_TRAJECTORY_POINTS = 30
    ELLIPSE_ALPHA = 0.3
```

### Strategie di Resize

- **`adaptive`** (default): Ridimensiona solo se > 1280px, ottimale per performance
- **`native`**: Mantiene risoluzione originale, massima qualità ma più lento
- **`fixed`**: Forza resize a 1280px, utile per testing

---

## 📊 Output

### Display Real-time

- **Bounding Box**: Rettangolo colorato per ogni ID univoco
- **Ellisse**: Maschera semitrasparente sovrapposta
- **Traiettoria**: Linea che segue il movimento (ultimi 30 punti)
- **Label**: `ID:XXX (confidence)` sopra ogni mela
- **HUD** (alto-sinistra):
  - Numero frame corrente
  - Mele visibili nel frame
  - Totale mele uniche rilevate
  - FPS corrente

### Log Terminale

```
Frame 0245 | Current: 8 apples | Total: 23 unique | FPS: 28.3
[APPLE] 🍎 Nuova mela rilevata! ID 012 | Totale unico: 24
[TRACK] 👻 ID 019 perso da 60 frame (rimosso da attivi)
```

---

## 🎮 Controlli

- **`q`**: Esci dal processing
- **`Ctrl+C`**: Interruzione immediata

---

## 🔧 Troubleshooting

### Problema: GPU non riconosciuta (FPS bassi ~4)

**Sintomo**: `[CUDA] ⚠️ GPU non disponibile, uso CPU`

**Soluzione**:
```bash
# Step 1: Diagnostica
python check_gpu.py

# Step 2: Setup GPU (installa PyTorch CUDA + rimuove lib conflittuali)
bash setup_gpu.sh

# Step 3: Usa il launcher GPU
./run_bacco_gpu.sh video.mp4
```

### Problema: "ModuleNotFoundError: No module named 'torch'"

**Soluzione**: Assicurati di aver attivato il venv
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Problema: FPS troppo bassi (<20) con GPU attiva

**Soluzioni**:
1. Riduci `TARGET_SIZE` a 960 o 640
2. Disattiva CLAHE: `USE_CLAHE = False`
3. Usa `RESIZE_STRATEGY = "fixed"` con `TARGET_SIZE = 640`

### Problema: Mele piccole non rilevate

**Soluzioni**:
1. Abbassa `YOLO_CONFIDENCE` a 0.02
2. Assicurati che `USE_CLAHE = True`
3. Usa `RESIZE_STRATEGY = "native"` (più lento ma più accurato)

### Problema: Double counting (stessa mela conta 2 volte)

**Soluzione**: Il sistema ora usa class merging + NMS automatico. Se persiste:
1. Aumenta `NMS_THRESHOLD` a 0.5 o 0.6
2. Verifica che prompts siano tutti attivi

### Problema: ID cambiano troppo spesso

**Soluzioni**:
1. Aumenta `TRACK_BUFFER` a 90 o 120
2. Riduci `MATCH_THRESH` a 0.6
3. Attiva smoothing aumentando `SMOOTH_WINDOW` a 7

---

## 📁 Struttura Progetto

```
Bacco/
├── models/                    # Modelli scaricati
│   ├── .gitkeep
│   └── yolov8s-worldv2.pt    # (scaricato automaticamente)
├── main.py                    # Codice principale v2.0 GPU
├── check_gpu.py               # Script diagnostica GPU
├── setup_gpu.sh               # Setup automatico GPU (JetPack 7.1)
├── run_bacco_gpu.sh           # Launcher con LD_LIBRARY_PATH
├── requirements.txt           # Dipendenze Python
├── requirements_gpu.txt       # Dipendenze GPU (no PyTorch)
├── .gitignore                 # Git ignore
└── README.md                  # Questo file
```

---

## 🧪 Test Scenari

### Video da iPhone (verticale)
```bash
python main.py ~/Videos/mele_iphone_vertical.mp4
# Risoluzione: 1080x1920 → resize a 720x1280
# FPS atteso: ~30
```

### Video da Meta Ray-Ban Glasses
```bash
python main.py ~/Videos/mele_meta_glasses.mp4
# Risoluzione: 1408x1408 → mantiene native
# FPS atteso: ~32
```

### Video 4K
```bash
python main.py ~/Videos/mele_4k.mp4
# Risoluzione: 3840x2160 → resize a 1280x720
# FPS atteso: ~25
```

---

## 🐛 Debug Mode

Per log dettagliati, modifica in `main.py`:

```python
# Nel metodo process_frame(), aggiungi:
if self.frame_count % 10 == 0:  # Log ogni 10 frame invece di 30
    print(f"[DEBUG] Detections: {len(detections)}, Tracks: {len(track_ids)}")
```

---

## 📝 Note Tecniche

### Perché Confidence 0.03?

YOLOWorld è conservativo con zero-shot detection. Confidence 0.03 (abbassato da 0.05) permette di catturare:
- Mele parzialmente visibili
- Mele in ombra o dietro foglie
- Mele lontane o piccole

ByteTrack filtrerà comunque false positive instabili.

### Perché Class Merging + NMS?

I prompt multipli `["apple", "red apple", "green apple"]` aumentano il recall ma causano double counting:
- "apple" detecta mela → bbox1
- "red apple" detecta STESSA mela → bbox2

**Soluzione**: Unifichiamo tutte le classi (class merging) e applichiamo NMS aggressivo (threshold 0.4) per rimuovere duplicati.

### Perché CLAHE?

CLAHE (Contrast Limited Adaptive Histogram Equalization) migliora contrasto locale senza overhead significativo:
- Migliora detection mele in ombra
- Aiuta con occlusioni parziali (foglie, rami)
- Preprocessing veloce (<0.5 FPS overhead)

### Perché ByteTrack?

- Gestisce occlusioni temporanee
- Riassegna ID corretti dopo re-identificazione
- Molto stabile nel tempo rispetto a altri tracker

### Colori Fissi

- **Rosso (255,0,0)**: Bbox + Ellisse (fill con 0.6 alpha)
- **Blu sfondo (255,0,0 BGR)**: Filtro sfondo con 0.4 alpha
- **Bordeaux (139,0,0)**: Traiettoria → Mostra movimento nel tempo

### Performance su Jetson Thor

**Con GPU (CUDA 13.0) + `./run_bacco_gpu.sh`**:
- 4K video → ~40 FPS (benchmark, senza display)
- Con rendering completo (ellissi + filtro sfondo): ~18-20 FPS ✅
- Orizzontale 1280x720: ~25-30 FPS ✅

**Senza GPU (CPU only)**:
- ~3-5 FPS ❌ → Esegui `bash setup_gpu.sh` per abilitare GPU!

**Bottleneck**: Inferenza YOLO (~60%), tracking (~20%), rendering (~15%), CLAHE (~5%)

---

## 🔮 Sviluppi Futuri

- [ ] Export CSV con tracking data completo
- [ ] Heatmap zone frequentazione mele
- [ ] Stima dimensione mele (in cm con calibrazione)
- [ ] Multi-camera stitching
- [ ] TensorRT optimization per 60+ FPS

---

## 📄 Licenza

Progetto proprietario - Bacco Team

---

## 👥 Contatti

Per domande o supporto, contatta il team Bacco.

---

**Buon tracking! 🍎🚀**