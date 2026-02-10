#!/usr/bin/env python3
"""
Bacco - Apple Detection & Tracking System
==========================================
Sistema di rilevamento e tracciamento mele su video con:
- Object Detection via YOLOWorld
- Multi-object tracking via ByteTrack
- Maschere ellittiche
- Conteggio unico e persistente
- Visualizzazione real-time

Version: 2.0 - GPU Optimized + Class Merging + CLAHE

Author: Bacco Team
"""

import os
import sys
from pathlib import Path
from collections import defaultdict, deque
from typing import Tuple, List, Dict, Optional
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import supervision as sv


# =============================================================================
# CONFIGURAZIONE GLOBALE
# =============================================================================

class Config:
    """Configurazione centralizzata del sistema"""
    
    # === PERCORSI ===
    MODELS_DIR = Path("./models")
    YOLO_MODEL_NAME = "yolov8s-worldv2.pt"
    
    # === DETECTION (HIGH RECALL + TEMPORAL FILTER) ===
    YOLO_CONFIDENCE = 0.05    # Basso per catturare mele parziali/dietro vetro
    YOLO_PROMPTS = [
        "apple",                     # Base detection
        "red apple",                 # Colore specifico
        "apple on tree",             # Contesto frutteto
        "round red fruit",           # Forma + colore
        "ripe apple on branch",      # Mela matura su ramo
        "small red sphere in leaves", # Mele piccole tra foglie
        "apple partially hidden",    # Mele parzialmente occluse
        "red round fruit hanging",   # Mele pendenti
        "red object on a tree", 
        "red object under a leaf"
    ]
    NMS_THRESHOLD = 0.3       # Aggressivo per eliminare duplicati
    YOLO_IMGSZ = 1280         # Risoluzione interna YOLO (più alto = vede meglio piccoli)
    
    # === ADAPTIVE PARAMETERS ===
    ADAPTIVE_PARAMS = False   # Disabled for stable parameters
    MIN_CONFIDENCE = 0.01
    MAX_CONFIDENCE = 0.10
    RECALIBRATE_INTERVAL = 30
    
    # === TRACKING (CONSERVATIVE) ===
    TRACK_THRESH = 0.15       # Soglia attivazione bilanciata
    TRACK_BUFFER = 60         # Buffer 2s @ 30fps
    MATCH_THRESH = 0.8        # IoU matching rigoroso per ID stabili
    MIN_CONSECUTIVE = 3       # Track confermato dopo 3 frame consecutivi
    
    # === SMOOTHING (AGGRESSIVE) ===
    SMOOTH_WINDOW = 20        # Smoothing forte per bounding box stabili
    
    # === TEMPORAL FILTER ===
    MIN_CONSECUTIVE_FRAMES = 2   # 2 frame consecutivi = stabile (più reattivo)
    TEMPORAL_MATCH_DISTANCE = 60 # Distanza leggermente maggiore per matching
    
    # === DETECTION CARRY-FORWARD ===
    CARRY_FRAMES = 5             # Porta avanti detection mancanti per N frame
    CARRY_DECAY = 0.9            # Decadimento confidence per frame portato avanti
    
    # === ENHANCEMENT (CONDITIONAL) ===
    USE_CLAHE = True          # Abilita CLAHE per mele in ombra/parziali
    CLAHE_CLIP_LIMIT = 1.5    # Ridotto per limitare amplificazione noise
    CLAHE_TILE_SIZE = (8, 8)  # Dimensione tile CLAHE
    
    # === RESIZE (auto-scaling based on input) ===
    RESIZE_STRATEGY = "auto"  # 'auto', 'adaptive', 'native', 'fixed'
    TARGET_SIZE = 1920        # Aumentato per migliore detection (Thor ha 131GB VRAM)
    MIN_SIZE = 640            # Dimensione minima processing
    MAX_SIZE = 2560           # Dimensione massima processing
    
    # === VISUALIZZAZIONE (COLORI FISSI) ===
    BBOX_COLOR = (0, 0, 255)         # Rosso BGR per bounding box
    ELLIPSE_COLOR = (0, 0, 255)      # Rosso BGR per ellisse (fill)
    TRAJECTORY_COLOR = (0, 0, 139)   # Bordeaux (dark red) per traiettoria
    MAX_TRAJECTORY_POINTS = 30       # Punti massimi per traiettoria
    ELLIPSE_ALPHA = 0.6              # Trasparenza ellissi
    BBOX_THICKNESS = 2               # Spessore bounding box
    TEXT_SCALE = 0.6                 # Scala testo
    LINE_THICKNESS = 2               # Spessore linee traiettoria
    SHOW_TRAJECTORY = False          # Mostra linee traiettoria (disabilitato)
    
    # === BACKGROUND FILTER ===
    BG_FILTER_COLOR = (255, 0, 0)    # Blu BGR per filtro sfondo
    BG_FILTER_ALPHA = 0.4            # Trasparenza filtro sfondo
    
    # === PERFORMANCE ===
    TARGET_FPS = 20           # FPS target minimo
    DISPLAY_WIDTH = 1280      # Larghezza display window (per UI)


# =============================================================================
# MODEL MANAGER - Gestione download e caricamento modelli
# =============================================================================

class ModelManager:
    """Gestisce il download e caricamento automatico dei modelli"""
    
    def __init__(self, models_dir: Path = Config.MODELS_DIR):
        self.models_dir = models_dir
        self.models_dir.mkdir(exist_ok=True)
        
        # Crea file .gitkeep per mantenere la struttura
        gitkeep = self.models_dir / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.touch()
    
    def load_yolo_world(self, model_name: str = Config.YOLO_MODEL_NAME) -> YOLO:
        """
        Carica YOLOWorld, scaricandolo automaticamente se necessario
        
        Args:
            model_name: Nome del modello (es. 'yolov8s-worldv2.pt')
            
        Returns:
            Modello YOLO caricato
        """
        model_path = self.models_dir / model_name
        
        if not model_path.exists():
            print(f"[MODEL] 🔽 {model_name} non trovato. Download in corso...")
            try:
                # YOLO scarica automaticamente dalla repo ultralytics
                model = YOLO(model_name)
                
                # Copia dalla cache ultralytics a ./models/
                cache_path = Path.home() / ".cache" / "ultralytics" / model_name
                if cache_path.exists():
                    import shutil
                    shutil.copy(cache_path, model_path)
                    print(f"[MODEL] ✅ Salvato in {model_path}")
                else:
                    print(f"[MODEL] ⚠️  Modello scaricato ma non trovato in cache")
                    
            except Exception as e:
                print(f"[MODEL] ❌ Errore durante il download: {e}")
                sys.exit(1)
        else:
            print(f"[MODEL] ✅ Caricamento {model_name} da {model_path}")
        
        # Carica il modello
        model = YOLO(str(model_path))
        
        # Imposta i prompt per YOLOWorld
        model.set_classes(Config.YOLO_PROMPTS)
        
        return model


# =============================================================================
# IMAGE ENHANCEMENT - Preprocessing per mele parziali/in ombra
# =============================================================================

class ImageEnhancer:
    """Preprocessing per migliorare detection di mele parzialmente visibili"""
    
    @staticmethod
    def apply_clahe(frame: np.ndarray) -> np.ndarray:
        """
        Applica CLAHE (Contrast Limited Adaptive Histogram Equalization)
        Migliora contrasto locale per mele in ombra o dietro foglie
        
        Args:
            frame: Frame BGR
            
        Returns:
            Frame con contrasto migliorato
        """
        # Converti in LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Applica CLAHE solo sul canale L (luminanza)
        clahe = cv2.createCLAHE(
            clipLimit=Config.CLAHE_CLIP_LIMIT,
            tileGridSize=Config.CLAHE_TILE_SIZE
        )
        l_clahe = clahe.apply(l)
        
        # Ricomponi e converti in BGR
        lab_clahe = cv2.merge([l_clahe, a, b])
        frame_enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        
        return frame_enhanced


# =============================================================================
# RESIZE HANDLER - Gestione intelligente del ridimensionamento
# =============================================================================

class ResizeHandler:
    """Gestisce il ridimensionamento adattivo dei frame"""
    
    @staticmethod
    def adaptive_resize(
        frame: np.ndarray,
        target_size: int = Config.TARGET_SIZE,
        min_size: int = Config.MIN_SIZE
    ) -> Tuple[np.ndarray, float]:
        """
        Ridimensiona il frame mantenendo aspect ratio
        
        Strategia:
        - Frame > target_size → resize per performance
        - min_size <= Frame <= target_size → mantieni originale
        - Frame < min_size → upscale (raro)
        
        Args:
            frame: Frame input (HxWxC)
            target_size: Dimensione massima lato lungo
            min_size: Dimensione minima sotto cui non ridimensiona
            
        Returns:
            (frame_resized, scale_factor)
        """
        h, w = frame.shape[:2]
        max_dim = max(h, w)
        
        # Caso 1: Frame troppo grande → resize per performance
        if max_dim > target_size:
            scale = target_size / max_dim
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Assicura dimensioni pari (richiesto da alcuni codec)
            new_w = new_w - (new_w % 2)
            new_h = new_h - (new_h % 2)
            
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            return resized, scale
        
        # Caso 2: Frame nel range ottimale → mantieni originale
        elif max_dim >= min_size:
            return frame, 1.0
        
        # Caso 3: Frame troppo piccolo → upscale
        else:
            scale = min_size / max_dim
            new_w = int(w * scale)
            new_h = int(h * scale)
            new_w = new_w - (new_w % 2)
            new_h = new_h - (new_h % 2)
            
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            return resized, scale
    
    @staticmethod
    def native_resize(frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """Mantiene risoluzione nativa (no resize)"""
        return frame, 1.0
    
    @staticmethod
    def fixed_resize(frame: np.ndarray, size: int = Config.TARGET_SIZE) -> Tuple[np.ndarray, float]:
        """Resize fisso al lato lungo specificato"""
        h, w = frame.shape[:2]
        max_dim = max(h, w)
        scale = size / max_dim
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        new_w = new_w - (new_w % 2)
        new_h = new_h - (new_h % 2)
        
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return resized, scale
    
    @staticmethod
    def auto_resize(frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Auto-resize intelligente basato su dimensioni input.
        
        Strategia:
        - Video piccoli (< 640px): upscale a 640 per migliore detection
        - Video medi (640-1280): mantiene native per massima precisione
        - Video grandi (1280-1920): resize a 1280 per bilanciare speed/accuracy
        - Video 4K+ (> 1920): resize a 1280 per performance
        
        Returns:
            (frame_resized, scale_factor)
        """
        h, w = frame.shape[:2]
        max_dim = max(h, w)
        
        # Video piccolo → upscale per migliore detection
        if max_dim < Config.MIN_SIZE:
            target = Config.MIN_SIZE
            scale = target / max_dim
        # Video medio → mantiene native per precisione
        elif max_dim <= Config.TARGET_SIZE:
            return frame, 1.0
        # Video grande/4K → resize per performance
        else:
            target = Config.TARGET_SIZE
            scale = target / max_dim
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        new_w = new_w - (new_w % 2)
        new_h = new_h - (new_h % 2)
        
        interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(frame, (new_w, new_h), interpolation=interpolation)
        return resized, scale


# =============================================================================
# ID REGISTRY - Gestione ID unici persistenti
# =============================================================================

class IDRegistry:
    """
    Registry persistente per tracciare ID unici nel tempo.
    Include de-duplicazione spaziale: se un nuovo track appare dove
    un track precedente è appena scomparso, viene riconosciuto come
    lo stesso apple (non conta come nuovo).
    """
    
    def __init__(self, lost_threshold: int = Config.TRACK_BUFFER, dedup_distance: float = 80.0):
        self.unique_ids = set()              # Tutti gli ID mai visti
        self.active_tracks: Dict[int, int] = {}  # {track_id: last_seen_frame}
        self.track_positions: Dict[int, np.ndarray] = {}  # {track_id: last_bbox_center}
        self.lost_threshold = lost_threshold
        self.dedup_distance = dedup_distance
        self.total_new_apples = 0            # Contatore progressivo
        # Posizioni recenti di track persi per de-duplicazione
        # {track_id: {'center': (x,y), 'frame_lost': int, 'bbox': array}}
        self.recently_lost: Dict[int, Dict] = {}
        self.lost_memory_frames = 90         # Ricorda posizioni per 3 secondi
        self.dedup_map: Dict[int, int] = {}  # {new_id: original_id} mapping
    
    def update(self, track_ids: List[int], current_frame: int, 
               detections=None) -> int:
        """
        Aggiorna il registry con i track del frame corrente.
        Usa de-duplicazione spaziale per evitare overcounting.
        
        Args:
            track_ids: Lista di ID dei track attivi
            current_frame: Numero frame corrente
            detections: Detections con xyxy per posizione spaziale
            
        Returns:
            Numero di nuove mele rilevate in questo frame
        """
        new_apples = 0
        
        for i, tid in enumerate(track_ids):
            # Aggiorna posizione corrente
            if detections is not None and i < len(detections):
                bbox = detections.xyxy[i]
                center = np.array([(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2])
                self.track_positions[tid] = center
            
            # Nuovo ID mai visto prima
            if tid not in self.unique_ids:
                # Controlla se è un re-ID di un track perso recentemente
                matched_lost_id = self._find_spatial_match(tid, current_frame)
                
                if matched_lost_id is not None:
                    # È lo stesso apple! Non contare come nuovo
                    self.unique_ids.add(tid)
                    self.dedup_map[tid] = matched_lost_id
                    # Rimuovi dalla lista persi
                    if matched_lost_id in self.recently_lost:
                        del self.recently_lost[matched_lost_id]
                else:
                    # Genuinamente nuovo
                    self.unique_ids.add(tid)
                    new_apples += 1
                    self.total_new_apples += 1
                    print(f"[APPLE] 🍎 Nuova mela rilevata! ID {tid} | Totale unico: {self.total_new_apples}")
            
            # Aggiorna ultimo frame visto
            self.active_tracks[tid] = current_frame
        
        # Cleanup track persi da troppo tempo
        self._cleanup_lost_tracks(current_frame)
        
        return new_apples
    
    def _find_spatial_match(self, new_tid: int, current_frame: int) -> int:
        """
        Cerca se il nuovo track corrisponde spazialmente a un track perso.
        
        Returns:
            ID del track perso corrispondente, o None
        """
        if new_tid not in self.track_positions:
            return None
        
        new_center = self.track_positions[new_tid]
        best_match = None
        best_dist = float('inf')
        
        for lost_tid, lost_info in self.recently_lost.items():
            # Controlla che non sia troppo vecchio
            frames_since_lost = current_frame - lost_info['frame_lost']
            if frames_since_lost > self.lost_memory_frames:
                continue
            
            lost_center = lost_info['center']
            dist = np.sqrt(np.sum((new_center - lost_center) ** 2))
            
            if dist < self.dedup_distance and dist < best_dist:
                best_dist = dist
                best_match = lost_tid
        
        if best_match is not None:
            print(f"[DEDUP] 🔄 ID {new_tid} riconosciuto come ex-ID {best_match} (dist={best_dist:.0f}px)")
        
        return best_match
    
    def _cleanup_lost_tracks(self, current_frame: int):
        """Rimuove track persi da troppo tempo dal registry attivo"""
        lost_ids = []
        
        for tid, last_seen in self.active_tracks.items():
            frames_lost = current_frame - last_seen
            if frames_lost > self.lost_threshold:
                lost_ids.append(tid)
        
        for tid in lost_ids:
            # Salva posizione per de-duplicazione futura
            if tid in self.track_positions:
                self.recently_lost[tid] = {
                    'center': self.track_positions[tid].copy(),
                    'frame_lost': current_frame
                }
            del self.active_tracks[tid]
            print(f"[TRACK] 👻 ID {tid} perso da {self.lost_threshold} frame (rimosso da attivi)")
        
        # Cleanup memorie troppo vecchie
        old_lost = [tid for tid, info in self.recently_lost.items()
                    if current_frame - info['frame_lost'] > self.lost_memory_frames]
        for tid in old_lost:
            del self.recently_lost[tid]
    
    def get_stats(self) -> Dict[str, int]:
        """Ritorna statistiche correnti"""
        return {
            "total_unique": self.total_new_apples,  # Usa contatore de-duplicato
            "currently_active": len(self.active_tracks),
            "total_new": self.total_new_apples
        }


# =============================================================================
# TRAJECTORY SMOOTHER - Smoothing temporale coordinate
# =============================================================================

class TrajectorySmooth:
    """
    Applica smoothing temporale alle coordinate bbox per ridurre jitter
    Usa media mobile su finestra configurabile
    """
    
    def __init__(self, window_size: int = Config.SMOOTH_WINDOW):
        self.window_size = window_size
        # History per ogni track: {track_id: deque([bbox1, bbox2, ...])}
        self.history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=window_size))
    
    def smooth_bbox(self, track_id: int, bbox: np.ndarray) -> np.ndarray:
        """
        Applica smoothing alla bounding box
        
        Args:
            track_id: ID del track
            bbox: Array [x1, y1, x2, y2]
            
        Returns:
            Bbox smoothed (media mobile)
        """
        # Aggiungi alla history
        self.history[track_id].append(bbox.copy())
        
        # Se non abbastanza dati, ritorna originale
        if len(self.history[track_id]) < 2:
            return bbox
        
        # Calcola media mobile
        smoothed = np.mean(self.history[track_id], axis=0)
        return smoothed.astype(np.float32)
    
    def get_center_history(self, track_id: int, max_points: int = Config.MAX_TRAJECTORY_POINTS) -> List[Tuple[int, int]]:
        """
        Ottiene lo storico dei centri per disegnare traiettorie
        
        Args:
            track_id: ID del track
            max_points: Numero massimo di punti da ritornare
            
        Returns:
            Lista di punti (x, y) della traiettoria
        """
        if track_id not in self.history:
            return []
        
        centers = []
        for bbox in list(self.history[track_id])[-max_points:]:
            cx = int((bbox[0] + bbox[2]) / 2)
            cy = int((bbox[1] + bbox[3]) / 2)
            centers.append((cx, cy))
        
        return centers


# =============================================================================
# TEMPORAL FILTER - Filtra detection instabili
# =============================================================================

class TemporalFilter:
    """
    Filtra detection che non appaiono per N frame consecutivi
    Riduce flickering da detection instabili
    """
    
    def __init__(
        self, 
        min_consecutive_frames: int = Config.MIN_CONSECUTIVE_FRAMES,
        max_distance: float = Config.TEMPORAL_MATCH_DISTANCE
    ):
        self.min_frames = min_consecutive_frames
        self.max_distance = max_distance
        self.detection_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=min_consecutive_frames)
        )
        self.frame_count = 0
        # Cleanup periodico per detection vecchie
        self.last_cleanup_frame = 0
        self.cleanup_interval = 100
    
    def filter(self, detections: sv.Detections) -> sv.Detections:
        """
        Ritorna solo detection stabili (presenti per >= min_frames consecutivi)
        
        Args:
            detections: Detection raw dal modello
            
        Returns:
            Detection filtrate (solo quelle stabili)
        """
        self.frame_count += 1
        
        if len(detections) == 0:
            return detections
        
        stable_indices = []
        
        for i in range(len(detections)):
            bbox = detections.xyxy[i]
            bbox_center = (
                float((bbox[0] + bbox[2]) / 2), 
                float((bbox[1] + bbox[3]) / 2)
            )
            
            # Trova detection simile in history
            detection_id = self._find_matching_id(bbox_center)
            
            if detection_id is None:
                # Nuova detection → crea nuovo ID basato su posizione
                detection_id = f"{bbox_center[0]:.0f}_{bbox_center[1]:.0f}_{self.frame_count}"
            
            # Aggiungi a history
            self.detection_history[detection_id].append({
                'center': bbox_center,
                'frame': self.frame_count
            })
            
            # Mantieni solo se stabile (presente per min_frames)
            if len(self.detection_history[detection_id]) >= self.min_frames:
                stable_indices.append(i)
        
        # Cleanup periodico
        if self.frame_count - self.last_cleanup_frame > self.cleanup_interval:
            self._cleanup_old_detections()
            self.last_cleanup_frame = self.frame_count
        
        # Filtra detections mantenendo solo indici stabili
        if stable_indices:
            # Crea nuovo oggetto Detections con solo detection stabili
            return sv.Detections(
                xyxy=detections.xyxy[stable_indices],
                confidence=detections.confidence[stable_indices] if detections.confidence is not None else None,
                class_id=detections.class_id[stable_indices] if detections.class_id is not None else None,
                tracker_id=detections.tracker_id[stable_indices] if detections.tracker_id is not None else None
            )
        else:
            return sv.Detections.empty()
    
    def _find_matching_id(self, center: Tuple[float, float]) -> Optional[str]:
        """
        Trova detection ID esistente entro max_distance
        
        Args:
            center: Centro bbox (x, y)
            
        Returns:
            ID detection se trovata, None altrimenti
        """
        for det_id, history in self.detection_history.items():
            if history:
                last_entry = history[-1]
                last_center = last_entry['center']
                last_frame = last_entry['frame']
                
                # Ignora detection troppo vecchie (> 10 frame fa)
                if self.frame_count - last_frame > 10:
                    continue
                
                # Calcola distanza euclidea
                dist = np.sqrt(
                    (center[0] - last_center[0])**2 + 
                    (center[1] - last_center[1])**2
                )
                
                if dist < self.max_distance:
                    return det_id
        
        return None
    
    def _cleanup_old_detections(self):
        """Rimuove detection non viste da troppo tempo"""
        ids_to_remove = []
        
        for det_id, history in self.detection_history.items():
            if history:
                last_frame = history[-1]['frame']
                if self.frame_count - last_frame > 50:  # Non viste da 50 frame
                    ids_to_remove.append(det_id)
        
        for det_id in ids_to_remove:
            del self.detection_history[det_id]



# =============================================================================
# DETECTION CARRY-FORWARD - Persistenza temporale detection
# =============================================================================

class DetectionCarryForward:
    """
    Se un apple trackato scompare per breve tempo (occlusione momentanea),
    porta avanti la sua ultima posizione nota per alcuni frame.
    Riduce drammaticamente la perdita di track e il churn degli ID.
    """
    
    def __init__(
        self,
        carry_frames: int = Config.CARRY_FRAMES,
        carry_decay: float = Config.CARRY_DECAY,
        match_distance: float = Config.TEMPORAL_MATCH_DISTANCE
    ):
        self.carry_frames = carry_frames
        self.carry_decay = carry_decay
        self.match_distance = match_distance
        # {track_id: {'bbox': array, 'confidence': float, 'frames_missing': int}}
        self.tracked_positions: Dict[int, Dict] = {}
    
    def update_and_carry(
        self,
        detections: sv.Detections,
        tracked_detections: sv.Detections
    ) -> sv.Detections:
        """
        Aggiorna posizioni note e porta avanti detection mancanti.
        
        Args:
            detections: Detection raw (pre-tracking, post-temporal-filter)
            tracked_detections: Detection con tracker_id da ByteTrack
            
        Returns:
            Detection arricchite con carry-forward
        """
        current_track_ids = set()
        
        # Aggiorna posizioni note dai track attivi
        if tracked_detections.tracker_id is not None:
            for i in range(len(tracked_detections)):
                tid = int(tracked_detections.tracker_id[i])
                current_track_ids.add(tid)
                self.tracked_positions[tid] = {
                    'bbox': tracked_detections.xyxy[i].copy(),
                    'confidence': float(tracked_detections.confidence[i]) if tracked_detections.confidence is not None else 0.5,
                    'frames_missing': 0
                }
        
        # Trova track persi e genera carry-forward
        carry_bboxes = []
        carry_confidences = []
        carry_track_ids = []
        lost_ids = []
        
        for tid, info in self.tracked_positions.items():
            if tid not in current_track_ids:
                info['frames_missing'] += 1
                
                if info['frames_missing'] <= self.carry_frames:
                    # Porta avanti con confidence decadente
                    carried_conf = info['confidence'] * (self.carry_decay ** info['frames_missing'])
                    
                    # Verifica che non sia troppo vicino a una detection esistente
                    bbox_center = (
                        (info['bbox'][0] + info['bbox'][2]) / 2,
                        (info['bbox'][1] + info['bbox'][3]) / 2
                    )
                    
                    # Controlla sovrapposizione con detection esistenti
                    is_duplicate = False
                    if len(detections) > 0:
                        for j in range(len(detections)):
                            det_center = (
                                (detections.xyxy[j][0] + detections.xyxy[j][2]) / 2,
                                (detections.xyxy[j][1] + detections.xyxy[j][3]) / 2
                            )
                            dist = np.sqrt(
                                (bbox_center[0] - det_center[0])**2 +
                                (bbox_center[1] - det_center[1])**2
                            )
                            if dist < self.match_distance:
                                is_duplicate = True
                                break
                    
                    if not is_duplicate:
                        carry_bboxes.append(info['bbox'])
                        carry_confidences.append(carried_conf)
                        carry_track_ids.append(tid)
                else:
                    lost_ids.append(tid)
        
        # Rimuovi track persi definitivamente
        for tid in lost_ids:
            del self.tracked_positions[tid]
        
        # Se ci sono detection da portare avanti, uniscile
        if carry_bboxes and len(detections) > 0:
            merged_xyxy = np.concatenate([
                detections.xyxy,
                np.array(carry_bboxes)
            ], axis=0)
            
            merged_conf = np.concatenate([
                detections.confidence if detections.confidence is not None else np.ones(len(detections)),
                np.array(carry_confidences)
            ])
            
            merged_class_id = np.zeros(len(merged_xyxy), dtype=int)
            
            return sv.Detections(
                xyxy=merged_xyxy,
                confidence=merged_conf,
                class_id=merged_class_id
            )
        elif carry_bboxes and len(detections) == 0:
            return sv.Detections(
                xyxy=np.array(carry_bboxes),
                confidence=np.array(carry_confidences),
                class_id=np.zeros(len(carry_bboxes), dtype=int)
            )
        
        return detections


# =============================================================================
# VISUALIZATION MANAGER - Gestione rendering con COLORI FISSI
# =============================================================================

class VisualizationManager:
    """Gestisce il rendering di bbox, ellissi, traiettorie e UI"""
    
    def __init__(self):
        # Non serve più color palette, usiamo colori fissi
        pass
    
    @staticmethod
    def get_bbox_color() -> Tuple[int, int, int]:
        """Ritorna colore fisso per bbox (rosso)"""
        return Config.BBOX_COLOR
    
    @staticmethod
    def get_ellipse_color() -> Tuple[int, int, int]:
        """Ritorna colore fisso per ellisse (rosso)"""
        return Config.ELLIPSE_COLOR
    
    @staticmethod
    def get_trajectory_color() -> Tuple[int, int, int]:
        """Ritorna colore fisso per traiettoria (bordeaux)"""
        return Config.TRAJECTORY_COLOR
    
    def draw_background_filter(
        self,
        frame: np.ndarray,
        color: Tuple[int, int, int] = None,
        alpha: float = None
    ) -> np.ndarray:
        """
        Applica filtro colore semitrasparente su tutto il frame
        
        Args:
            frame: Frame su cui applicare il filtro
            color: Colore BGR del filtro (default: BG_FILTER_COLOR)
            alpha: Trasparenza filtro (default: BG_FILTER_ALPHA)
        
        Returns:
            Frame con filtro applicato
        """
        if color is None:
            color = Config.BG_FILTER_COLOR
        if alpha is None:
            alpha = Config.BG_FILTER_ALPHA
        
        # Crea overlay colorato
        overlay = np.full(frame.shape, color, dtype=np.uint8)
        
        # Blend con alpha
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        return frame
    
    def draw_bbox(
        self,
        frame: np.ndarray,
        bbox: np.ndarray,
        track_id: int,
        confidence: float
    ) -> np.ndarray:
        """Disegna bounding box con colore fisso rosso"""
        x1, y1, x2, y2 = map(int, bbox)
        color = self.get_bbox_color()
        
        # Disegna rettangolo
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, Config.BBOX_THICKNESS)
        
        # Label con ID e confidence
        label = f"ID:{track_id} ({confidence:.2f})"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, Config.TEXT_SCALE, 2)[0]
        
        # Background per label
        cv2.rectangle(
            frame,
            (x1, y1 - label_size[1] - 10),
            (x1 + label_size[0] + 5, y1),
            color,
            -1
        )
        
        # Testo label
        cv2.putText(
            frame,
            label,
            (x1 + 2, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            Config.TEXT_SCALE,
            (255, 255, 255),
            2
        )
        
        return frame
    
    def draw_ellipse(
        self,
        frame: np.ndarray,
        bbox: np.ndarray,
        track_id: int,
        alpha: float = Config.ELLIPSE_ALPHA
    ) -> np.ndarray:
        """
        Disegna ellisse semitrasparente con colore fisso rosso
        
        Args:
            frame: Frame su cui disegnare
            bbox: Bounding box [x1, y1, x2, y2]
            track_id: ID del track (non più usato per colore)
            alpha: Trasparenza (0-1)
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # Centro e assi ellisse
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        axis_x = (x2 - x1) // 2
        axis_y = (y2 - y1) // 2
        
        # Colore fisso
        color = self.get_ellipse_color()
        
        # Crea overlay per trasparenza
        overlay = frame.copy()
        cv2.ellipse(
            overlay,
            (center_x, center_y),
            (axis_x, axis_y),
            0,  # angolo
            0,  # start angle
            360,  # end angle
            color,
            -1  # filled
        )
        
        # Blend con alpha
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Bordo ellisse
        cv2.ellipse(
            frame,
            (center_x, center_y),
            (axis_x, axis_y),
            0, 0, 360,
            color,
            2
        )
        
        return frame
    
    def draw_trajectory(
        self,
        frame: np.ndarray,
        centers: List[Tuple[int, int]],
        track_id: int
    ) -> np.ndarray:
        """Disegna linea di traiettoria con colore fisso bordeaux"""
        if len(centers) < 2:
            return frame
        
        color = self.get_trajectory_color()
        
        # Disegna linea connessa tra i punti
        for i in range(len(centers) - 1):
            pt1 = centers[i]
            pt2 = centers[i + 1]
            cv2.line(frame, pt1, pt2, color, Config.LINE_THICKNESS)
        
        # Disegna punto finale più grande
        if centers:
            cv2.circle(frame, centers[-1], 4, color, -1)
        
        return frame
    
    def draw_hud(
        self,
        frame: np.ndarray,
        frame_number: int,
        current_count: int,
        total_unique: int,
        fps: float
    ) -> np.ndarray:
        """
        Disegna HUD con statistiche in alto a sinistra
        
        Args:
            frame: Frame su cui disegnare
            frame_number: Numero frame corrente
            current_count: Mele nel frame corrente
            total_unique: Totale mele uniche
            fps: FPS corrente
        """
        # Background HUD
        hud_height = 120
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (400, hud_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Testi
        texts = [
            f"Frame: {frame_number}",
            f"Mele correnti: {current_count}",
            f"Totale uniche: {total_unique}",
            f"FPS: {fps:.1f}"
        ]
        
        y_offset = 25
        for text in texts:
            cv2.putText(
                frame,
                text,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            y_offset += 30
        
        return frame


# =============================================================================
# APPLE DETECTION SYSTEM - Sistema principale
# =============================================================================

class AppleDetectionSystem:
    """
    Sistema completo di detection e tracking mele
    Integra tutti i componenti per processing video real-time
    """
    
    def __init__(
        self,
        resize_strategy: str = Config.RESIZE_STRATEGY,
        device: str = None
    ):
        print("[INIT] 🚀 Inizializzazione Bacco Apple Detection System...")
        
        # Device setup con fallback intelligente
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        
        # Diagnostica GPU dettagliata
        if self.device == "cuda":
            print(f"[INIT] 💻 Device: {self.device}")
            print(f"[INIT] 🎮 GPU: {torch.cuda.get_device_name(0)}")
            print(f"[INIT] 📊 Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            # Ottimizzazioni GPU
            torch.backends.cudnn.benchmark = True
            print("[INIT] ⚡ CuDNN benchmark abilitato")
        else:
            print(f"[INIT] 💻 Device: {self.device}")
            print("[INIT] ⚠️  GPU non disponibile, uso CPU (più lento)")
            print("[INIT] 💡 Per abilitare GPU, esegui: bash setup_gpu.sh")
        
        # Carica modelli
        model_manager = ModelManager()
        self.model = model_manager.load_yolo_world()
        self.model.to(self.device)
        
        # Setup ByteTrack tramite supervision
        self.tracker = sv.ByteTrack(
            track_activation_threshold=Config.TRACK_THRESH,
            lost_track_buffer=Config.TRACK_BUFFER,
            minimum_matching_threshold=Config.MATCH_THRESH,
            frame_rate=30,
            minimum_consecutive_frames=Config.MIN_CONSECUTIVE
        )
        
        # Componenti
        self.id_registry = IDRegistry()
        self.smoother = TrajectorySmooth()
        self.temporal_filter = TemporalFilter()
        self.carry_forward = DetectionCarryForward()
        self.viz = VisualizationManager()
        self.resize_handler = ResizeHandler()
        self.enhancer = ImageEnhancer()
        self.resize_strategy = resize_strategy
        
        # Statistiche
        self.frame_count = 0
        self.fps = 0.0
        
        print("[INIT] ✅ Sistema pronto!")
    
    def detect_apples(self, frame: np.ndarray) -> sv.Detections:
        """
        Esegue detection delle mele su un frame con CLASS MERGING + NMS.
        Usa imgsz alto per catturare oggetti piccoli senza tiling.
        
        Args:
            frame: Frame input (BGR)
            
        Returns:
            Detections in formato supervision (senza duplicati)
        """
        # Inferenza full-frame con mixed precision e imgsz alto
        with torch.amp.autocast('cuda', enabled=(self.device == "cuda")):
            results = self.model.predict(
                frame,
                conf=Config.YOLO_CONFIDENCE,
                imgsz=Config.YOLO_IMGSZ,
                verbose=False,
                device=self.device
            )
        
        # Converte in formato supervision
        detections = sv.Detections.from_ultralytics(results[0])
        
        # *** CLASS MERGING + NMS per eliminare double counting ***
        if len(detections) > 0:
            if detections.class_id is not None:
                detections.class_id[:] = 0
            detections = detections.with_nms(threshold=Config.NMS_THRESHOLD)
        
        return detections
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Processa un singolo frame: enhancement → detection → tracking → visualization
        
        Args:
            frame: Frame input (BGR)
            
        Returns:
            (frame_annotato, statistiche)
        """
        start_time = time.time()
        
        # 1. Resize adattivo
        if self.resize_strategy == "auto":
            frame_resized, scale = self.resize_handler.auto_resize(frame)
        elif self.resize_strategy == "adaptive":
            frame_resized, scale = self.resize_handler.adaptive_resize(frame)
        elif self.resize_strategy == "native":
            frame_resized, scale = self.resize_handler.native_resize(frame)
        else:  # fixed
            frame_resized, scale = self.resize_handler.fixed_resize(frame)
        
        # 2. Enhancement con CLAHE (se abilitato)
        if Config.USE_CLAHE:
            frame_enhanced = self.enhancer.apply_clahe(frame_resized)
        else:
            frame_enhanced = frame_resized
        
        # 3. Detection con class merging + NMS
        detections = self.detect_apples(frame_enhanced)
        
        # 3.5. FILTRO TEMPORALE
        detections_before = len(detections)
        detections = self.temporal_filter.filter(detections)
        detections_after = len(detections)
        
        # Log ogni 30 frame
        if self.frame_count % 30 == 0 and detections_before != detections_after:
            print(f"[FILTER] Frame {self.frame_count}: {detections_before} det → {detections_after} stabili")
        # 3.5b. CARRY-FORWARD: inietta detection portate avanti nel flusso
        detections = self.carry_forward.update_and_carry(detections, detections)
        
        # 4. Tracking con ByteTrack
        tracked_detections = self.tracker.update_with_detections(detections)
        
        # 4.5. Aggiorna carry-forward con posizioni tracciati correnti
        self.carry_forward.update_and_carry(detections, tracked_detections)
        detections = tracked_detections
        
        # 5. Aggiorna ID Registry
        track_ids = detections.tracker_id.tolist() if detections.tracker_id is not None else []
        self.id_registry.update(track_ids, self.frame_count, detections=detections)
        
        # 6. Annotazione frame (usa frame_resized, non enhanced, per visualizzazione)
        frame_annotated = frame_resized.copy()
        
        # 6.1. Applica filtro sfondo blu
        frame_annotated = self.viz.draw_background_filter(frame_annotated)
        
        if len(detections) > 0:
            # Pre-calcola tutte le bbox smoothed
            smoothed_data = []
            for i in range(len(detections)):
                bbox = detections.xyxy[i]
                track_id = int(detections.tracker_id[i]) if detections.tracker_id is not None else -1
                confidence = float(detections.confidence[i]) if detections.confidence is not None else 0.0
                
                if track_id == -1:
                    continue
                
                bbox_smooth = self.smoother.smooth_bbox(track_id, bbox)
                smoothed_data.append((bbox_smooth, track_id, confidence))
            
            # 6.2. Disegna TUTTE le ellissi su un singolo overlay, poi blend una volta
            if smoothed_data:
                overlay = frame_annotated.copy()
                ellipse_color = self.viz.get_ellipse_color()
                for bbox_smooth, track_id, _ in smoothed_data:
                    x1, y1, x2, y2 = map(int, bbox_smooth)
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    axes = ((x2 - x1) // 2, (y2 - y1) // 2)
                    cv2.ellipse(overlay, center, axes, 0, 0, 360, ellipse_color, -1)
                
                cv2.addWeighted(overlay, Config.ELLIPSE_ALPHA, frame_annotated, 
                               1 - Config.ELLIPSE_ALPHA, 0, frame_annotated)
                
                # Bordi ellissi (non trasparenti)
                for bbox_smooth, track_id, _ in smoothed_data:
                    x1, y1, x2, y2 = map(int, bbox_smooth)
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    axes = ((x2 - x1) // 2, (y2 - y1) // 2)
                    cv2.ellipse(frame_annotated, center, axes, 0, 0, 360, ellipse_color, 2)
            
            # 6.3. Disegna bbox e traiettorie (opachi, sopra le ellissi)
            for bbox_smooth, track_id, confidence in smoothed_data:
                frame_annotated = self.viz.draw_bbox(frame_annotated, bbox_smooth, track_id, confidence)
                
                if Config.SHOW_TRAJECTORY:
                    centers = self.smoother.get_center_history(track_id)
                    frame_annotated = self.viz.draw_trajectory(frame_annotated, centers, track_id)
        
        # 7. HUD con statistiche
        stats = self.id_registry.get_stats()
        current_drawn = len(smoothed_data) if len(detections) > 0 else 0
        frame_annotated = self.viz.draw_hud(
            frame_annotated,
            self.frame_count,
            current_drawn,  # Mostra mele effettivamente disegnate
            stats["total_unique"],
            self.fps
        )
        
        # 8. Calcola FPS
        elapsed = time.time() - start_time
        self.fps = 1.0 / elapsed if elapsed > 0 else 0.0
        
        # 9. Incrementa contatore frame
        self.frame_count += 1
        
        # Statistiche
        statistics = {
            "frame": self.frame_count,
            "current_apples": stats["currently_active"],
            "total_unique": stats["total_unique"],
            "fps": self.fps,
            "scale": scale
        }
        
        return frame_annotated, statistics
    
    def process_video(self, video_path: str, display: bool = True):
        """
        Processa un video completo
        
        Args:
            video_path: Path al video MP4
            display: Se True, mostra finestra real-time
        """
        print(f"\n[VIDEO] 📹 Apertura video: {video_path}")
        
        # Apri video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] ❌ Impossibile aprire video: {video_path}")
            return
        
        # Info video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_video = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"[VIDEO] 📊 Risoluzione: {width}x{height}")
        print(f"[VIDEO] 🎬 Frame totali: {total_frames} @ {fps_video:.2f} FPS")
        print(f"[VIDEO] ⚙️  Resize strategy: {self.resize_strategy}")
        print(f"[VIDEO] 🎨 Enhancement CLAHE: {'ATTIVO' if Config.USE_CLAHE else 'DISATTIVO'}")
        print(f"[VIDEO] 🎯 Target FPS: {Config.TARGET_FPS}+")
        print("\n[VIDEO] ▶️  Avvio processing... (premi 'q' per uscire)\n")
        
        # *** FIX: Definisci start_time ***
        start_time = time.time()
        
        # Loop frame
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("\n[VIDEO] 🏁 Fine video raggiunta")
                    break
                
                # Processa frame
                frame_annotated, stats = self.process_frame(frame)
                
                # Display
                if display:
                    # Resize per display se troppo grande
                    display_frame = frame_annotated
                    if frame_annotated.shape[1] > Config.DISPLAY_WIDTH:
                        scale = Config.DISPLAY_WIDTH / frame_annotated.shape[1]
                        new_h = int(frame_annotated.shape[0] * scale)
                        display_frame = cv2.resize(frame_annotated, (Config.DISPLAY_WIDTH, new_h))
                    
                    cv2.imshow("Bacco - Apple Detection", display_frame)
                    
                    # Check tasto 'q' per uscire
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\n[VIDEO] ⏹️  Interruzione manuale")
                        break
                
                # Log periodico (ogni 30 frame)
                if self.frame_count % 30 == 0:
                    print(
                        f"[PROGRESS] Frame {stats['frame']:04d}/{total_frames} | "
                        f"Correnti: {stats['current_apples']:02d} | "
                        f"Uniche: {stats['total_unique']:02d} | "
                        f"FPS: {stats['fps']:05.1f}"
                    )
        
        except KeyboardInterrupt:
            print("\n[VIDEO] ⏹️  Interruzione da tastiera")
        
        finally:
            # Cleanup
            cap.release()
            if display:
                cv2.destroyAllWindows()
            
            # *** FIX: Usa start_time definito sopra ***
            elapsed_total = time.time() - start_time
            
            # Statistiche finali
            final_stats = self.id_registry.get_stats()
            print("\n" + "="*60)
            print("[SUMMARY] 📊 STATISTICHE FINALI")
            print("="*60)
            print(f"Frame processati:      {self.frame_count}")
            print(f"Mele uniche totali:    {final_stats['total_unique']}")
            print(f"FPS medio:             {self.frame_count / elapsed_total:.2f}")
            print(f"Tempo totale:          {elapsed_total:.1f}s")
            print("="*60 + "\n")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Entry point principale"""
    
    print("""
    ╔════════════════════════════════════════════════════╗
    ║                                                    ║
    ║        🍎 BACCO - Apple Detection System 🍎       ║
    ║                                                    ║
    ║  YOLOWorld + ByteTrack + Elliptical Masks         ║
    ║  Class Merging + CLAHE Enhancement                ║
    ║                                                    ║
    ╚════════════════════════════════════════════════════╝
    """)
    
    # Verifica CUDA con diagnostica dettagliata
    if torch.cuda.is_available():
        print(f"[CUDA] ✅ GPU disponibile: {torch.cuda.get_device_name(0)}")
        print(f"[CUDA] 📊 Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"[CUDA] 🔢 CUDA Version: {torch.version.cuda}")
    else:
        print("[CUDA] ⚠️  GPU non disponibile, uso CPU (più lento)")
        print("[CUDA] 💡 Per fix GPU, esegui: bash setup_gpu.sh")
        print("[CUDA] 🚀 Poi usa: ./run_bacco_gpu.sh video.mp4")
        print("[CUDA] 🔍 Per diagnostica: python check_gpu.py")
    
    # Path video (modifica con il tuo video)
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        print("\n[INPUT] 📁 Specifica il path del video MP4:")
        print("Uso: python main.py <path_to_video.mp4>")
        print("\nOppure trascina il file MP4 qui: ", end="")
        video_path = input().strip().strip('"').strip("'")
    
    # Verifica esistenza file
    if not os.path.exists(video_path):
        print(f"[ERROR] ❌ File non trovato: {video_path}")
        sys.exit(1)
    
    # Inizializza sistema
    system = AppleDetectionSystem(
        resize_strategy=Config.RESIZE_STRATEGY
    )
    
    # Processa video
    system.process_video(video_path, display=True)
    
    print("\n[EXIT] 👋 Bacco terminato. Arrivederci!\n")


if __name__ == "__main__":
    main()