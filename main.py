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
    
    # === DETECTION ===
    YOLO_CONFIDENCE = 0.03  # Threshold molto basso per catturare mele parziali (era 0.05)
    YOLO_PROMPTS = ["apple", "red apple", "green apple"]  # Multi-prompt per recall
    NMS_THRESHOLD = 0.4     # NMS aggressivo per rimuovere duplicati da multi-prompt
    
    # === TRACKING ===
    TRACK_THRESH = 0.4        # Confidence minima per iniziare un track
    TRACK_BUFFER = 60         # Frame di buffer prima di "uccidere" un ID
    MATCH_THRESH = 0.7        # IoU threshold per matching tracks
    
    # === SMOOTHING ===
    SMOOTH_WINDOW = 5         # Finestra per media mobile coordinate
    
    # === ENHANCEMENT (NUOVO) ===
    USE_CLAHE = True          # Abilita CLAHE per mele in ombra/parziali
    CLAHE_CLIP_LIMIT = 2.0    # Limite contrasto CLAHE
    CLAHE_TILE_SIZE = (8, 8)  # Dimensione tile CLAHE
    
    # === RESIZE ===
    RESIZE_STRATEGY = "adaptive"  # 'adaptive', 'native', 'fixed'
    TARGET_SIZE = 1280        # Dimensione massima lato lungo
    MIN_SIZE = 640            # Dimensione minima (sotto questa, non resize)
    
    # === VISUALIZZAZIONE (COLORI FISSI) ===
    BBOX_COLOR = (0, 0, 255)         # Rosso BGR per bounding box
    ELLIPSE_COLOR = (0, 0, 255)      # Rosso BGR per ellisse
    TRAJECTORY_COLOR = (0, 0, 139)   # Bordeaux (dark red) per traiettoria
    MAX_TRAJECTORY_POINTS = 30       # Punti massimi per traiettoria
    ELLIPSE_ALPHA = 0.3              # Trasparenza ellissi
    BBOX_THICKNESS = 2               # Spessore bounding box
    TEXT_SCALE = 0.6                 # Scala testo
    LINE_THICKNESS = 2               # Spessore linee traiettoria
    
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


# =============================================================================
# ID REGISTRY - Gestione ID unici persistenti
# =============================================================================

class IDRegistry:
    """
    Registry persistente per tracciare ID unici nel tempo
    Gestisce l'apparizione/scomparsa di track per conteggio accurato
    """
    
    def __init__(self, lost_threshold: int = Config.TRACK_BUFFER):
        self.unique_ids = set()              # Tutti gli ID mai visti
        self.active_tracks: Dict[int, int] = {}  # {track_id: last_seen_frame}
        self.lost_threshold = lost_threshold
        self.total_new_apples = 0            # Contatore progressivo
    
    def update(self, track_ids: List[int], current_frame: int) -> int:
        """
        Aggiorna il registry con i track del frame corrente
        
        Args:
            track_ids: Lista di ID dei track attivi
            current_frame: Numero frame corrente
            
        Returns:
            Numero di nuove mele rilevate in questo frame
        """
        new_apples = 0
        
        for tid in track_ids:
            # Nuovo ID mai visto prima
            if tid not in self.unique_ids:
                self.unique_ids.add(tid)
                new_apples += 1
                self.total_new_apples += 1
                print(f"[APPLE] 🍎 Nuova mela rilevata! ID {tid} | Totale unico: {len(self.unique_ids)}")
            
            # Aggiorna ultimo frame visto
            self.active_tracks[tid] = current_frame
        
        # Cleanup track persi da troppo tempo
        self._cleanup_lost_tracks(current_frame)
        
        return new_apples
    
    def _cleanup_lost_tracks(self, current_frame: int):
        """Rimuove track persi da troppo tempo dal registry attivo"""
        lost_ids = []
        
        for tid, last_seen in self.active_tracks.items():
            frames_lost = current_frame - last_seen
            if frames_lost > self.lost_threshold:
                lost_ids.append(tid)
        
        for tid in lost_ids:
            del self.active_tracks[tid]
            if lost_ids:  # Log solo se ce n'è almeno uno
                print(f"[TRACK] 👻 ID {tid} perso da {self.lost_threshold} frame (rimosso da attivi)")
    
    def get_stats(self) -> Dict[str, int]:
        """Ritorna statistiche correnti"""
        return {
            "total_unique": len(self.unique_ids),
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
            print("[INIT] 💡 Per abilitare GPU, esegui: bash fix_cuda.sh")
        
        # Carica modelli
        model_manager = ModelManager()
        self.model = model_manager.load_yolo_world()
        self.model.to(self.device)
        
        # Setup ByteTrack tramite supervision
        self.tracker = sv.ByteTrack(
            track_activation_threshold=Config.TRACK_THRESH,
            lost_track_buffer=Config.TRACK_BUFFER,
            minimum_matching_threshold=Config.MATCH_THRESH,
            frame_rate=30  # stima iniziale
        )
        
        # Componenti
        self.id_registry = IDRegistry()
        self.smoother = TrajectorySmooth()
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
        Esegue detection delle mele su un frame con CLASS MERGING + NMS
        
        Args:
            frame: Frame input (BGR)
            
        Returns:
            Detections in formato supervision (senza duplicati)
        """
        # Inferenza con mixed precision se disponibile
        with torch.amp.autocast('cuda', enabled=(self.device == "cuda")):
            results = self.model.predict(
                frame,
                conf=Config.YOLO_CONFIDENCE,
                verbose=False,
                device=self.device
            )
        
        # Converte in formato supervision
        detections = sv.Detections.from_ultralytics(results[0])
        
        # *** CLASS MERGING + NMS per eliminare double counting ***
        if len(detections) > 0:
            # Step 1: Unisci tutte le classi ["apple", "red apple", "green apple"] in classe 0
            if detections.class_id is not None:
                detections.class_id[:] = 0
            
            # Step 2: Applica NMS aggressivo per rimuovere detection sovrapposte
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
        if self.resize_strategy == "adaptive":
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
        
        # 4. Tracking con ByteTrack
        detections = self.tracker.update_with_detections(detections)
        
        # 5. Aggiorna ID Registry
        track_ids = detections.tracker_id.tolist() if detections.tracker_id is not None else []
        self.id_registry.update(track_ids, self.frame_count)
        
        # 6. Annotazione frame (usa frame_resized, non enhanced, per visualizzazione)
        frame_annotated = frame_resized.copy()
        
        if len(detections) > 0:
            # Per ogni detection con track
            for i in range(len(detections)):
                bbox = detections.xyxy[i]
                track_id = int(detections.tracker_id[i]) if detections.tracker_id is not None else -1
                confidence = float(detections.confidence[i]) if detections.confidence is not None else 0.0
                
                if track_id == -1:
                    continue
                
                # Smooth bbox
                bbox_smooth = self.smoother.smooth_bbox(track_id, bbox)
                
                # Disegna ellisse (ROSSO)
                frame_annotated = self.viz.draw_ellipse(frame_annotated, bbox_smooth, track_id)
                
                # Disegna bbox (ROSSO)
                frame_annotated = self.viz.draw_bbox(frame_annotated, bbox_smooth, track_id, confidence)
                
                # Disegna traiettoria (BORDEAUX)
                centers = self.smoother.get_center_history(track_id)
                frame_annotated = self.viz.draw_trajectory(frame_annotated, centers, track_id)
        
        # 7. HUD con statistiche
        stats = self.id_registry.get_stats()
        frame_annotated = self.viz.draw_hud(
            frame_annotated,
            self.frame_count,
            stats["currently_active"],
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
        print("[CUDA] 💡 Per fix GPU, esegui: bash fix_cuda.sh")
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