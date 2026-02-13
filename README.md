# 🍎 BACCO — Apple Detection & Tracking System

> **Open-vocabulary object detection and multi-object tracking for real-time fruit counting on edge devices.**

Bacco uses YOLOWorld (open-vocabulary) + ByteTrack (multi-object tracking) to detect, track, and uniquely count apples in orchard video — in real-time on NVIDIA Jetson Thor.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Processing Pipeline](#processing-pipeline)
- [Mathematical Foundations](#mathematical-foundations)
- [Configuration Parameters — In Depth](#configuration-parameters--in-depth)
- [Class Reference](#class-reference)
- [Visualization System](#visualization-system)
- [Performance Benchmarks](#performance-benchmarks)
- [Setup & Usage](#setup--usage)
- [Future Improvements](#future-improvements)

---

## Architecture Overview

```
┌──────────────┐    ┌────────────┐    ┌──────────────┐    ┌───────────┐
│  Video Input │───▶│  CLAHE     │───▶│  YOLOWorld   │───▶│ Temporal  │
│  (Auto-Resize)   │  Enhance   │    │  Detection   │    │  Filter   │
└──────────────┘    └────────────┘    └──────────────┘    └─────┬─────┘
                                                               │
                    ┌────────────┐    ┌──────────────┐    ┌────▼──────┐
                    │  Premium   │◀───│  ID Registry │◀───│ ByteTrack │
                    │  Rendering │    │  (Dedup)     │    │  Tracker  │
                    └────────────┘    └──────────────┘    └───────────┘
```

**Single-file design** (`main.py`, ~1400 lines) — all logic in one file for portability on edge devices.

### Technology Stack

| Component | Technology | Role |
|-----------|-----------|------|
| Detection | YOLOWorld v2 (`yolov8s-worldv2`) | Open-vocabulary apple detection via text prompts |
| Tracking | ByteTrack (via `supervision`) | Online multi-object tracking with Kalman filter |
| Enhancement | CLAHE | Adaptive histogram equalization for shadow/occlusion |
| Rendering | OpenCV | Premium dual-layer visualization with HUD |
| Inference | PyTorch + CUDA | GPU-accelerated on Jetson Thor (JetPack 7.1) |

---

## Processing Pipeline

Each frame passes through **9 sequential stages**:

### Stage 1 — Adaptive Resize

Scales the input frame to balance detection quality vs. inference speed:

```
scale = TARGET_SIZE / max(h, w)
new_size = (w × scale, h × scale)      # clamped to [MIN_SIZE, MAX_SIZE]
```

- 4K input (3840×2160) → resized to ~1920×1080
- 360×640 input → upscaled to ~640 (minimum processing size)

### Stage 2 — CLAHE Enhancement

**Contrast-Limited Adaptive Histogram Equalization** improves detection of apples in shadow, behind glass, or partially occluded by leaves.

Mathematically, CLAHE divides the image into tiles and applies histogram equalization independently per tile, with a **clip limit** that redistributes counts above a threshold to prevent noise amplification:

```
For each tile T:
    H(k) = histogram of T at intensity k
    clip_limit = (N_pixels / N_bins) × CLAHE_CLIP_LIMIT
    excess = Σ max(0, H(k) - clip_limit)
    H'(k) = min(H(k), clip_limit) + excess / N_bins
    CDF(k) = Σ_{j=0}^{k} H'(j)        # cumulative distribution
    pixel_out = CDF(pixel_in) × 255     # equalized output
```

Applied only to the L-channel (LAB color space) to preserve color fidelity.

### Stage 3 — YOLOWorld Detection

**Open-vocabulary detection** — no pre-training needed for "apple" class. Instead, text prompts are encoded via CLIP and matched against visual features:

```
text_embeddings = CLIP_text_encoder(YOLO_PROMPTS)     # 10 prompts → 10 embeddings
visual_features = YOLO_backbone(frame)                 # feature maps at 3 scales
detections = cross_attention(visual_features, text_embeddings)
```

**Class merging + NMS**: All prompts produce separate class IDs. We merge them to `class_id=0` ("apple") then apply Non-Maximum Suppression:

```
For each pair of detections (d_i, d_j):
    IoU(d_i, d_j) = Area(d_i ∩ d_j) / Area(d_i ∪ d_j)
    if IoU > NMS_THRESHOLD:
        suppress lower-confidence detection
```

### Stage 4 — Temporal Filter

Eliminates flickering false positives by requiring each detection to appear in **≥ N consecutive frames** at approximately the same position:

```
For each detection d at center (cx, cy):
    match = argmin_{id ∈ history} euclidean_dist(d.center, history[id].center)
    if dist(match) < TEMPORAL_MATCH_DISTANCE:
        history[match].append(frame_count)
    else:
        create new history entry

    if len(history[id]) >= MIN_CONSECUTIVE_FRAMES:
        mark d as STABLE → pass through
    else:
        mark d as UNSTABLE → filter out
```

### Stage 5 — Detection Carry-Forward

Bridges short occlusions by injecting "phantom" detections at the last known position of a temporarily lost track:

```
For each lost track t (not in current detections):
    if t.frames_missing ≤ CARRY_FRAMES:
        confidence_carried = confidence_original × CARRY_DECAY^frames_missing
        inject synthetic detection at t.last_bbox with confidence_carried
    else:
        discard track permanently
```

The exponential decay (`0.9^n`) ensures carried detections lose priority:

| Frame missed | Confidence multiplier |
|---|---|
| 1 | 0.90 |
| 3 | 0.73 |
| 5 | 0.59 |
| 8 | 0.43 |

### Stage 6 — ByteTrack Multi-Object Tracking

ByteTrack is a **two-stage association** tracker:

**Stage A — High-confidence association**: Match high-confidence detections (`score > track_activation_threshold`) against existing tracks using IoU + Kalman-predicted positions:

```
cost_matrix[i,j] = 1 - IoU(track_i.predicted_bbox, detection_j.bbox)
assignment = HungarianAlgorithm(cost_matrix)
if cost[i,j] < (1 - MATCH_THRESH):
    associate track_i ↔ detection_j
```

**Stage B — Low-confidence recovery**: Remaining unmatched tracks are matched against low-confidence detections using the same IoU criterion.

**Kalman Filter** predicts each track's position using a constant-velocity model:

```
State: x = [cx, cy, aspect_ratio, height, vx, vy, v_ar, v_h]
Prediction: x̂_{t+1} = F · x_t       (constant velocity)
Update: x_t = x̂_t + K · (z_t - H·x̂_t)   (Kalman gain K)
```

A track is **confirmed** only after appearing in `MIN_CONSECUTIVE` consecutive frames.

### Stage 7 — ID Registry & Spatial Deduplication

Maintains a **globally unique apple count** by detecting when ByteTrack assigns a new ID to an apple that was previously tracked:

```
For new track_id:
    search recently_lost tracks within dedup_distance (80px):
        dist = ||center_new - center_lost||₂
        if dist < 80:
            map new_id → original_id (deduplicated)
        else:
            count as genuinely new apple
```

### Stage 8 — Bounding Box Smoothing

Exponential Moving Average (EMA) over the last `SMOOTH_WINDOW` frames prevents jitter:

```
bbox_smooth[t] = (1/W) × Σ_{i=t-W+1}^{t} bbox[i]

where W = min(SMOOTH_WINDOW, frames_seen_for_this_track)
```

### Stage 9 — Premium Rendering

Dual-layer rendering with confidence-based coloring (see [Visualization System](#visualization-system)).

---

## Configuration Parameters — In Depth

### Detection Parameters

| Parameter | Value | Mathematical rationale | Next improvement |
|-----------|-------|----------------------|------------------|
| `YOLO_CONFIDENCE` | **0.05** | Very low threshold to maximize recall. We accept ~90% false positive rate at detection stage because downstream filters (temporal, ByteTrack) eliminate noise. The ROC operating point is chosen to be far-right on the recall axis: `TPR ≈ 0.95, FPR ≈ 0.85`. | Adaptive thresholding per-scene based on initial frame statistics |
| `YOLO_IMGSZ` | **1280** | Internal YOLO resolution. Higher = smaller apples resolved, but inference time scales as O(n²). At 1280, apples ≥20px diameter are detectable. | Multi-scale inference (640 + 1280) with NMS merging |
| `NMS_THRESHOLD` | **0.45** | IoU threshold for Non-Maximum Suppression. At 0.3, nearby but distinct apples were suppressed. At 0.5+, true duplicates survive. 0.45 is the empirical sweet spot for apple-on-tree scenes where fruits cluster. | Soft-NMS with Gaussian decay instead of hard suppression |
| `YOLO_PROMPTS` | **10 prompts** | Open-vocabulary CLIP matching. Multiple prompts increase recall by covering semantic variants: `"apple"`, `"red round fruit"`, `"apple partially hidden"`. Each prompt acts as an independent detector — class merging unifies them. | Negative prompts (`"leaf"`, `"branch"`) to reduce FP. Prompt tuning via CLIP similarity analysis |

### Tracking Parameters

| Parameter | Value | Mathematical rationale | Next improvement |
|-----------|-------|----------------------|------------------|
| `TRACK_THRESH` | **0.10** | ByteTrack's `track_activation_threshold`. Internally, ByteTrack uses `det_thresh = max(TRACK_THRESH, 1 - TRACK_THRESH)` for the two-stage split. At 0.10, almost all temporally-filtered detections enter Stage A. | Dynamic thresholding based on detection density per frame |
| `MATCH_THRESH` | **0.80** | Minimum IoU for track↔detection association. **Dense scenes require strict matching** — tested 0.65 and 0.70, both caused cross-association (apple A steals apple B's track) reducing avg tracked from 5.3 to 1.3/frame. 0.8 ensures a detection overlaps ≥80% with the predicted bbox. | Appearance-based re-identification (ReID features) to supplement IoU |
| `TRACK_BUFFER` | **60** | Frames to keep a lost track alive in the Kalman filter (= 2 seconds at 30fps). After 60 frames without matching, the track is terminated. | Adaptive buffer based on camera motion speed |
| `MIN_CONSECUTIVE` | **2** | Frames of consecutive detection before track confirmation. At 3, slow-appearing apples were missed. At 1, noise creates ghost tracks. 2 is the minimum for noise rejection while staying reactive. | Confidence-weighted confirmation (high-confidence detections require fewer frames) |

### Temporal Filter Parameters

| Parameter | Value | Mathematical rationale | Next improvement |
|-----------|-------|----------------------|------------------|
| `MIN_CONSECUTIVE_FRAMES` | **2** | Pre-ByteTrack stability filter. A detection must appear at a similar position for 2 frames to be considered real. This eliminates single-frame noise from YOLO's low confidence threshold. | Weighted by detection confidence — high-confidence detections pass immediately |
| `TEMPORAL_MATCH_DISTANCE` | **60 px** | Maximum Euclidean distance to consider two detections in consecutive frames as "the same object". At 30fps with moderate camera motion, apples move ~20-40px/frame. 60px provides margin for camera shake. | Velocity-aware matching using optical flow |

### Carry-Forward Parameters

| Parameter | Value | Mathematical rationale | Next improvement |
|-----------|-------|----------------------|------------------|
| `CARRY_FRAMES` | **8** | Frames to maintain a "phantom" detection after an apple disappears. Covers ~0.27s of occlusion at 30fps. At 5, apples behind a momentarily intervening branch were lost. | Motion-predicted carry-forward using Kalman state extrapolation |
| `CARRY_DECAY` | **0.90** | Exponential confidence decay: `conf × 0.9^n`. After 8 frames, confidence drops to 43% of original, ensuring phantom detections don't override real ones. The geometric series `Σ 0.9^n` converges, preventing confidence accumulation. | Adaptive decay based on detection stability history |

### Smoothing Parameters

| Parameter | Value | Mathematical rationale | Next improvement |
|-----------|-------|----------------------|------------------|
| `SMOOTH_WINDOW` | **12** | Simple Moving Average window for bbox coordinates. Larger = smoother but laggier. At 20, bboxes lagged behind during camera pan. At 12, lag is ~200ms at 30fps, acceptable for visualization. The SMA acts as a low-pass filter with cutoff frequency `f_c = fps / (π × W)`. | Replace SMA with EMA for O(1) computation and better transient response |

### Image Enhancement Parameters

| Parameter | Value | Mathematical rationale | Next improvement |
|-----------|-------|----------------------|------------------|
| `USE_CLAHE` | **True** | Enables adaptive histogram equalization. Critical for detecting apples in shadow (under leaves) or behind reflective glass | Selective CLAHE only on low-contrast regions detected via variance map |
| `CLAHE_CLIP_LIMIT` | **1.5** | Limits histogram redistribution to 1.5× the average bin height. Higher = more contrast but amplifies noise. At 2.0, leaf textures created false detections. | Scene-adaptive clip limit based on global contrast ratio |
| `CLAHE_TILE_SIZE` | **(8, 8)** | Divides image into 8×8 tiles for local adaptation. Smaller tiles = more local adaptation but more boundary artifacts. 8×8 balances locality vs. smoothness. | Variable tile size based on object density heatmap |

---

## Class Reference

| Class | Lines | Responsibility |
|-------|-------|---------------|
| `Config` | 35–133 | Centralized configuration. All parameters in one place. |
| `ModelManager` | 139–200 | Loads YOLOWorld model, sets CLIP prompts, handles CUDA/CPU device selection |
| `ImageEnhancer` | 206–260 | CLAHE on LAB L-channel. Single-instance, reusable. |
| `ResizeHandler` | 266–360 | 4 strategies: `auto` (scale to TARGET_SIZE), `adaptive`, `native`, `fixed` |
| `IDRegistry` | 362–465 | Unique ID management with spatial dedup of re-identified tracks |
| `TrajectorySmooth` | 470–530 | Per-track SMA smoothing of bbox coordinates |
| `TemporalFilter` | 535–660 | Pre-tracking stability filter using consecutive-frame matching |
| `DetectionCarryForward` | 665–805 | Injects phantom detections for temporarily occluded tracks |
| `VisualizationManager` | 810–1055 | Premium rendering: detection rings, tracked annotations, glassmorphism HUD |
| `AppleDetectionSystem` | 1060–1422 | Main orchestrator: init, `process_frame()`, `process_video()` |

---

## Visualization System

### Dual-Layer Rendering

1. **Background filter** — subtle dark tint (`α=0.20`) to increase annotation contrast
2. **Detection rings** (optional, `SHOW_DETECTED=True`) — semi-transparent ellipses for ALL detections
3. **Tracked apple annotations** — three-pass rendering:
   - **Glow pass**: thick outer ring (`6px`, `α=0.40`) with confidence color
   - **Fill pass**: semi-transparent ellipse interior (`α=0.30`)
   - **Label pass**: opaque border + ID badge + confidence percentage

### Confidence-Based Coloring (BGR)

| Confidence | Color | Label |
|-----------|-------|-------|
| ≥ 35% | `(80, 220, 100)` — Green | High confidence |
| ≥ 20% | `(0, 190, 255)` — Amber | Medium confidence |
| < 20% | `(60, 130, 255)` — Orange-Red | Low confidence |

### Premium HUD

Glassmorphism panel with:
- **BACCO** title with amber accent line
- Tracked count (current frame) + Unique count (cumulative)
- FPS with color coding: green (≥15), amber (≥8), red (<8)
- Progress bar at bottom of frame

---

## Performance Benchmarks

Tested on **NVIDIA Jetson Thor** (131 GB VRAM, JetPack 7.1, CUDA 13.0):

| Video | Resolution | Frames | FPS | Unique Apples | Tracked/Frame |
|-------|-----------|--------|-----|--------------|--------------|
| `test_video_2.mp4` | 360×640 | 192 | **33–37** | 18–19 | 5–9 |
| `test_video_4.mp4` | 2560×1440 | 391 | **8–9** | 7 | 4–6 |
| `test_video.mp4` | 3840×2160 | ~600 | **8–10** | varies | varies |

### Computational Complexity per Frame

| Stage | Time (640p) | Time (4K) | Complexity |
|-------|------------|-----------|-----------|
| Resize + CLAHE | ~2ms | ~8ms | O(pixels) |
| YOLO inference | ~20ms | ~80ms | O(imgsz²) |
| Temporal filter | ~0.5ms | ~0.5ms | O(detections) |
| ByteTrack | ~1ms | ~1ms | O(tracks × detections) |
| Rendering | ~5ms | ~15ms | O(detections) |
| **Total** | **~28ms** | **~105ms** | — |

---

## Setup & Usage

### Prerequisites

- **NVIDIA Jetson Thor** with JetPack 7.1+ (or any CUDA-capable GPU)
- Python 3.10
- CUDA 13.0 + cuDNN

### Installation

```bash
# Clone repository
git clone https://github.com/marcoshehata/bacco.git
cd bacco

# GPU setup (recommended — installs PyTorch with CUDA support)
bash setup_gpu.sh

# Or CPU-only (slower, ~4 FPS)
pip install -r requirements.txt
```

### Running

```bash
# GPU mode (recommended)
./run_bacco_gpu.sh test_video_2.mp4

# Custom video
./run_bacco_gpu.sh /path/to/video.mp4

# CPU mode
python3.10 main.py video.mp4
```

### Controls

- **`q`** — Quit during playback
- **`Ctrl+C`** — Force stop

### Key Configuration Toggles

Edit `Config` class in `main.py`:

```python
SHOW_DETECTED = True      # Show ALL detection rings (not just tracked)
SHOW_TRAJECTORY = True    # Show trajectory lines for tracked apples
USE_CLAHE = False         # Disable CLAHE enhancement
```

---

## Future Improvements

### Short-Term (Parameter Tuning)

1. **Adaptive YOLO confidence** — Calibrate `YOLO_CONFIDENCE` per-scene based on the distribution of scores in the first 30 frames
2. **Soft-NMS** — Replace hard NMS with Gaussian-decay suppression to preserve nearby apples:
   ```
   score_j = score_j × exp(-IoU(d_i, d_j)² / σ²)   instead of   score_j = 0
   ```
3. **Velocity-aware temporal matching** — Use optical flow to predict where a detection should appear in the next frame, improving `TEMPORAL_MATCH_DISTANCE` accuracy

### Medium-Term (Architecture)

4. **Appearance-based ReID** — Extract a 128-d feature embedding per apple using a lightweight CNN head. When ByteTrack loses a track, match by appearance similarity instead of position alone. This would dramatically reduce ID churn.
5. **Multi-scale inference** — Run YOLO at both 640 and 1280, merge results with NMS. Catches both distant small apples and close-up large ones.
6. **Segment Anything integration** — Replace elliptical approximation with pixel-precise segmentation masks for more accurate fruit boundaries.

### Long-Term (System)

7. **ONNX/TensorRT export** — Convert YOLOWorld to TensorRT FP16 engine for 2–3× speedup on Jetson.
8. **Streaming mode** — RTSP/V4L2 input for live camera feeds (drone, tractor-mounted).
9. **3D spatial mapping** — Combine with stereo camera or LiDAR to place apple counts on a spatial map of the orchard.
10. **Yield estimation model** — Map unique apple counts + average sizes to estimated kg/tree using allometric regression.

---

## License

MIT

## Author

Bacco Team — Built for *Fiera di Rimini 2026*