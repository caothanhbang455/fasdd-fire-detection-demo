---
title: fasdd-cv-demo
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.34.2"
app_file: app.py
pinned: false
---

# FASDD-CV Fire & Smoke Detection

Demo for the VDT Phase 1 fire/smoke detection project.

**Pipeline:**
1. YOLO11m-seg trained on FASDD_CV (95,314 images) with SAM2 pseudo-masks (4-tier cascade)
2. FLAME post-processing: background subtraction + trajectory analysis for FP suppression

## Setup — Adding Checkpoints

Upload your trained `.pt` checkpoints via the **Files** tab:

```
models/det_scratch.pt       # YOLO11m detection baseline
models/seg_v1_best.pt    # YOLO11m-seg v1
models/det_tuning.pt    # 
```

## Usage

1. Upload a surveillance video (mp4/avi/mov)
2. Select a checkpoint from the dropdown
3. Toggle FLAME on/off to compare FP suppression effect
4. Click **Process Video**

## References

- Gragnaniello et al., *FLAME: fire detection in videos combining a deep neural network with a model-based motion analysis*, Neural Computing and Applications, 2025.
- Ultralytics YOLO11, 2024.
- SAM 2: Segment Anything in Images and Videos, 2024.
