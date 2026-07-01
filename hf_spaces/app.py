"""
app.py — FASDD-CV Fire & Smoke Detection Demo  (v2 fixed)
Fixes: ByteTrack deprecated, Gradio 6 theme/css, frame skip, GPU device,
       H.264 re-encode, Stop button + queue cancel
"""
from __future__ import annotations

import os, shutil, subprocess, tempfile, time
from pathlib import Path
from typing import Generator, Optional

import cv2
import gradio as gr
import numpy as np
import supervision as sv
import torch
from ultralytics import YOLO

from flame import FLAMEPipeline

# ── Device + capabilities ─────────────────────────────────────────────────────
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
HAS_FFMPEG = shutil.which("ffmpeg") is not None
print(f"Device  : {DEVICE}")
print(f"ffmpeg  : {'yes' if HAS_FFMPEG else 'NO — output may not be browser-compatible'}")

# ── Constants ─────────────────────────────────────────────────────────────────
YOLO_SKIP    = 2      # run YOLO every N frames; reuse last det for skipped frames
YOLO_IMGSZ   = 640
MAX_FRAMES   = 1800   # 60s @ 30fps cap
CONF_DEFAULT = 0.25
IOU_DEFAULT  = 0.45

# ── Model registry ────────────────────────────────────────────────────────────
CHECKPOINT_PATHS = {
    "YOLO11m Detection (baseline)": "models/det_best.pt",
    "YOLO11m-seg v1 (SAM2 masks)":  "models/seg_v1_best.pt",
    "YOLO11m-seg v2 (refined)":     "models/seg_v2_best.pt",
}
CLASS_NAMES = {0: "fire", 1: "smoke"}


def _load_models():
    loaded = {}
    for name, path in CHECKPOINT_PATHS.items():
        p = Path(path)
        if p.exists():
            try:
                m = YOLO(str(p))
                m.to(DEVICE)
                loaded[name] = m
                print(f"OK  {name}")
            except Exception as e:
                print(f"ERR {name}: {e}")
                loaded[name] = None
        else:
            print(f"--- {name}  (missing: {path})")
            loaded[name] = None
    return loaded


MODELS         = _load_models()
AVAILABLE      = [k for k, v in MODELS.items() if v is not None]


# ── Helpers ───────────────────────────────────────────────────────────────────
def _reencode_h264(src: str) -> str:
    """Re-encode to H.264 (browser-compatible). Returns path to new file."""
    if not HAS_FFMPEG:
        return src
    dst = src.replace(".mp4", "_out.mp4")
    cmd = ["ffmpeg", "-y", "-i", src,
           "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
           "-movflags", "+faststart", "-an", dst]
    r = subprocess.run(cmd, capture_output=True, timeout=300)
    if r.returncode == 0 and Path(dst).exists():
        os.remove(src)
        return dst
    return src


def _build_ann():
    return {
        "box":   sv.BoxAnnotator(thickness=2),
        "mask":  sv.MaskAnnotator(opacity=0.35),
        "label": sv.LabelAnnotator(text_scale=0.45, text_thickness=1, text_padding=4),
    }


def _annotate(frame, dets, ann, is_alert=False):
    if len(dets) == 0:
        return frame
    labels = []
    for i, (cls_id, conf) in enumerate(zip(dets.class_id, dets.confidence)):
        n   = CLASS_NAMES.get(int(cls_id), f"cls{cls_id}")
        tid = dets.tracker_id[i] if dets.tracker_id is not None else None
        labels.append(f"{n}{'#'+str(int(tid)) if tid is not None else ''} {conf:.0%}")
    vis = frame.copy()
    if dets.mask is not None:
        vis = ann["mask"].annotate(vis, dets)
    vis = ann["box"].annotate(vis, dets)
    vis = ann["label"].annotate(vis, dets, labels=labels)
    if is_alert:
        h, w = vis.shape[:2]
        cv2.rectangle(vis, (0, 0), (w, h), (0, 0, 220), 4)
        cv2.putText(vis, "FIRE ALERT", (14, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 220), 2)
    return vis


# ── Main processing  (generator for streaming + Gradio cancel) ────────────────
def process_video(
    video_path: str,
    model_name: str,
    use_flame: bool,
    conf: float,
    bg_warmup: int,
    min_track: int,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> Generator[tuple[Optional[str], str], None, None]:

    if not video_path:
        yield None, "Please upload a video first."
        return

    model = MODELS.get(model_name)
    if model is None:
        yield None, f"Checkpoint **{model_name}** not loaded. Upload `.pt` to `models/`."
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        yield None, "Could not open video."
        return

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), MAX_FRAMES)

    raw_mp4 = tempfile.mktemp(suffix=".mp4")
    writer  = cv2.VideoWriter(raw_mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    flame = FLAMEPipeline(bg_warmup_frames=bg_warmup, min_track_frames=min_track)
    flame.reset()
    ann = _build_ann()

    st = {"frames":0,"raw":0,"s1":0,"s2":0,"conf":0,"alerts":0,"ms":[],"afr":[]}
    last_dets = sv.Detections.empty()

    yield None, f"Processing on **{DEVICE}** (skip={YOLO_SKIP}, imgsz={YOLO_IMGSZ})…"

    for fi in progress.tqdm(range(total), desc="Frames"):
        ret, frame = cap.read()
        if not ret:
            break
        st["frames"] += 1

        if fi % YOLO_SKIP == 0:
            t0 = time.perf_counter()
            res = model.predict(frame, imgsz=YOLO_IMGSZ, conf=conf,
                                iou=IOU_DEFAULT, device=DEVICE,
                                verbose=False, stream=False)[0]
            st["ms"].append((time.perf_counter() - t0) * 1000)
            last_dets = sv.Detections.from_ultralytics(res) if (
                res is not None and len(res.boxes) > 0
            ) else sv.Detections.empty()

        raw = last_dets
        st["raw"] += len(raw)

        if use_flame:
            r = flame.process(frame, raw)
            confirmed = r.detections
            st["s1"] += r.fp_stage1
            st["s2"] += r.fp_stage2
            st["conf"] += len(confirmed)
            is_alert = bool(r.new_alerts)
            if is_alert:
                st["alerts"] += 1
                st["afr"].append(fi)
            vis = _annotate(frame, confirmed, ann, is_alert)
            if not r.bg_warmed_up:
                cv2.putText(vis, f"BG calibrating {fi}/{bg_warmup}",
                            (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (160, 160, 160), 1)
        else:
            st["conf"] += len(raw)
            vis = _annotate(frame, raw, ann)

        writer.write(vis)

        if fi % 60 == 0 and fi > 0:
            avg = float(np.mean(st["ms"])) if st["ms"] else 0
            yield None, (
                f"⏳ **{int(fi/total*100)}%** — {fi}/{total} frames "
                f"| {avg:.0f} ms/frame | dets: {st['raw']}"
            )

    cap.release()
    writer.release()

    yield None, "Re-encoding to H.264…"
    out = _reencode_h264(raw_mp4)

    # stats
    avg   = float(np.mean(st["ms"])) if st["ms"] else 0
    fptot = st["s1"] + st["s2"]
    fpr   = fptot / max(1, st["raw"]) * 100
    ats   = [f"{f/fps:.1f}s" for f in st["afr"][:5]]

    flame_md = ""
    if use_flame:
        flame_md = f"""
### FLAME Results
| | |
|---|---|
| Stage 1 (BG gate) suppressed | **{st['s1']}** |
| Stage 2 (trajectory) suppressed | **{st['s2']}** |
| **Total FP suppressed** | **{fptot}** ({fpr:.1f}% of raw) |
| Fire alerts raised | **{st['alerts']}** |
| Alert times (first 5) | {', '.join(ats) if ats else '—'} |
"""
    md = f"""
### ✅ Done

| | |
|---|---|
| Frames | **{st['frames']}** |
| Device | **{DEVICE}** |
| Model | {model_name} |
| FLAME | {"✅" if use_flame else "❌"} |
| Frame skip | every **{YOLO_SKIP}** |
| Raw dets | **{st['raw']}** |
| Confirmed | **{st['conf']}** |
| Avg inference | **{avg:.1f} ms/frame** |
{flame_md}"""

    yield out, md


# ── UI ────────────────────────────────────────────────────────────────────────
_TITLE = """
<div style="text-align:center;padding:18px 0 6px">
  <h1 style="font-size:1.75rem;font-weight:800;color:#E85D04;margin:0">
    🔥 FASDD-CV · Fire &amp; Smoke Detection
  </h1>
  <p style="color:#6B7280;margin:6px 0 0;font-size:.9rem">
    YOLO11m-seg · SAM2 Pseudo-Masks · FLAME Temporal FP Suppression ·
    <a href="https://github.com/YOUR_USERNAME/fasdd-cv" target="_blank" style="color:#E85D04">GitHub</a> ·
    <a href="YOUR_REPORT" target="_blank" style="color:#E85D04">Report</a> ·
    <a href="YOUR_PAGES" target="_blank" style="color:#E85D04">Project Page</a>
  </p>
</div>
"""

_CSS = "footer{display:none!important}"

with gr.Blocks(
    title="FASDD-CV",
    theme=gr.themes.Soft(
        primary_hue="orange",
        neutral_hue="slate"
    ),
    css=_CSS,
) as demo:
    gr.HTML(_TITLE)

    if not AVAILABLE:
        gr.Markdown("> ⚠️ No checkpoints loaded. Upload `.pt` files to `models/`.")

    with gr.Row():
        with gr.Column(scale=1):
            video_in = gr.Video(label="Upload Video", sources=["upload"])
            ckpt_dd  = gr.Dropdown(
                choices=AVAILABLE or list(CHECKPOINT_PATHS),
                value=AVAILABLE[-1] if AVAILABLE else None,
                label="Checkpoint",
            )
            flame_cb  = gr.Checkbox(value=True, label="FLAME Post-Processing")
            conf_sl   = gr.Slider(0.1, 0.9, CONF_DEFAULT, step=0.05,
                                  label="Confidence Threshold")

            with gr.Accordion("⚙ FLAME Parameters", open=False):
                bg_sl  = gr.Slider(10, 150, 50, step=10,
                                   label="BG Warmup Frames",
                                   info="50 frames ≈ 1.7s @ 30fps for fixed cameras")
                trk_sl = gr.Slider(2, 10, 4, step=1,
                                   label="Min Track Frames",
                                   info="N consecutive detections before alert")

            with gr.Row():
                run_btn  = gr.Button("▶ Process", variant="primary", size="lg")
                stop_btn = gr.Button("⏹ Stop",    variant="stop",    size="lg")

            gr.Examples(examples=[], inputs=[video_in], label="Example Clips")

        with gr.Column(scale=1):
            video_out = gr.Video(label="Annotated Output", autoplay=True)
            stats_md  = gr.Markdown("Results will appear here.")

    ev = run_btn.click(
        fn=process_video,
        inputs=[video_in, ckpt_dd, flame_cb, conf_sl, bg_sl, trk_sl],
        outputs=[video_out, stats_md],
        show_progress="full",
    )
    stop_btn.click(fn=None, cancels=[ev])

    gr.Markdown("""
---
**FLAME:** Stage 1 — MOG2 background subtraction removes static FP (sunsets, signs).  
Stage 2 — ByteTrack trajectory validation discards non-fire dynamics (headlight flashes, steam).  
*Gragnaniello et al., Neural Computing and Applications 2025.*
""")

demo.queue()

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
    )