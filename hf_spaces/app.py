"""
app.py — FASDD-CV Fire & Smoke Detection Demo
HuggingFace Spaces · Gradio · YOLO11m-seg + FLAME post-processing

Upload a surveillance video → select checkpoint → toggle FLAME →
get an annotated output video with confirmed fire/smoke alerts.
"""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Optional

import cv2
import gradio as gr
import numpy as np
import supervision as sv
from ultralytics import YOLO

from flame import FLAMEPipeline

# ── Model registry ────────────────────────────────────────────────────────────
CHECKPOINT_PATHS: dict[str, str] = {
    "YOLO11m Detection (baseline)": "models/det_best.pt",
    "YOLO11m-seg v1 (SAM2 masks)":  "models/seg_v1_best.pt",
    "YOLO11m-seg v2 (refined)":     "models/seg_v2_best.pt",
}

CLASS_NAMES   = {0: "fire", 1: "smoke"}
CLASS_COLORS  = {0: sv.Color(r=230, g=70,  b=4),    # fire orange
                 1: sv.Color(r=100, g=120, b=140)}   # smoke grey-blue
ALERT_COLOR   = sv.Color(r=220, g=20,  b=20)

MAX_FRAMES   = 1800   # cap at 60s @ 30fps to avoid OOM
CONF_DEFAULT = 0.25
IOU_DEFAULT  = 0.45


def _load_models() -> dict[str, Optional[YOLO]]:
    """Load all available checkpoints; missing ones return None."""
    loaded: dict[str, Optional[YOLO]] = {}
    for display_name, path in CHECKPOINT_PATHS.items():
        p = Path(path)
        if p.exists():
            try:
                loaded[display_name] = YOLO(str(p))
                print(f"✓ Loaded  : {display_name}  ({path})")
            except Exception as exc:
                print(f"✗ Failed  : {display_name}  — {exc}")
                loaded[display_name] = None
        else:
            print(f"⚠ Missing : {display_name}  ({path})")
            loaded[display_name] = None
    return loaded


MODELS: dict[str, Optional[YOLO]] = _load_models()
AVAILABLE_MODELS = [k for k, v in MODELS.items() if v is not None]

if not AVAILABLE_MODELS:
    print("⚠ No model checkpoints found. Place .pt files in models/")
    print("  Expected:")
    for name, path in CHECKPOINT_PATHS.items():
        print(f"    {path}  ({name})")


# ── Annotation helpers ────────────────────────────────────────────────────────
def _build_annotators(thickness: int = 2) -> dict:
    return {
        "box":   sv.BoxAnnotator(thickness=thickness),
        "mask":  sv.MaskAnnotator(opacity=0.35),
        "label": sv.LabelAnnotator(
            text_scale=0.45, text_thickness=1,
            text_padding=4,
        ),
    }


def _det_to_sv(results) -> sv.Detections:
    """Convert ultralytics Results object → supervision Detections."""
    if results is None or len(results.boxes) == 0:
        return sv.Detections.empty()
    return sv.Detections.from_ultralytics(results)


def _annotate_frame(
    frame: np.ndarray,
    detections: sv.Detections,
    annotators: dict,
    is_alert: bool = False,
    alert_tracks: Optional[set] = None,
    class_names: dict = CLASS_NAMES,
) -> np.ndarray:
    if len(detections) == 0:
        return frame

    # Build label strings
    labels = []
    for i, (cls_id, conf) in enumerate(
        zip(detections.class_id, detections.confidence)
    ):
        name = class_names.get(int(cls_id), f"cls{cls_id}")
        tid  = (detections.tracker_id[i] if detections.tracker_id is not None else None)
        tid_str = f"#{int(tid)}" if tid is not None else ""
        labels.append(f"{name}{tid_str}  {conf:.0%}")

    vis = frame.copy()

    # Masks (seg models only)
    if detections.mask is not None:
        vis = annotators["mask"].annotate(vis, detections)

    # Boxes
    vis = annotators["box"].annotate(vis, detections)
    vis = annotators["label"].annotate(vis, detections, labels=labels)

    # Alert overlay
    if is_alert:
        h, w = vis.shape[:2]
        cv2.rectangle(vis, (0, 0), (w, h), (0, 0, 220), 4)
        cv2.putText(vis, "🔥 FIRE ALERT", (12, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 220), 2)

    return vis


# ── Core processing function ──────────────────────────────────────────────────
def process_video(
    video_path: str,
    model_display_name: str,
    use_flame: bool,
    conf_threshold: float,
    bg_warmup_frames: int,
    min_track_frames: int,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> tuple[Optional[str], str]:
    """
    Main processing function called by Gradio.

    Returns (output_video_path, stats_markdown).
    """
    if not video_path:
        return None, "⚠ Please upload a video first."

    model = MODELS.get(model_display_name)
    if model is None:
        return None, (
            f"⚠ **{model_display_name}** checkpoint not found.\n\n"
            "Place the `.pt` file in the `models/` folder and restart the Space."
        )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "⚠ Failed to open video file."

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), MAX_FRAMES)

    # Output file
    tmp_out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    out_path = tmp_out.name
    tmp_out.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    flame = FLAMEPipeline(
        bg_warmup_frames=bg_warmup_frames,
        min_track_frames=min_track_frames,
    )
    flame.reset()

    annotators = _build_annotators()

    # Stats
    stats = {
        "frames":     0,
        "raw_dets":   0,
        "fp_stage1":  0,
        "fp_stage2":  0,
        "confirmed":  0,
        "alerts":     0,
        "inf_ms":     [],
        "alert_frames": [],
    }
    alert_tracks: set = set()

    for frame_idx in progress.tqdm(range(total_frames), desc="Processing"):
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO inference
        t0 = time.perf_counter()
        results = model.predict(
            frame, conf=conf_threshold, iou=IOU_DEFAULT,
            verbose=False, stream=False,
        )[0]
        inf_ms = (time.perf_counter() - t0) * 1000
        stats["inf_ms"].append(inf_ms)

        raw_dets = _det_to_sv(results)
        stats["raw_dets"] += len(raw_dets)
        stats["frames"]   += 1

        if use_flame:
            flame_result = flame.process(frame, raw_dets)
            confirmed    = flame_result.detections
            stats["fp_stage1"] += flame_result.fp_stage1
            stats["fp_stage2"] += flame_result.fp_stage2
            stats["confirmed"] += len(confirmed)

            is_alert = len(flame_result.new_alerts) > 0
            if is_alert:
                stats["alerts"] += 1
                stats["alert_frames"].append(frame_idx)
            for tid in flame_result.new_alerts:
                alert_tracks.add(tid)

            vis = _annotate_frame(
                frame, confirmed, annotators,
                is_alert=is_alert, alert_tracks=alert_tracks,
            )
            # Warmup indicator
            if not flame_result.bg_warmed_up:
                cv2.putText(vis, "BG calibrating…", (12, height - 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        else:
            stats["confirmed"] += len(raw_dets)
            vis = _annotate_frame(frame, raw_dets, annotators)

        writer.write(vis)

    cap.release()
    writer.release()

    # ── Stats markdown ────────────────────────────────────────────────────────
    avg_inf = float(np.mean(stats["inf_ms"])) if stats["inf_ms"] else 0.0
    fp_total = stats["fp_stage1"] + stats["fp_stage2"]
    fp_rate  = fp_total / max(1, stats["raw_dets"]) * 100
    alert_ts = [f"{f/fps:.1f}s" for f in stats["alert_frames"][:5]]

    flame_section = ""
    if use_flame:
        flame_section = f"""
### FLAME Post-Processing
| Stage | Suppressed |
|---|---|
| Stage 1 — Background gate | **{stats["fp_stage1"]}** |
| Stage 2 — Trajectory filter | **{stats["fp_stage2"]}** |
| **Total FP suppressed** | **{fp_total}** ({fp_rate:.1f}% of raw) |
| Fire alerts triggered | **{stats["alerts"]}** |
| Alert timestamps (first 5) | {', '.join(alert_ts) if alert_ts else '—'} |
"""

    md = f"""
### Processing Complete

| Metric | Value |
|---|---|
| Frames processed | **{stats["frames"]}** |
| Model | {model_display_name} |
| FLAME | {"✅ Enabled" if use_flame else "❌ Disabled"} |
| Raw detections | **{stats["raw_dets"]}** |
| Confirmed detections | **{stats["confirmed"]}** |
| Avg inference | **{avg_inf:.1f} ms/frame** (~{1000/max(1,avg_inf):.0f} FPS) |
{flame_section}
"""
    return out_path, md


# ── Gradio UI ─────────────────────────────────────────────────────────────────
_TITLE = """
<div style="text-align:center;padding:24px 0 8px">
  <h1 style="font-size:2rem;font-weight:800;color:#E85D04;margin:0">
    🔥 FASDD-CV Fire &amp; Smoke Detection
  </h1>
  <p style="color:#6B7280;margin:8px 0 0;font-size:1rem">
    YOLO11m-seg + SAM2 Pseudo-Masks + FLAME Temporal FP Suppression<br/>
    <a href="https://github.com/YOUR_USERNAME/fasdd-cv" target="_blank"
       style="color:#E85D04">GitHub</a>
    &ensp;·&ensp;
    <a href="YOUR_REPORT_LINK" target="_blank"
       style="color:#E85D04">Technical Report</a>
    &ensp;·&ensp;
    <a href="YOUR_GITHUB_PAGES" target="_blank"
       style="color:#E85D04">Project Page</a>
  </p>
</div>
"""

_NO_MODELS_WARNING = """
> ⚠️ **No model checkpoints found.** Upload your `.pt` files to the `models/` folder:
> - `models/det_best.pt` → YOLO11m detection
> - `models/seg_v1_best.pt` → YOLO11m-seg v1
> - `models/seg_v2_best.pt` → YOLO11m-seg v2
"""

with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="orange", neutral_hue="slate"),
    css="""
    .gr-button-primary { background: #E85D04 !important; border-color: #E85D04 !important; }
    footer { display: none !important; }
    """,
    title="FASDD-CV Fire & Smoke Detection",
) as demo:
    gr.HTML(_TITLE)

    if not AVAILABLE_MODELS:
        gr.Markdown(_NO_MODELS_WARNING)

    with gr.Row():
        # ── Left column: inputs ──────────────────────────────
        with gr.Column(scale=1):
            video_input = gr.Video(label="Upload Video", sources=["upload"])

            model_choice = gr.Dropdown(
                choices=AVAILABLE_MODELS if AVAILABLE_MODELS else list(CHECKPOINT_PATHS.keys()),
                value=AVAILABLE_MODELS[-1] if AVAILABLE_MODELS else None,
                label="Checkpoint",
                info="Switch between detection and segmentation models",
            )

            with gr.Row():
                use_flame = gr.Checkbox(
                    value=True,
                    label="FLAME Post-Processing",
                    info="Background subtraction + trajectory analysis",
                )

            conf_slider = gr.Slider(
                minimum=0.1, maximum=0.9, value=CONF_DEFAULT, step=0.05,
                label="Confidence Threshold",
            )

            with gr.Accordion("FLAME Parameters", open=False):
                bg_warmup = gr.Slider(
                    minimum=10, maximum=150, value=50, step=10,
                    label="BG Warmup Frames",
                    info="Frames to calibrate background model (fixed cameras: 50)",
                )
                min_track = gr.Slider(
                    minimum=2, maximum=10, value=4, step=1,
                    label="Min Track Frames",
                    info="Minimum consecutive frames before raising alert",
                )

            run_btn = gr.Button("▶ Process Video", variant="primary", size="lg")

            # Example clips
            gr.Examples(
                examples=[],        # populate with paths once you have test videos
                inputs=[video_input],
                label="Example Surveillance Clips",
            )

        # ── Right column: outputs ────────────────────────────
        with gr.Column(scale=1):
            video_output = gr.Video(label="Annotated Output", autoplay=True)
            stats_md     = gr.Markdown(value="Results will appear here after processing.")

    run_btn.click(
        fn=process_video,
        inputs=[video_input, model_choice, use_flame, conf_slider, bg_warmup, min_track],
        outputs=[video_output, stats_md],
        show_progress="full",
    )

    gr.Markdown("""
---
**How FLAME works:**
- **Stage 1 — Background Gate:** MOG2 background subtraction filters detections anchored to static scene regions (sunset glows, permanently lit areas).
- **Stage 2 — Trajectory Analysis:** ByteTrack assigns consistent IDs; tracks are validated for fire-like dynamics (persistence, stable area growth, sustained confidence) before raising an alert.

*FLAME: Gragnaniello et al., Neural Computing and Applications, 2025.*
""")


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
