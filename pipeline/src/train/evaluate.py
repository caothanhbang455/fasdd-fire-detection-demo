"""
src/train/evaluate.py

Evaluate a trained YOLO11m-seg checkpoint on the official test split
and extract the box/seg metrics used in the project's run registry.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from ultralytics import YOLO

CLASS_NAMES = {0: "fire", 1: "smoke"}


def evaluate_test_split(
    weights_path: Path,
    data_yaml: Path,
    imgsz: int = 640,
    batch: int = 32,
    device: str = "0",
):
    """Run `.val(split='test')` and return the raw ultralytics metrics object."""
    model = YOLO(str(weights_path))
    return model.val(
        data=str(data_yaml),
        split="test",
        imgsz=imgsz,
        batch=batch,
        device=device,
        verbose=True,
    )


def extract_metrics(metrics) -> Dict[str, Optional[float]]:
    """
    Flatten ultralytics' metrics object into the registry row schema used
    across the project (box + seg mAP, per-class AP50, smoke-fire AP gap).
    """

    def safe_round(v, nd=4):
        return round(float(v), nd) if v is not None else None

    out: Dict[str, Optional[float]] = {
        "box_mAP50": safe_round(metrics.box.map50),
        "box_mAP5095": safe_round(metrics.box.map),
        "precision": safe_round(metrics.box.mp),
        "recall": safe_round(metrics.box.mr),
        "box_ap50_fire": None,
        "box_ap50_smoke": None,
        "seg_mAP50": None,
        "seg_mAP5095": None,
        "seg_ap50_fire": None,
        "seg_ap50_smoke": None,
        "box_AP_gap": None,
    }

    try:
        if len(metrics.box.ap50) >= 2:
            out["box_ap50_fire"] = safe_round(metrics.box.ap50[0])
            out["box_ap50_smoke"] = safe_round(metrics.box.ap50[1])
            out["box_AP_gap"] = safe_round(metrics.box.ap50[1] - metrics.box.ap50[0])
    except Exception:
        pass

    try:
        out["seg_mAP50"] = safe_round(metrics.seg.map50)
        out["seg_mAP5095"] = safe_round(metrics.seg.map)
        if len(metrics.seg.ap50) >= 2:
            out["seg_ap50_fire"] = safe_round(metrics.seg.ap50[0])
            out["seg_ap50_smoke"] = safe_round(metrics.seg.ap50[1])
    except Exception:
        pass

    return out


def benchmark_latency(
    weights_path: Path,
    sample_paths: list[str],
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.45,
    n_warmup: int = 5,
) -> Dict[str, float]:
    """Average single-image inference latency / FPS over `sample_paths`."""
    import time

    model = YOLO(str(weights_path))
    for p in sample_paths[:n_warmup]:
        model.predict(p, imgsz=imgsz, conf=conf, iou=iou, verbose=False)

    bench = sample_paths[n_warmup:]
    t0 = time.perf_counter()
    for p in bench:
        model.predict(p, imgsz=imgsz, conf=conf, iou=iou, verbose=False)
    elapsed = time.perf_counter() - t0

    inf_ms = elapsed / max(1, len(bench)) * 1000
    model_mb = round(Path(weights_path).stat().st_size / 1e6, 1)
    return {"inference_ms": round(inf_ms, 1), "fps": round(1000 / inf_ms, 1), "model_mb": model_mb}
