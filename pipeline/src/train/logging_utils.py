"""
src/train/logging_utils.py

Three-layer experiment tracking used across the training notebooks:
  Layer 1 -- wandb (optional, live dashboards)
  Layer 2 -- run_config.json   (per-run snapshot, written before AND
                                 updated after training)
  Layer 3 -- registry.csv      (one row per run, append-only, for
                                 cross-run comparison tables)
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

REGISTRY_COLUMNS = [
    "run_name", "date", "model", "mode", "imgsz", "epochs_run",
    "box_mAP50", "box_mAP50_fire", "box_mAP50_smoke", "box_mAP5095",
    "seg_mAP50", "seg_mAP50_fire", "seg_mAP50_smoke", "seg_mAP5095",
    "AP_gap_smoke_minus_fire_box", "precision", "recall",
    "inference_ms_t4", "fps_t4", "model_mb", "notes", "wandb_run_url",
]


def init_wandb(
    enabled: bool,
    project: str,
    run_name: str,
    config: Optional[Dict] = None,
):
    """Initialize a wandb run if enabled and the package is importable."""
    if not enabled:
        return None
    try:
        import wandb
        return wandb.init(project=project, name=run_name, config=config or {})
    except ImportError:
        print("wandb not installed -- skipping experiment tracking.")
        return None


def save_run_config(run_dir: Path, extra: Optional[Dict] = None) -> Path:
    """Write/merge run_config.json (Layer 2) inside the run directory."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = run_dir / "run_config.json"

    cfg = {}
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
    cfg.setdefault("created_at", datetime.now().isoformat())
    cfg.update(extra or {})

    cfg_path.write_text(json.dumps(cfg, indent=2))
    return cfg_path


def update_run_results(run_dir: Path, results: Dict) -> Path:
    """Merge post-training results into run_config.json's `results` key."""
    cfg_path = Path(run_dir) / "run_config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    cfg.setdefault("results", {}).update(results)
    cfg_path.write_text(json.dumps(cfg, indent=2))
    return cfg_path


def load_registry(registry_path: Path) -> pd.DataFrame:
    registry_path = Path(registry_path)
    if registry_path.exists():
        return pd.read_csv(registry_path)
    return pd.DataFrame(columns=REGISTRY_COLUMNS)


def append_to_registry(registry_path: Path, row: Dict) -> pd.DataFrame:
    """Append (or update, if run_name already present) one row in registry.csv."""
    df = load_registry(registry_path)
    if "run_name" in df.columns and (df["run_name"] == row.get("run_name")).any():
        mask = df["run_name"] == row["run_name"]
        for k, v in row.items():
            df.loc[mask, k] = v
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(registry_path, index=False)
    return df
