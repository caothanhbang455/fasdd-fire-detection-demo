#!/usr/bin/env python3
"""
scripts/run_eval.py

CLI entrypoint for evaluating a trained YOLO11m-seg checkpoint on the
official test split, plus an optional inference-latency benchmark.
Mirrors S10-S11 of project-phase1-vdt.ipynb.

Example:

    python scripts/run_eval.py \
        --weights /kaggle/working/runs/yolo11m_seg_v1/weights/best.pt \
        --data-yaml /kaggle/working/fasdd_cv_seg.yaml \
        --test-list /kaggle/working/test_images.txt \
        --registry-csv /kaggle/working/registry.csv \
        --run-name yolo11m_seg_v1
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train.evaluate import benchmark_latency, evaluate_test_split, extract_metrics
from src.train.logging_utils import append_to_registry


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a YOLO11m-seg checkpoint on the test split.")
    p.add_argument("--weights", required=True, type=Path)
    p.add_argument("--data-yaml", required=True, type=Path)
    p.add_argument("--test-list", type=Path, default=None, help="for the latency benchmark")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--device", default="0")
    p.add_argument("--registry-csv", type=Path, default=None)
    p.add_argument("--run-name", default=None)
    p.add_argument("--n-bench", type=int, default=20)
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Evaluating: {args.weights}")
    metrics = evaluate_test_split(
        args.weights, args.data_yaml, imgsz=args.imgsz, batch=args.batch, device=args.device,
    )
    result = extract_metrics(metrics)

    print("\n=== Test Split Results ===")
    for k, v in result.items():
        print(f"  {k:18s}: {v}")

    if args.test_list and args.test_list.exists():
        paths = [p.strip() for p in args.test_list.read_text().splitlines() if p.strip()]
        sample = random.sample(paths, min(5 + args.n_bench, len(paths)))
        bench = benchmark_latency(args.weights, sample, imgsz=args.imgsz, n_warmup=5)
        print("\n=== Latency Benchmark ===")
        for k, v in bench.items():
            print(f"  {k:18s}: {v}")
        result.update(bench)

    if args.registry_csv and args.run_name:
        result["run_name"] = args.run_name
        append_to_registry(args.registry_csv, result)
        print(f"\nAppended to registry: {args.registry_csv}")


if __name__ == "__main__":
    main()
