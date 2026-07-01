# FASDD_CV Seg Pipeline

Structured `src/` rebuild of three working notebooks for the FASDD_CV
fire/smoke auxiliary-segmentation project:

1. `FASDD_CV_SAM2_MaskRefine_v2.ipynb` -- 4-tier SAM2 pseudo-mask refinement (Colab)
2. `FASDD_CV_Seg_EDA_Train_labels_seg_refined_resume.ipynb` -- EDA + cleaning + YOLO11m-seg training (Kaggle)
3. `project-phase1-vdt.ipynb` -- final training run after uploading `labels_seg_refined.zip` as a Kaggle dataset (Kaggle)

## Layout

```
fasdd_seg_pipeline/
├── src/
│   ├── data/        annotation loading, dataset.yaml + symlink helpers
│   ├── refine/       the 4-tier mask refinement cascade (config, matching, tiers, progress, pipeline)
│   ├── quality/       coverage/duplicate/polygon EDA, cleaning, visualization
│   └── train/        YOLO11m-seg model loading/resume, training, evaluation, experiment logging
├── scripts/          thin CLI wrappers around src/ for running each stage standalone
├── configs/
│   └── paths.yaml     Colab vs Kaggle path conventions used across the original notebooks
└── notebooks/         drop your own .ipynb files here (kept empty in this repo)
```

## Pipeline order

```
1. run_refinement.py   (Colab)   GT boxes -> match existing masks -> 4-tier cascade -> labels_seg_refined/
2. run_quality_eda.py  (Colab)   coverage + duplicate audit -> still-missing visualization -> labels_seg_clean/
   (zip labels_seg_clean/ or labels_seg_refined/, upload as a Kaggle dataset)
3. run_training.py     (Kaggle)  build YAML + symlinks -> train YOLO11m-seg (fresh or resume)
4. run_eval.py          (Kaggle)  evaluate best.pt on the official test split + latency benchmark
```

## Known data-quality caveat

`labels_seg_refined/` is **not** 100% clean even after the 4-tier cascade:

- Some files contain exact-duplicate polygon lines. Ultralytics' dataloader
  silently collapses identical lines at load time, which understates how
  many distinct masks a `n_seg` count actually represents in training.
  `src/quality/eda.py::scan_image_coverage` reports `n_exact_dup` per image;
  `src/quality/cleaning.py::clean_seg_lines` removes them before training.
- Some GT boxes still have **no** mask after all 4 tiers (coverage stayed
  below `min_coverage_keep`). `src/quality/eda.py::still_missing_after_refinement`
  + `src/quality/visualize.py::visualize_still_missing` surface these cases
  so they can be inspected manually.

## Quick start

```bash
pip install -r requirements.txt

# Colab: refine masks
python scripts/run_refinement.py --img-dir ... --label-dir ... \
    --labels-seg-orig ... --labels-seg-ref ... --progress-json ...

# Colab: audit + clean
python scripts/run_quality_eda.py --img-dir ... --label-dir ... \
    --labels-seg-dir ... --labels-seg-clean-dir ... --out-dir ./quality_report --bbox-fallback

# Kaggle: train
python scripts/run_training.py --img-dir ... --labels-seg-clean-dir ... \
    --train-list ... --val-list ... --work-dir /kaggle/working \
    --project-dir /kaggle/working/runs --run-name yolo11m_seg_v1 --epochs 100

# Kaggle: evaluate
python scripts/run_eval.py --weights .../best.pt --data-yaml .../fasdd_cv_seg.yaml \
    --test-list ... --registry-csv /kaggle/working/registry.csv --run-name yolo11m_seg_v1
```

## notebooks/

Empty on purpose -- drop your own working notebooks here. Nothing in
`src/` or `scripts/` depends on this folder.
