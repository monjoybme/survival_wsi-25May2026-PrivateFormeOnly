# WSI Survival Analysis Pipeline
## Deep Learning-Based Survival Prediction from H&E Whole Slide Images

This pipeline implements survival analysis using a modified VGG-16 network trained with the **Cox proportional hazards negative log-partial likelihood** loss on H&E stained WSI tiles, exactly as described in the Sherlock-Lung/MSK study.

---

## Project Structure

```
survival_wsi/
│
├── config.py                        # Central configuration (all hyperparameters)
│
├── preprocess/
│   └── tiling.py                    # WSI → non-overlapping 256×256 @ 20× tiles
│
├── data/
│   ├── splitter.py                  # Patient-level splits: train/val + 10-fold CV
│   ├── dataset.py                   # PyTorch Datasets (training & inference)
│   └── sampler.py                   # Patient-aware batch sampler + augmentation
│
├── models/
│   ├── vgg16_survival.py            # Modified VGG-16 → scalar log-hazard output
│   └── cox_loss.py                  # Cox negative log-partial likelihood loss
│
├── training/
│   ├── trainer.py                   # Training loop (Adam, AMP, checkpointing, resume)
│   ├── cross_validation.py          # 10-fold CV → OOF threshold derivation
│   └── early_stopping.py            # Patience-based early stopping
│
├── evaluation/
│   ├── inference.py                 # Tile inference + slide-level aggregation
│   └── metrics.py                   # C-index, log-rank, threshold derivation
│
├── utils/
│   ├── checkpoint.py                # Save/load full training state
│   ├── logger.py                    # Console + rotating file logger
│   ├── visualization.py             # Loss curves, KM plots, risk distributions
│   └── data_utils.py               # CSV validation + template generator
│
├── run_tiling.py                    # Step 1: Tile all WSIs
├── train.py                         # Step 2: Train (CV + final model)
├── test.py                          # Step 3: Evaluate on val + external cohorts
└── requirements.txt
```

---

## Environment Setup

```bash
conda create -n survival_wsi python=3.10
conda activate survival_wsi

# Install OpenSlide system library first
# Ubuntu/Debian:
sudo apt-get install openslide-tools libopenslide-dev
# macOS:
brew install openslide

pip install -r requirements.txt
```

---

## Data Preparation

### Clinical CSV Format
Both Sherlock-Lung and MSK CSVs must have these columns:

| Column | Type | Description |
|---|---|---|
| `patient_id` | str | Unique patient identifier |
| `wsi_path` | str | Absolute path to the `.svs` / `.ndpi` / `.tiff` WSI file |
| `survival_time` | float | Overall survival in days (or months — be consistent) |
| `event` | int | 1 = death occurred, 0 = censored |

Optional columns (ignored by pipeline): `age`, `sex`, `stage`, `histology`.

Generate a template:
```bash
python utils/data_utils.py template data/sherlock_clinical.csv --n 10 --cohort sherlock
python utils/data_utils.py template data/msk_clinical.csv      --n 10 --cohort msk
```

Validate before running:
```bash
python utils/data_utils.py validate data/sherlock_clinical.csv --check_files
```

---

## Full Pipeline

### Step 1 — Tile WSIs
```bash
python run_tiling.py \
    --sherlock_csv data/sherlock_clinical.csv \
    --msk_csv      data/msk_clinical.csv \
    --out_dir      data/tiles \
    --tile_size    256 \
    --magnification 20 \
    --tissue_threshold 0.5 \
    --max_tiles    2000
```
Produces: `data/tiles/{patient_id}/{row}_{col}.png`
A `tile_manifest.json` is saved for resume support.

---

### Step 2 — Train
```bash
# Full pipeline: 10-fold CV → threshold derivation → final model training
python train.py \
    --sherlock_csv data/sherlock_clinical.csv \
    --msk_csv      data/msk_clinical.csv \
    --device       cuda

# Skip CV (use if threshold already derived or to save time)
python train.py --no_cv

# Resume interrupted training
python train.py --resume checkpoints/final_epoch_042.pth
```

**What happens:**
1. Loads Sherlock-Lung patients; builds train (n=454) / val (n=141) split.
2. Runs 10-fold stratified CV within training set:
   - Each fold: ~408 train / ~46 val patients
   - Saves best checkpoint per fold under `checkpoints/cv_fold_XX_best.pth`
   - Collects out-of-fold (OOF) predictions
3. Derives risk threshold via log-rank optimisation on OOF predictions.
4. Retrains final model on **all 454 training patients**.
5. Saves best final model to `checkpoints/final_best.pth`.

**Key outputs:**
```
checkpoints/final_best.pth          ← best model (early-stopped)
checkpoints/cv_fold_XX_best.pth     ← per-fold best models
results/oof_risk_scores.csv         ← out-of-fold predictions
results/cv_summary.json             ← CV C-indices per fold
results/risk_threshold.json         ← derived dichotomisation threshold
figures/final_loss.png              ← train/val loss curve
figures/final_cindex.png            ← validation C-index curve
figures/cv_cindex.png               ← per-fold C-index bar chart
```

---

### Step 3 — Evaluate
```bash
# Evaluate on both internal (n=141) and external (MSK) validation
python test.py --checkpoint checkpoints/final_best.pth

# Internal validation only
python test.py --checkpoint checkpoints/final_best.pth --cohort internal

# MSK external validation only
python test.py --checkpoint checkpoints/final_best.pth --cohort msk

# Apply a specific risk threshold
python test.py --checkpoint checkpoints/final_best.pth --threshold 0.123
```

**Key outputs:**
```
results/internal_validation_risk_scores.csv
results/msk_external_risk_scores.csv
results/evaluation_summary.json           ← C-index, p-value table
figures/internal_validation_km_curve.png  ← Kaplan-Meier plot
figures/internal_validation_risk_dist.png ← Risk score distribution
figures/msk_external_km_curve.png
figures/msk_external_risk_dist.png
```

---

## Model Architecture

```
Input: (B, 3, 256, 256)  ← ImageNet-normalised RGB tile
    │
    ▼
VGG-16 Convolutional Backbone (5 blocks, frozen during feature extraction)
    │
    ▼
AdaptiveAvgPool2d → (B, 512, 7, 7)
    │
    ▼
Flatten → (B, 25088)
    │
    ▼
Linear(25088 → 4096) → ReLU → Dropout(0.50)
    │
    ▼
Linear(4096 → 4096) → ReLU → Dropout(0.50)
    │
    ▼
Linear(4096 → 1)           ← no activation
    │
    ▼
Output: (B,)  ← log-hazard score (log-risk) per tile
```

---

## Loss Function

The network is optimised with the **Cox negative log-partial likelihood**:

```
L = -∑_{i ∈ U} [ R_i − log ∑_{j ∈ Ω_i} exp(R_j) ]
```

where:
- `U` = set of uncensored (event = 1) instances in the mini-batch
- `Ω_i` = risk set: all j in the batch with `survival_time_j ≥ survival_time_i`
- `R_i` = predicted log-hazard for tile i

Implemented in `models/cox_loss.py` with:
- Ascending time-sort for efficient risk-set computation
- Log-sum-exp trick for numerical stability
- Gradient clipping (max norm = 5.0) for training stability

---

## Optimiser Configuration (exact paper values)

| Hyperparameter | Value |
|---|---|
| Optimiser | Adam |
| Learning rate | 0.001 |
| β₁ | 0.9 |
| β₂ | 0.999 |
| ε | 1e-8 |
| L2 weight decay | 4 × 10⁻⁴ |
| Dropout | 50% |
| Max epochs | 150 |
| LR scheduler | ReduceLROnPlateau (patience=7, factor=0.5) |
| Early stopping | patience=15 epochs |

---

## Slide-Level Risk Aggregation

At inference, every tile in a WSI receives a log-hazard score.  These are aggregated to a single slide-level risk score:

```python
slide_risk = mean(tile_log_hazards)    # primary (paper default)
# Alternatives evaluated but not used:
# slide_risk = median(tile_log_hazards)
# slide_risk = max(tile_log_hazards)
```

The slide risk score is then dichotomised at the OOF-derived threshold for Kaplan-Meier analysis.

---

## Resuming Interrupted Runs

The pipeline automatically saves:
- A full epoch checkpoint every epoch (`checkpoints/final_epoch_XXX.pth`)
- The best model so far (`checkpoints/final_best.pth`)
- Training metrics inside every checkpoint

On restart, `train.py` automatically detects and loads the latest epoch checkpoint:
```bash
python train.py   # auto-resumes if checkpoints/ contains epoch files
```

Or specify explicitly:
```bash
python train.py --resume checkpoints/final_epoch_042.pth
```

---

## Expected Results (Reference)

Based on the published Sherlock-Lung study:

| Cohort | C-index |
|---|---|
| CV (10-fold mean) | ≈ 0.62–0.68 |
| Internal validation (n=141) | ≈ 0.65 |
| MSK external (n=varies) | ≈ 0.62 |

*Exact values depend on slide quality, stain normalisation, and hardware.*
