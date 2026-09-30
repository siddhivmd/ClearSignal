# ClearSignal — Production Content Moderation API

## Project Status

> [!IMPORTANT]
> **Current Status**: 🟢 **Phase 1 Complete (Data Pipeline)** | 🟡 **Phase 2 In Progress (Binary Baseline Trained)** | 🔴 **Phases 3–5 Planned**
>
> - **Implemented**:
>   - Phase 1 Data Pipeline (`scripts/prepare_data.py`, `scripts/auto_label.py`, `scripts/clean_data.py`).
>   - Phase 2 baseline: DistilBERT binary classifier (flagged vs clean) trained on human labels (`scripts/train.py`).
> - **NOT Yet Implemented**:
>   - Phase 2 target: DeBERTa-v3 multi-label classification across all 8 categories
>   - Phase 3: FastAPI Backend & SHAP Token Attribution
>   - Phase 4: MLOps Containerization & AWS EC2/ECR Deployment
>   - Phase 5: MMD Input Drift Monitoring & CUPED A/B Testing Infrastructure

---

## Overview

ClearSignal is designed as a multi-label content moderation system that categorizes text across 8 harmful content dimensions (`toxic`, `hate_speech`, `harassment`, `misinformation`, `spam`, `self_harm`, `sexual_content`, `clean`) with confidence scores and explainability features.

Currently, **Phase 1 (Data Preparation & Labelling Pipeline)** is fully implemented, and **Phase 2** has a working binary baseline. The data pipeline aggregates public datasets, keeps each dataset's human annotations, runs automated labelling with a pre-trained transformer model, and produces a cleaned dataset ready for model training.

---

## Implemented Data Pipeline (Phase 1)

The `scripts/` directory contains the working data preparation suite:

1. **Data Ingestion (`scripts/prepare_data.py`)**:
   - Downloads 1,000 tweets from `tweets_hate_speech_detection` (500 hate, 500 clean).
   - Downloads 1,000 comments from `ucberkeley-dlab/measuring-hate-speech` (500 hateful, 500 clean).
   - **Pre-Sampling Deduplication & Determinism**: Fixed a data leakage bug where `measuring-hate-speech` (~3.4 annotator entries per text) caused inflated duplicate counts downstream. Deduplication by text is performed *prior* to shuffling (`random_state=42`) and sampling, ensuring 100% deterministic and unique post selection.
   - **Human labels**: each record keeps its source dataset's annotation as `human_label` (1 = hateful, 0 = clean). For tweets this is the dataset's `label`; for measuring-hate-speech it is `hate_speech_score > 0.5` (hateful) or `< 0.2` (clean).
   - Aggregates 2,000 unique records into `data/to_label.json`.

2. **Auto-Labelling (`scripts/auto_label.py`)**:
   - Uses `cardiffnlp/twitter-roberta-base-hate-latest` HuggingFace transformer to score texts.
   - Assigns tags (`hate_speech`, `toxic`, `clean`) and hate confidence scores. Note: `hate_speech` and `toxic` are always assigned together, so these labels are effectively binary.
   - Output saved to `data/auto_labelled.json`.

3. **Data Cleaning (`scripts/clean_data.py`)**:
   - Confirms 0 duplicates post-ingestion.
   - Filters out short noise texts (< 3 words).
   - Saves 1,996 cleaned records (999 human-labelled hateful, 997 clean) to `data/custom_labels_clean.csv`.

---

## Model Training (Phase 2 — Binary Baseline)

`scripts/train.py` fine-tunes `distilbert-base-uncased` as a binary classifier (flagged = 1, clean = 0), using the settings in `config.yaml`.

- **Labels**: `data.label_source` in `config.yaml` selects the ground truth. `"human"` (default) uses the source datasets' annotations; `"auto"` uses the RoBERTa auto-labels.
- **Split**: stratified 80 / 10 / 10 train / val / test (`seed: 42`). The best epoch by validation F1 is saved.
- **Benchmark**: on the test set, the script scores both DistilBERT and the RoBERTa auto-labeller against the same labels and saves the result to `test_metrics.json`.

### Why human labels

The first model (`clearsignal-v1`) was trained on RoBERTa's auto-labels and reached 89% validation accuracy, but that only measured agreement with RoBERTa. Compared with the human annotations, RoBERTa labelled 474 of 999 hateful texts as clean. `clearsignal-v2` is trained and evaluated on the human labels instead.

### Results (`clearsignal-v2`, 200 human-labelled test texts)

| Metric | DistilBERT v2 | RoBERTa auto-labeller |
|---|---|---|
| Accuracy | **0.755** | 0.735 |
| Precision (flagged) | 0.738 | **0.841** |
| Recall (flagged) | **0.790** | 0.580 |
| F1 (flagged) | **0.763** | 0.686 |
| Macro F1 | **0.755** | 0.729 |

With 200 test texts, differences of a few points are within noise. The recall gap is the clear result: RoBERTa misses 42% of hateful texts.

---

## Roadmap & Phase Status

| Phase | Component | Status | Description |
|---|---|---|---|
| **Phase 1** | **Dataset & Data Pipeline** | ✅ **Completed** | Ingestion, pre-deduplication, auto-labelling, and cleaning pipeline (1,996 records) |
| **Phase 2** | Model Fine-Tuning | 🟡 **In Progress** | ✅ DistilBERT binary baseline on human labels. ⏳ DeBERTa-v3 multi-label with focal loss & temperature scaling |
| **Phase 3** | FastAPI Service & SHAP | ⏳ Planned | Real-time moderation API with token-level explainability |
| **Phase 4** | AWS MLOps Deployment | ⏳ Planned | Docker container, ECR registry, EC2 instance, Redis caching |
| **Phase 5** | Monitoring & CUPED A/B | ⏳ Planned | MMD embedding drift detection & variance-reduced A/B testing |

---

## Repository Structure

```
ClearSignal/
├── config.yaml                    # Training configuration
├── data/
│   ├── custom_labels_clean.csv    # Cleaned dataset with human + auto labels (1,996 records)
│   ├── custom_labels_binary.csv   # Binary training dataset (written by train.py)
│   ├── to_label.json              # Excluded by .gitignore (raw aggregated)
│   └── auto_labelled.json         # Excluded by .gitignore (intermediate)
├── models/
│   ├── clearsignal-v1/            # DistilBERT trained on RoBERTa auto-labels
│   └── clearsignal-v2/            # DistilBERT trained on human labels (current)
├── scripts/
│   ├── prepare_data.py            # Dataset downloader & aggregator
│   ├── auto_label.py              # RoBERTa auto-labelling pipeline
│   ├── clean_data.py              # Deduplication & text cleaner
│   └── train.py                   # DistilBERT fine-tuning & evaluation
├── .gitignore                     # Git exclusions for raw data and model weights
└── README.md                      # Project documentation
```

Model weights (`model.safetensors`, ~268 MB) are excluded from git. Run `scripts/train.py` to produce them.

---

## Getting Started

### Installation

```bash
pip install datasets transformers torch pandas scikit-learn pyyaml
```

### Running the Data Pipeline

```bash
# 1. Download raw data
python scripts/prepare_data.py

# 2. Run auto-labelling
python scripts/auto_label.py

# 3. Clean dataset & export CSV
python scripts/clean_data.py
```

### Training the Model

```bash
# Trains DistilBERT and saves to models/clearsignal-v2 (~20 min on CPU)
python scripts/train.py
```
