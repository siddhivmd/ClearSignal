# ClearSignal — Production Content Moderation API

## Project Status

> [!IMPORTANT]
> **Current Status**: 🟢 **Phase 1 Complete (Data Pipeline)** | 🔴 **Phases 2–5 Planned (In Development)**
>
> - **Implemented**: Phase 1 Data Pipeline (`scripts/prepare_data.py`, `scripts/auto_label.py`, `scripts/clean_data.py`).
> - **NOT Yet Implemented**:
>   - Phase 2: Model Fine-Tuning (DeBERTa-v3 multi-label classification)
>   - Phase 3: FastAPI Backend & SHAP Token Attribution
>   - Phase 4: MLOps Containerization & AWS EC2/ECR Deployment
>   - Phase 5: MMD Input Drift Monitoring & CUPED A/B Testing Infrastructure

---

## Overview

ClearSignal is designed as a multi-label content moderation system that categorizes text across 8 harmful content dimensions (`toxic`, `hate_speech`, `harassment`, `misinformation`, `spam`, `self_harm`, `sexual_content`, `clean`) with confidence scores and explainability features.

Currently, **Phase 1 (Data Preparation & Labelling Pipeline)** is fully implemented. The data pipeline aggregates public datasets, runs automated labelling with a pre-trained transformer model, and produces a cleaned dataset ready for model training.

---

## Implemented Data Pipeline (Phase 1)

The `scripts/` directory contains the working data preparation suite:

1. **Data Ingestion (`scripts/prepare_data.py`)**:
   - Downloads 1,000 tweets from `tweets_hate_speech_detection` (500 hate, 500 clean).
   - Downloads 1,000 comments from `ucberkeley-dlab/measuring-hate-speech` (500 hateful, 500 clean).
   - **Pre-Sampling Deduplication & Determinism**: Fixed a data leakage bug where `measuring-hate-speech` (~3.4 annotator entries per text) caused inflated duplicate counts downstream. Deduplication by text is performed *prior* to shuffling (`random_state=42`) and sampling, ensuring 100% deterministic and unique post selection.
   - Aggregates 2,000 unique records into `data/to_label.json`.

2. **Auto-Labelling (`scripts/auto_label.py`)**:
   - Uses `cardiffnlp/twitter-roberta-base-hate-latest` HuggingFace transformer to score texts.
   - Assigns multi-label tags (`hate_speech`, `toxic`, `clean`) and hate confidence scores.
   - Output saved to `data/auto_labelled.json`.

3. **Data Cleaning (`scripts/clean_data.py`)**:
   - Confirms 0 duplicates post-ingestion.
   - Filters out short noise texts (< 3 words).
   - Saves 1,996 cleaned records (1,336 `clean`, 660 `hate_speech`, 660 `toxic`) to `data/custom_labels_clean.csv`.

---

## Roadmap & Phase Status

| Phase | Component | Status | Description |
|---|---|---|---|
| **Phase 1** | **Dataset & Data Pipeline** | ✅ **Completed** | Ingestion, pre-deduplication, auto-labelling, and cleaning pipeline (1,996 records) |
| **Phase 2** | DeBERTa-v3 Model Fine-Tuning | ⏳ Planned | Multi-label fine-tuning with focal loss & temperature scaling |
| **Phase 3** | FastAPI Service & SHAP | ⏳ Planned | Real-time moderation API with token-level explainability |
| **Phase 4** | AWS MLOps Deployment | ⏳ Planned | Docker container, ECR registry, EC2 instance, Redis caching |
| **Phase 5** | Monitoring & CUPED A/B | ⏳ Planned | MMD embedding drift detection & variance-reduced A/B testing |

---

## Repository Structure

```
ClearSignal/
├── data/
│   ├── custom_labels_clean.csv    # Tracked cleaned dataset (1,996 records)
│   ├── to_label.json              # Excluded by .gitignore (raw aggregated)
│   └── auto_labelled.json         # Excluded by .gitignore (intermediate)
├── scripts/
│   ├── prepare_data.py            # Dataset downloader & aggregator
│   ├── auto_label.py              # RoBERTa auto-labelling pipeline
│   └── clean_data.py              # Deduplication & text cleaner
├── .gitignore                     # Git exclusions for raw/intermediate data
└── README.md                      # Project documentation
```

---

## Getting Started

### Installation

```bash
pip install datasets transformers torch pandas
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
