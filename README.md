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

## Label Scope (Phase 2 Multi-Label)

The multi-label model targets 5 labels, listed under `data.labels` in `config.yaml`. `clean` is not a separate output: a text is clean when no label is positive. Every label comes from human judgments, not model predictions.

| Label | Source dataset | Rule | Positives | Licence | Caveat |
|---|---|---|---|---|---|
| `toxic` | `google/civil_comments` + MHS ratings | `toxicity >= 0.5`; MHS: majority rate `insult` or `respect` >= 3 | ~144k + 770 | CC0 / CC BY 4.0 | MHS rule is an approximation (MHS has no toxicity question) |
| `hate_speech` | `google/civil_comments` + Phase 1 datasets | `identity_attack >= 0.5`; Phase 1 `human_label` | ~13k + 999 | CC0 / dataset licences | |
| `harassment` | `google/civil_comments` + MHS ratings | `insult >= 0.5` or `threat >= 0.5`; MHS: majority rate `insult` or `violence` >= 3 | ~107k + 733 | CC0 / CC BY 4.0 | |
| `spam` | `codesignal/sms-spam-collection` | `label == spam` | ~750 | CC BY 4.0 | SMS messages, not comments |
| `self_harm` | `Ram07/Detection-for-Suicide` | `class == suicide` | ~67k | MIT | **Provisional.** Label is the subreddit the post came from (r/SuicideWatch vs r/teenagers), not a per-post human judgment. Includes posts *about* someone else. A hand-labelled sample is needed to measure the noise. |

Civil Comments scores are the fraction of ~10 human raters who applied the label, so `>= 0.5` means a majority agreed.

Measuring-hate-speech (MHS) rows also carry the original annotators' 0–4 ratings (higher = worse, ~3.4 raters per comment). A rater agrees when they give 3 or 4. The label is `1` when at least half agree, `0` when none do, and `-1` when raters are split. Many MHS comments that are not hate speech are still insulting: 271 of 497 were rated toxic by a majority.

**Deferred: `misinformation`.** Whether a post is false usually can't be judged from its text alone; it needs fact-checking against outside sources. No public human-labelled dataset of short posts passed the checks.

**Out of scope: `sexual_content`.** Excluded by project decision. Data is available if it is added later: set `sexual_explicit >= 0.5` on `google/civil_comments` (~4.7k positives, CC0) and add the label to `data.labels`. Clean comments are still required to score below 0.1 on `sexual_explicit`.

**Rejected candidates:**
- `vibhorag101/suicide_prediction_dataset_phr`: text is lemmatized with stopwords removed, so it doesn't look like real posts.
- `av9ash/CSSR-S_labelled_suicidewatch_posts_reddit`: labelled by LLMs, not people.
- `ucirvine/sms_spam`: licence listed as unknown. `codesignal/sms-spam-collection` has the same data under CC BY 4.0.

**Missing labels:** each dataset only annotates some labels (an SMS was never rated for hate speech). Unannotated labels are treated as *unknown*, not negative, and are masked out of the training loss.

### Building the multi-label dataset

`scripts/build_multilabel.py` combines the sources above into `data/multilabel.csv` (settings under `multilabel:` in `config.yaml`):

- Each label column holds `1` (positive), `0` (negative) or `-1` (unknown: that source never annotated this label).
- Civil Comments: up to 5,000 positives sampled per configured label, plus 10,000 clean comments scoring below 0.1 on every attribute. Comments scoring between 0.1 and 0.5 are left out, so they are never used as negatives.
- Self-harm: 5,000 posts from each class. SMS spam and Phase 1 data: all rows.
- Duplicate texts and texts under 3 words are dropped. The `split` column fixes an 80/10/10 train/val/test split, stratified by source and by whether any label is positive.
- The output file is excluded from git. Run the script to regenerate it.

### Multi-label training

`scripts/train_multilabel.py` fine-tunes a model with one sigmoid output per label (settings under `multilabel_training:` in `config.yaml`).

- **Masked loss**: binary cross-entropy is computed only on known labels; `-1` entries are ignored.
- **Metrics**: precision / recall / F1 per label, on rows where that label is known, plus macro F1. The best epoch by validation macro F1 is saved.
- **Out-of-source flag rate**: for each label, the share of texts flagged from sources that never annotated it (e.g. spam on Civil Comments). A high rate means the model learned a source's writing style instead of the label.
- **Outputs** in `save_dir`: model, `metrics.json`, `test_metrics.json`, `train_log.csv`, and `val_probs.csv` / `test_probs.csv` (per-label probabilities, used later for threshold tuning).
- `max_train_rows` / `max_eval_rows` subsample each split proportionally by source, for quick CPU runs. Set them to `null` for a full run.

---

## Roadmap & Phase Status

| Phase | Component | Status | Description |
|---|---|---|---|
| **Phase 1** | **Dataset & Data Pipeline** | ✅ **Completed** | Ingestion, pre-deduplication, auto-labelling, and cleaning pipeline (1,996 records) |
| **Phase 2** | Model Fine-Tuning | 🟡 **In Progress** | ✅ DistilBERT binary baseline on human labels. ✅ Label scope defined (5 labels). ⏳ DeBERTa-v3 multi-label with focal loss & temperature scaling |
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
│   ├── multilabel.csv             # Excluded by .gitignore (written by build_multilabel.py)
│   ├── to_label.json              # Excluded by .gitignore (raw aggregated)
│   └── auto_labelled.json         # Excluded by .gitignore (intermediate)
├── models/
│   ├── clearsignal-v1/            # DistilBERT trained on RoBERTa auto-labels
│   └── clearsignal-v2/            # DistilBERT trained on human labels (current)
├── scripts/
│   ├── prepare_data.py            # Dataset downloader & aggregator
│   ├── auto_label.py              # RoBERTa auto-labelling pipeline
│   ├── clean_data.py              # Deduplication & text cleaner
│   ├── build_multilabel.py        # Multi-label dataset builder (Phase 2)
│   ├── train_multilabel.py        # Multi-label training with masked loss (Phase 2)
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

### Building the Multi-Label Dataset

```bash
# Downloads Civil Comments (~1.8M rows), SMS spam and Reddit data, writes data/multilabel.csv
python scripts/build_multilabel.py
```

### Training the Multi-Label Model

```bash
# Quick CPU run by default (6k training rows, 1 epoch). Saves to models/clearsignal-ml-v1
python scripts/train_multilabel.py
```

### Training the Binary Model

```bash
# Trains DistilBERT and saves to models/clearsignal-v2 (~20 min on CPU)
python scripts/train.py
```
