import os
import yaml
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# Label value meaning: 1 = positive, 0 = negative, -1 = unknown (source never annotated it).
# Unknown labels are masked out of the training loss.
UNKNOWN = -1


def civil_comments_frame(labels, cfg, seed):
    """Sample Civil Comments. Scores are the fraction of human raters applying each label."""
    print("Loading google/civil_comments...", flush=True)
    ds = load_dataset("google/civil_comments", split="train")
    df = ds.to_pandas()
    threshold = cfg["threshold"]

    rules = {
        "toxic": df["toxicity"] >= threshold,
        "hate_speech": df["identity_attack"] >= threshold,
        "harassment": (df["insult"] >= threshold) | (df["threat"] >= threshold),
        "sexual_content": df["sexual_explicit"] >= threshold,
    }
    for name, mask in rules.items():
        df[name] = mask.astype(int)

    # Clean = every rater score below the borderline cutoff, so ambiguous comments are not used as negatives
    score_cols = ["toxicity", "severe_toxicity", "obscene", "threat", "insult", "identity_attack", "sexual_explicit"]
    is_clean = (df[score_cols] < cfg["clean_max_score"]).all(axis=1)

    # Sample positives label by label; a comment picked for one label keeps its other labels too
    rng = np.random.default_rng(seed)
    picked = set()
    for name in rules:
        pos_idx = df.index[df[name] == 1].difference(list(picked))
        n = min(cfg["samples_per_label"], len(pos_idx))
        picked.update(rng.choice(pos_idx, size=n, replace=False))
    clean_idx = df.index[is_clean].difference(list(picked))
    picked.update(rng.choice(clean_idx, size=min(cfg["clean_samples"], len(clean_idx)), replace=False))

    out = df.loc[sorted(picked), ["text"] + list(rules)].copy()
    out["id"] = ["civil_" + str(i) for i in out.index]
    out["source"] = "civil_comments"
    for name in labels:
        if name not in rules:
            out[name] = UNKNOWN
    return out


def phase1_frame(labels, input_csv):
    """Phase 1 tweets + measuring-hate-speech, human-annotated for hate speech only."""
    df = pd.read_csv(input_csv)
    out = pd.DataFrame({"id": df["id"], "text": df["text"], "source": df["source"]})
    for name in labels:
        out[name] = df["human_label"].astype(int) if name == "hate_speech" else UNKNOWN
    return out


def sms_spam_frame(labels):
    """SMS Spam Collection, annotated for spam only."""
    print("Loading codesignal/sms-spam-collection...", flush=True)
    df = load_dataset("codesignal/sms-spam-collection", split="train").to_pandas()
    out = pd.DataFrame({"id": ["sms_" + str(i) for i in df.index], "text": df["message"], "source": "sms_spam"})
    for name in labels:
        out[name] = (df["label"] == "spam").astype(int) if name == "spam" else UNKNOWN
    return out


def self_harm_frame(labels, cfg, seed):
    """Reddit posts labelled by subreddit (r/SuicideWatch vs r/teenagers). Provisional, noisy labels."""
    print("Loading Ram07/Detection-for-Suicide...", flush=True)
    df = load_dataset("Ram07/Detection-for-Suicide", split="train").to_pandas()
    df = df.drop_duplicates(subset=["text"])
    pos = df[df["class"] == "suicide"].sample(n=cfg["samples_per_class"], random_state=seed)
    neg = df[df["class"] == "non-suicide"].sample(n=cfg["samples_per_class"], random_state=seed)
    df = pd.concat([pos, neg])
    out = pd.DataFrame({"id": ["reddit_" + str(i) for i in df.index], "text": df["text"], "source": "reddit_suicide"})
    for name in labels:
        out[name] = (df["class"] == "suicide").astype(int) if name == "self_harm" else UNKNOWN
    return out


def assign_splits(df, train_split, val_split, seed):
    """80/10/10 split, stratified by source and by whether any label is positive."""
    strata = df["source"] + "_" + df["any_positive"].astype(str)
    train_df, temp_df = train_test_split(df, test_size=1 - train_split, random_state=seed, stratify=strata)
    temp_strata = temp_df["source"] + "_" + temp_df["any_positive"].astype(str)
    val_frac = val_split / (1 - train_split)
    val_df, test_df = train_test_split(temp_df, test_size=1 - val_frac, random_state=seed, stratify=temp_strata)
    df.loc[train_df.index, "split"] = "train"
    df.loc[val_df.index, "split"] = "val"
    df.loc[test_df.index, "split"] = "test"
    return df


def print_summary(df, labels):
    print("\n--- Rows per source ---", flush=True)
    print(df["source"].value_counts().to_string(), flush=True)
    print("\n--- Label counts (positive / negative / unknown) ---", flush=True)
    for name in labels:
        counts = df[name].value_counts()
        print(f"{name:<16}{counts.get(1, 0):>8}{counts.get(0, 0):>8}{counts.get(UNKNOWN, 0):>8}", flush=True)
    print("\n--- Rows per split ---", flush=True)
    print(df["split"].value_counts().to_string(), flush=True)


def build_multilabel(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    data_cfg = config["data"]
    ml_cfg = config["multilabel"]
    labels = data_cfg["labels"]
    seed = data_cfg["seed"]

    frames = [
        phase1_frame(labels, data_cfg["input_csv"]),
        civil_comments_frame(labels, ml_cfg["civil_comments"], seed),
        sms_spam_frame(labels),
        self_harm_frame(labels, ml_cfg["self_harm"], seed),
    ]
    df = pd.concat(frames, ignore_index=True)
    initial_count = len(df)

    # Same cleaning rules as Phase 1: drop duplicate texts and texts under 3 words
    df["text"] = df["text"].astype(str).str.strip()
    df = df.drop_duplicates(subset=["text"])
    print(f"\nRemoved {initial_count - len(df)} duplicate texts across sources.", flush=True)
    before_short = len(df)
    df = df[df["text"].str.split().str.len() >= 3].copy()
    print(f"Removed {before_short - len(df)} short texts (< 3 words).", flush=True)

    df["any_positive"] = (df[labels] == 1).any(axis=1).astype(int)
    df = assign_splits(df.reset_index(drop=True), data_cfg["train_split"], data_cfg["val_split"], seed)

    out_path = ml_cfg["output_csv"]
    df = df[["id", "text", "source", "split"] + labels]
    df.to_csv(out_path, index=False, encoding="utf-8")

    print_summary(df, labels)
    print(f"\nSaved {len(df)} records to {out_path}", flush=True)


if __name__ == "__main__":
    build_multilabel()
