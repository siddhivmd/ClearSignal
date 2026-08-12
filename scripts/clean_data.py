import json
import os
import pandas as pd


def clean_data():
    """Deduplicate, clean short texts, and export custom_labels_clean.csv."""
    input_path = os.path.join("data", "auto_labelled.json")
    output_path = os.path.join("data", "custom_labels_clean.csv")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"{input_path} not found. Run auto_label.py first.")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    initial_count = len(df)
    print(f"Loaded {initial_count} items from {input_path}")

    # 1. Remove duplicate texts
    df = df.drop_duplicates(subset=["text"]).copy()
    after_dedup = len(df)
    print(f"Removed {initial_count - after_dedup} duplicates.")

    # 2. Remove short texts (< 3 words)
    df["word_count"] = df["text"].apply(lambda t: len(str(t).strip().split()))
    df = df[df["word_count"] >= 3].copy()
    after_short_filt = len(df)
    print(f"Removed {after_dedup - after_short_filt} short texts (< 3 words).")

    # Format labels column
    if "labels" in df.columns:
        df["labels"] = df["labels"].apply(
            lambda l: ",".join(l) if isinstance(l, list) else str(l)
        )

    cols = ["id", "text", "labels", "hate_score", "confidence"]
    available_cols = [c for c in cols if c in df.columns]
    clean_df = df[available_cols]

    clean_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Saved {len(clean_df)} cleaned records to {output_path}")


if __name__ == "__main__":
    clean_data()
