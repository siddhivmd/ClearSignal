import json
import os
from datasets import load_dataset
import pandas as pd


def prepare_data():
    """Download raw datasets from HuggingFace, deduplicate by text, and compile to_label.json.

    Each record keeps the human annotation from its source dataset as `human_label`
    (1 = hateful, 0 = clean) so it can be used as ground truth downstream.
    """
    os.makedirs("data", exist_ok=True)
    combined_data = []

    # 1. Download tweets dataset (500 hate, 500 clean)
    print("Loading tweets-hate-speech-detection/tweets_hate_speech_detection dataset...", flush=True)
    try:
        tweets_ds = load_dataset("tweets-hate-speech-detection/tweets_hate_speech_detection", split="train")
        df_tweets = tweets_ds.to_pandas().drop_duplicates(subset=["tweet"]).sample(frac=1, random_state=42)
        tweets_records = df_tweets.to_dict(orient="records")

        hate_tweets = [
            {"id": f"tweet_{i}", "text": row["tweet"], "source": "tweets_hate", "human_label": 1}
            for i, row in enumerate(tweets_records)
            if row["label"] == 1
        ][:500]
        clean_tweets = [
            {"id": f"tweet_{i}", "text": row["tweet"], "source": "tweets_clean", "human_label": 0}
            for i, row in enumerate(tweets_records)
            if row["label"] == 0
        ][:500]
        combined_data.extend(hate_tweets)
        combined_data.extend(clean_tweets)
    except Exception as e:
        print(f"Error loading tweets dataset: {e}", flush=True)

    # 2. Download measuring hate speech dataset (500 hate, 500 clean)
    print("Loading ucberkeley-dlab/measuring-hate-speech dataset...")
    try:
        mhs_ds = load_dataset("ucberkeley-dlab/measuring-hate-speech", split="train")
        df_mhs = mhs_ds.to_pandas().drop_duplicates(subset=["text"]).sample(frac=1, random_state=42)
        mhs_records = df_mhs.to_dict(orient="records")

        hate_mhs = [
            {"id": f"mhs_{i}", "text": row["text"], "source": "mhs_hate", "human_label": 1}
            for i, row in enumerate(mhs_records)
            if row.get("hate_speech_score", 0) > 0.5
        ][:500]
        clean_mhs = [
            {"id": f"mhs_{i}", "text": row["text"], "source": "mhs_clean", "human_label": 0}
            for i, row in enumerate(mhs_records)
            if row.get("hate_speech_score", 0) < 0.2
        ][:500]
        combined_data.extend(hate_mhs)
        combined_data.extend(clean_mhs)
    except Exception as e:
        print(f"Error loading measuring hate speech dataset: {e}")

    out_path = os.path.join("data", "to_label.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(combined_data, f, indent=2, ensure_ascii=False)

    print(f"Successfully prepared {len(combined_data)} examples at {out_path}")


if __name__ == "__main__":
    prepare_data()
