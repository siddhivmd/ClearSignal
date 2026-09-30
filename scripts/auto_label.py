import json
import os
from transformers import pipeline


def auto_label_data():
    """Auto-label texts in data/to_label.json using pre-trained RoBERTa model."""
    input_path = os.path.join("data", "to_label.json")
    output_path = os.path.join("data", "auto_labelled.json")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"{input_path} not found. Run prepare_data.py first.")

    with open(input_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    print("Loading cardiffnlp/twitter-roberta-base-hate-latest classifier...", flush=True)
    classifier = pipeline(
        "text-classification",
        model="cardiffnlp/twitter-roberta-base-hate-latest",
        top_k=None,
    )

    results = []
    print(f"Auto-labelling {len(items)} items in batches...", flush=True)
    batch_size = 64
    for i in range(0, len(items), batch_size):
        batch_items = items[i : i + batch_size]
        texts = [it.get("text", "")[:512] for it in batch_items]
        try:
            batch_scores = classifier(texts)
            for item, scores in zip(batch_items, batch_scores):
                hate_score = 0.0
                for s in scores:
                    label_name = str(s.get("label", "")).upper()
                    if label_name in ["HATE", "LABEL_1"] or "HATE" in label_name:
                        if label_name != "NOT-HATE" and label_name != "NOT_HATE":
                            hate_score = s.get("score", 0.0)
                            break

                if hate_score > 0.5:
                    labels = ["hate_speech", "toxic"]
                else:
                    labels = ["clean"]

                item_res = {
                    "id": item.get("id"),
                    "text": item.get("text", ""),
                    "source": item.get("source"),
                    "human_label": item.get("human_label"),
                    "labels": labels,
                    "hate_score": round(hate_score, 4),
                    "confidence": 2,
                }
                results.append(item_res)
        except Exception as e:
            print(f"Error in batch auto-labelling: {e}", flush=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Auto-labelling complete. Saved {len(results)} items to {output_path}", flush=True)


if __name__ == "__main__":
    auto_label_data()
