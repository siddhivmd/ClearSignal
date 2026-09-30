import os
import json
import yaml
import csv
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class BinaryTextDataset(Dataset):
    """PyTorch Dataset for binary text classification."""

    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


def prepare_binary_dataset(input_csv, binary_csv, label_source="human"):
    """Build the binary dataset (flagged=1 vs clean=0).

    label_source="human" uses the source datasets' human annotations as ground truth.
    label_source="auto" uses the RoBERTa auto-labels (the original v1 behaviour).
    The RoBERTa prediction is always kept as `teacher_label` so it can be scored as a baseline.
    """
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} records from {input_csv}", flush=True)

    def map_binary(label_str):
        if not isinstance(label_str, str):
            return 0
        labels_list = [l.strip().lower() for l in label_str.split(",")]
        if "hate_speech" in labels_list or "toxic" in labels_list:
            return 1
        return 0

    df["teacher_label"] = df["labels"].apply(map_binary)
    if label_source == "human":
        if "human_label" not in df.columns:
            raise ValueError(f"{input_csv} has no human_label column. Re-run the Phase 1 pipeline.")
        df["label"] = df["human_label"].astype(int)
    elif label_source == "auto":
        df["label"] = df["teacher_label"]
    else:
        raise ValueError(f"Unknown label_source: {label_source}")
    print(f"Training labels taken from: {label_source}", flush=True)

    binary_df = pd.DataFrame(
        {
            "id": df["id"],
            "text": df["text"],
            "source": df.get("source"),
            "labels_raw": df["labels"],
            "teacher_label": df["teacher_label"],
            "label": df["label"],
        }
    )

    os.makedirs(os.path.dirname(binary_csv), exist_ok=True)
    binary_df.to_csv(binary_csv, index=False, encoding="utf-8")
    print(f"Saved binary dataset to {binary_csv} with distribution:", flush=True)
    print(binary_df["label"].value_counts().to_dict(), flush=True)
    agreement = (binary_df["label"] == binary_df["teacher_label"]).mean()
    print(f"RoBERTa auto-label agreement with training labels: {agreement:.4f}", flush=True)

    return binary_df


def binary_metrics(targets, preds):
    """Accuracy, flagged-class precision/recall/F1, macro F1 and confusion matrix."""
    acc = accuracy_score(targets, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(targets, preds, average="binary", zero_division=0)
    _, _, macro_f1, _ = precision_recall_fscore_support(targets, preds, average="macro", zero_division=0)
    return {
        "accuracy": round(acc, 4),
        "precision_flagged": round(prec, 4),
        "recall_flagged": round(rec, 4),
        "f1_flagged": round(f1, 4),
        "macro_f1": round(macro_f1, 4),
        "confusion_matrix": confusion_matrix(targets, preds, labels=[0, 1]).tolist(),
    }


def train_and_evaluate(config_path="config.yaml"):
    """Main training and evaluation routine."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    seed = config["data"]["seed"]
    set_seed(seed)

    # 1. Prepare binary dataset
    input_csv = config["data"]["input_csv"]
    binary_csv = config["data"]["binary_csv"]
    df = prepare_binary_dataset(input_csv, binary_csv, config["data"].get("label_source", "human"))

    # 2. Stratified train / val / test split (80 / 10 / 10)
    train_df, temp_df = train_test_split(
        df,
        test_size=(config["data"]["val_split"] + config["data"]["test_split"]),
        random_state=seed,
        stratify=df["label"],
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=seed,
        stratify=temp_df["label"],
    )

    print("\n--- Stratified Split Class Distribution ---", flush=True)
    print(f"Train set: {len(train_df)} rows | Labels: {train_df['label'].value_counts().to_dict()}")
    print(f"Val set:   {len(val_df)} rows | Labels: {val_df['label'].value_counts().to_dict()}")
    print(f"Test set:  {len(test_df)} rows | Labels: {test_df['label'].value_counts().to_dict()}\n")

    # 3. Tokenizer & Datasets
    model_name = config["model"]["name"]
    max_length = config["model"]["max_length"]
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

    train_dataset = BinaryTextDataset(train_df["text"], train_df["label"], tokenizer, max_length)
    val_dataset = BinaryTextDataset(val_df["text"], val_df["label"], tokenizer, max_length)
    test_dataset = BinaryTextDataset(test_df["text"], test_df["label"], tokenizer, max_length)

    batch_size = config["training"]["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 4. Model Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    model = DistilBertForSequenceClassification.from_pretrained(
        model_name, num_labels=config["model"]["num_labels"]
    )
    model.to(device)

    epochs = config["training"]["epochs"]
    lr = float(config["training"]["learning_rate"])
    weight_decay = float(config["training"]["weight_decay"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    # 5. Logging Setup
    save_dir = config["training"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    log_csv_path = os.path.join(save_dir, "train_log.csv")

    with open(log_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["epoch", "train_loss", "val_loss", "val_accuracy", "val_precision", "val_recall", "val_f1"]
        )

    best_val_f1 = 0.0

    # 6. Training Loop
    print("\n--- Starting Model Training ---", flush=True)
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0.0
        val_preds, val_targets = [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_val_loss += loss.item()

                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(labels.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = accuracy_score(val_targets, val_preds)
        val_prec, val_rec, val_f1, _ = precision_recall_fscore_support(
            val_targets, val_preds, average="binary", zero_division=0
        )

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val Precision: {val_prec:.4f} | "
            f"Val Recall: {val_rec:.4f} | "
            f"Val F1: {val_f1:.4f}",
            flush=True,
        )

        # Log epoch metrics to CSV
        with open(log_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch,
                    round(avg_train_loss, 4),
                    round(avg_val_loss, 4),
                    round(val_acc, 4),
                    round(val_prec, 4),
                    round(val_rec, 4),
                    round(val_f1, 4),
                ]
            )

        # Checkpoint saving based on Best Validation F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            print(f"--> New best validation F1 score: {best_val_f1:.4f}. Saving model checkpoint to {save_dir}...", flush=True)
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)

            val_metrics = {
                "epoch": epoch,
                "best_val_loss": round(avg_val_loss, 4),
                "best_val_accuracy": round(val_acc, 4),
                "best_val_precision": round(val_prec, 4),
                "best_val_recall": round(val_rec, 4),
                "best_val_f1": round(val_f1, 4),
            }
            with open(os.path.join(save_dir, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump(val_metrics, f, indent=2)

    # 7. Final Test Evaluation
    print("\n--- Evaluating Best Model Checkpoint on Held-Out Test Set ---", flush=True)
    best_model = DistilBertForSequenceClassification.from_pretrained(save_dir)
    best_model.to(device)
    best_model.eval()

    test_preds, test_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = best_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            test_preds.extend(preds)
            test_targets.extend(labels.cpu().numpy())

    model_metrics = binary_metrics(test_targets, test_preds)
    # Baseline: the RoBERTa teacher's own predictions on the same test rows
    teacher_metrics = binary_metrics(test_df["label"].tolist(), test_df["teacher_label"].tolist())

    print("\n==============================================", flush=True)
    print("         HELD-OUT TEST SET EVALUATION        ", flush=True)
    print("==============================================", flush=True)
    print(f"{'Metric':<20}{'DistilBERT':>12}{'RoBERTa':>12}", flush=True)
    for key in ["accuracy", "precision_flagged", "recall_flagged", "f1_flagged", "macro_f1"]:
        print(f"{key:<20}{model_metrics[key]:>12.4f}{teacher_metrics[key]:>12.4f}", flush=True)
    cm = model_metrics["confusion_matrix"]
    print("\nDistilBERT Confusion Matrix (Rows: Actual, Cols: Predicted):", flush=True)
    print(f"[[TN={cm[0][0]}, FP={cm[0][1]}],", flush=True)
    print(f" [FN={cm[1][0]}, TP={cm[1][1]}]]", flush=True)
    print("==============================================\n", flush=True)

    test_results = {
        "label_source": config["data"].get("label_source", "human"),
        "test_size": len(test_df),
        "distilbert": model_metrics,
        "roberta_teacher": teacher_metrics,
    }
    with open(os.path.join(save_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=2)

    print(f"Saved test evaluation metrics to {os.path.join(save_dir, 'test_metrics.json')}", flush=True)


if __name__ == "__main__":
    train_and_evaluate()
