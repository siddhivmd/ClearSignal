import os
import csv
import json
import yaml
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

from train import set_seed

# Label value -1 = unknown (the source never annotated it). Those entries are masked out of the loss and metrics.
UNKNOWN = -1


def masked_bce_loss(logits, targets):
    """Binary cross-entropy per label, ignoring unknown (-1) targets."""
    mask = (targets != UNKNOWN).float()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets.clamp(min=0).float(), reduction="none")
    return (loss * mask).sum() / mask.sum().clamp(min=1)


def subsample(df, max_rows, seed):
    """Proportional sample within each source, so small sources (e.g. SMS spam) stay represented."""
    if not max_rows or len(df) <= max_rows:
        return df
    frac = max_rows / len(df)
    return df.groupby("source", group_keys=False).sample(frac=frac, random_state=seed)


def make_loader(df, labels, tokenizer, max_length, batch_size, shuffle):
    rows = list(zip(df["text"].astype(str), df[labels].values.tolist()))

    def collate(batch):
        texts, targets = zip(*batch)
        enc = tokenizer(list(texts), truncation=True, max_length=max_length, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor(targets, dtype=torch.long)
        return enc

    return DataLoader(rows, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


def predict(model, loader, device):
    """Return (sigmoid probabilities, targets, mean masked loss) for a loader."""
    model.eval()
    all_probs, all_targets, total_loss = [], [], 0.0
    with torch.no_grad():
        for batch in loader:
            targets = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            total_loss += masked_bce_loss(logits, targets).item()
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    return np.concatenate(all_probs), np.concatenate(all_targets), total_loss / len(loader)


def label_metrics(probs, targets, labels, threshold=0.5):
    """Per-label precision / recall / F1 on rows where the label is known, plus macro F1."""
    per_label = {}
    for j, name in enumerate(labels):
        known = targets[:, j] != UNKNOWN
        y_true, y_pred = targets[known, j], (probs[known, j] >= threshold).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        per_label[name] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "positives": int(y_true.sum()),
            "known_rows": int(known.sum()),
        }
    macro_f1 = round(float(np.mean([m["f1"] for m in per_label.values()])), 4)
    return per_label, macro_f1


def per_source_f1(probs, targets, sources, labels, threshold=0.5):
    """F1 for each label within each source that annotated it, e.g. toxic on Civil Comments vs on MHS."""
    result = {}
    for j, name in enumerate(labels):
        known = targets[:, j] != UNKNOWN
        result[name] = {}
        for src in np.unique(sources[known]):
            rows = known & (sources == src)
            y_true, y_pred = targets[rows, j], (probs[rows, j] >= threshold).astype(int)
            _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
            result[name][src] = {"f1": round(f1, 4), "positives": int(y_true.sum()), "rows": int(rows.sum())}
    return result


def out_of_source_flag_rates(probs, targets, sources, labels, threshold=0.5):
    """Share of rows flagged for each label, on sources that never annotated that label.

    A spam head trained only on SMS should flag almost no Civil Comments; a high rate here
    means the head learned the text style of its source rather than the label.
    """
    rates = {}
    for j, name in enumerate(labels):
        unknown = targets[:, j] == UNKNOWN
        if not unknown.any():
            continue
        flagged = pd.Series(probs[unknown, j] >= threshold).groupby(sources[unknown]).mean()
        rates[name] = {src: round(float(rate), 4) for src, rate in flagged.items()}
    return rates


def print_metrics(title, per_label, macro_f1):
    print(f"\n--- {title} ---", flush=True)
    print(f"{'label':<16}{'prec':>8}{'recall':>8}{'f1':>8}{'pos':>8}{'known':>8}", flush=True)
    for name, m in per_label.items():
        print(f"{name:<16}{m['precision']:>8.4f}{m['recall']:>8.4f}{m['f1']:>8.4f}{m['positives']:>8}{m['known_rows']:>8}", flush=True)
    print(f"Macro F1: {macro_f1:.4f}", flush=True)


def save_probs(path, df, probs, labels):
    out = df[["id", "source"] + labels].copy().reset_index(drop=True)
    for j, name in enumerate(labels):
        out[f"prob_{name}"] = probs[:, j].round(4)
    out.to_csv(path, index=False, encoding="utf-8")


def train_multilabel(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    labels = config["data"]["labels"]
    seed = config["data"]["seed"]
    cfg = config["multilabel_training"]
    set_seed(seed)

    df = pd.read_csv(config["multilabel"]["output_csv"])
    train_df = subsample(df[df["split"] == "train"], cfg["max_train_rows"], seed)
    val_df = subsample(df[df["split"] == "val"], cfg["max_eval_rows"], seed)
    test_df = subsample(df[df["split"] == "test"], cfg["max_eval_rows"], seed)
    print(f"Labels: {labels}", flush=True)
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)} rows", flush=True)

    model_name = cfg["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        problem_type="multi_label_classification",
        id2label=dict(enumerate(labels)),
        label2id={name: i for i, name in enumerate(labels)},
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}", flush=True)

    max_length, batch_size = cfg["max_length"], cfg["batch_size"]
    train_loader = make_loader(train_df, labels, tokenizer, max_length, batch_size, shuffle=True)
    val_loader = make_loader(val_df, labels, tokenizer, max_length, batch_size, shuffle=False)
    test_loader = make_loader(test_df, labels, tokenizer, max_length, batch_size, shuffle=False)

    epochs = cfg["epochs"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

    save_dir = cfg["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "train_log.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_macro_f1"] + [f"val_f1_{n}" for n in labels])

    best_macro_f1 = -1.0
    print("\n--- Starting Multi-Label Training ---", flush=True)
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            targets = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            loss = masked_bce_loss(model(**batch).logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_train_loss += loss.item()
            if step % 50 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} | loss {total_train_loss / step:.4f}", flush=True)
        avg_train_loss = total_train_loss / len(train_loader)

        val_probs, val_targets, val_loss = predict(model, val_loader, device)
        per_label, macro_f1 = label_metrics(val_probs, val_targets, labels)
        print(f"\nEpoch {epoch}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}", flush=True)
        print_metrics(f"Validation, epoch {epoch}", per_label, macro_f1)

        with open(log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [epoch, round(avg_train_loss, 4), round(val_loss, 4), macro_f1] + [per_label[n]["f1"] for n in labels]
            )

        # Checkpoint on best validation macro F1
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            print(f"--> New best validation macro F1: {macro_f1:.4f}. Saving to {save_dir}...", flush=True)
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            save_probs(os.path.join(save_dir, "val_probs.csv"), val_df, val_probs, labels)
            with open(os.path.join(save_dir, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump({"epoch": epoch, "val_loss": round(val_loss, 4), "val_macro_f1": macro_f1, "val_per_label": per_label}, f, indent=2)

    # Final evaluation of the best checkpoint on the held-out test set
    print("\n--- Evaluating Best Checkpoint on Held-Out Test Set ---", flush=True)
    best_model = AutoModelForSequenceClassification.from_pretrained(save_dir).to(device)
    test_probs, test_targets, test_loss = predict(best_model, test_loader, device)
    per_label, macro_f1 = label_metrics(test_probs, test_targets, labels)
    flag_rates = out_of_source_flag_rates(test_probs, test_targets, test_df["source"].values, labels)
    by_source = per_source_f1(test_probs, test_targets, test_df["source"].values, labels)
    print_metrics("Test set", per_label, macro_f1)

    print("\n--- Test F1 per source (f1 / positives / rows) ---", flush=True)
    for name, sources in by_source.items():
        print(f"{name:<16}" + "  ".join(f"{src}={m['f1']:.3f}/{m['positives']}/{m['rows']}" for src, m in sources.items()), flush=True)

    print("\n--- Out-of-source flag rate (should be low) ---", flush=True)
    for name, rates in flag_rates.items():
        print(f"{name:<16}" + "  ".join(f"{src}={rate:.3f}" for src, rate in rates.items()), flush=True)

    save_probs(os.path.join(save_dir, "test_probs.csv"), test_df, test_probs, labels)
    with open(os.path.join(save_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "labels": labels,
                "test_rows": len(test_df),
                "test_loss": round(test_loss, 4),
                "test_macro_f1": macro_f1,
                "test_per_label": per_label,
                "test_f1_per_source": by_source,
                "out_of_source_flag_rate": flag_rates,
            },
            f,
            indent=2,
        )
    print(f"\nSaved test metrics and probabilities to {save_dir}", flush=True)


if __name__ == "__main__":
    train_multilabel()
