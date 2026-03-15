import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import evaluate
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from ner_labels import LABELS, build_bio_labels, normalize_dataset_label


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def normalize_entities(text: str, entities: List[Dict[str, Any]], dropped: Counter) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    text_len = len(text)

    for ent in entities:
        try:
            start = int(ent["start"])
            end = int(ent["end"])
            raw_label = str(ent["label"])
        except Exception:
            dropped["<invalid_entity_format>"] += 1
            continue

        if end <= start:
            dropped["<invalid_span>"] += 1
            continue

        if start < 0:
            start = 0
        if end > text_len:
            end = text_len
        if start >= text_len:
            dropped["<out_of_range>"] += 1
            continue

        canonical = normalize_dataset_label(raw_label)
        if canonical is None:
            dropped[raw_label] += 1
            continue

        normalized.append({"start": start, "end": end, "label": canonical})

    normalized.sort(key=lambda item: (item["start"], item["end"]))
    return normalized


def normalize_examples(raw_rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Counter]:
    dropped = Counter()
    output_rows: List[Dict[str, Any]] = []

    for row in raw_rows:
        text = row.get("text", "")
        entities = row.get("entities", [])
        if not isinstance(text, str):
            dropped["<invalid_text>"] += 1
            continue
        if not isinstance(entities, list):
            dropped["<invalid_entities>"] += 1
            continue

        normalized_entities = normalize_entities(text, entities, dropped)
        output_rows.append({"text": text, "entities": normalized_entities})

    return output_rows, dropped


def tokenize_and_align_labels(example: Dict[str, Any], tokenizer, label2id: Dict[str, int], max_length: int):
    text = example["text"]
    entities = example["entities"]

    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
    )

    offsets = encoded["offset_mapping"]
    special_tokens_mask = encoded["special_tokens_mask"]

    char_owner = [-1] * len(text)
    for ent_idx, ent in enumerate(entities):
        start = max(0, ent["start"])
        end = min(len(text), ent["end"])
        for pos in range(start, end):
            if char_owner[pos] == -1:
                char_owner[pos] = ent_idx

    labels: List[int] = []
    prev_ent_idx = -1

    for token_idx, (start, end) in enumerate(offsets):
        if special_tokens_mask[token_idx] == 1 or start >= end:
            labels.append(-100)
            prev_ent_idx = -1
            continue

        start = max(start, 0)
        end = min(end, len(text))

        token_owners = [char_owner[pos] for pos in range(start, end) if char_owner[pos] != -1]
        if not token_owners:
            labels.append(label2id["O"])
            prev_ent_idx = -1
            continue

        owner_counts = Counter(token_owners)
        ent_idx = owner_counts.most_common(1)[0][0]
        ent_label = entities[ent_idx]["label"]

        prefix = "I" if ent_idx == prev_ent_idx else "B"
        labels.append(label2id[f"{prefix}-{ent_label}"])
        prev_ent_idx = ent_idx

    encoded["labels"] = labels
    encoded.pop("offset_mapping", None)
    encoded.pop("special_tokens_mask", None)
    return encoded


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune koELECTRA-small for NER token classification.")
    parser.add_argument("--model_dir", default="models/koelectra-small-initial")
    parser.add_argument("--train_file", default="train.jsonl")
    parser.add_argument("--valid_file", default="valid.jsonl")
    parser.add_argument("--output_dir", default="models/koelectra-small-finetuned")
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_valid_samples", type=int, default=0)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)

    train_rows = load_jsonl(Path(args.train_file))
    valid_rows = load_jsonl(Path(args.valid_file))

    if args.max_train_samples > 0:
        train_rows = train_rows[: args.max_train_samples]
    if args.max_valid_samples > 0:
        valid_rows = valid_rows[: args.max_valid_samples]

    train_rows, dropped_train = normalize_examples(train_rows)
    valid_rows, dropped_valid = normalize_examples(valid_rows)

    print(f"Train rows: {len(train_rows)}")
    print(f"Valid rows: {len(valid_rows)}")
    if dropped_train:
        print("Dropped labels/entities (train):", dict(dropped_train))
    if dropped_valid:
        print("Dropped labels/entities (valid):", dict(dropped_valid))

    bio_labels = build_bio_labels(LABELS)
    label2id = {label: idx for idx, label in enumerate(bio_labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_dir,
        num_labels=len(bio_labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    train_ds = Dataset.from_list(train_rows)
    valid_ds = Dataset.from_list(valid_rows)

    train_ds = train_ds.map(
        lambda x: tokenize_and_align_labels(x, tokenizer, label2id, args.max_length),
        remove_columns=train_ds.column_names,
    )
    valid_ds = valid_ds.map(
        lambda x: tokenize_and_align_labels(x, tokenizer, label2id, args.max_length),
        remove_columns=valid_ds.column_names,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    seqeval = evaluate.load("seqeval")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=2)

        true_predictions: List[List[str]] = []
        true_labels: List[List[str]] = []

        for pred_row, label_row in zip(predictions, labels):
            row_preds: List[str] = []
            row_labels: List[str] = []
            for pred_id, label_id in zip(pred_row, label_row):
                if label_id == -100:
                    continue
                row_preds.append(bio_labels[pred_id])
                row_labels.append(bio_labels[label_id])
            true_predictions.append(row_preds)
            true_labels.append(row_labels)

        scores = seqeval.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": scores.get("overall_precision", 0.0),
            "recall": scores.get("overall_recall", 0.0),
            "f1": scores.get("overall_f1", 0.0),
            "accuracy": scores.get("overall_accuracy", 0.0),
        }

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        fp16=bool(args.fp16 and torch.cuda.is_available()),
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)

    (out_dir / "base_labels.txt").write_text("\n".join(LABELS) + "\n", encoding="utf-8")
    (out_dir / "bio_labels.txt").write_text("\n".join(bio_labels) + "\n", encoding="utf-8")
    (out_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Training complete.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
