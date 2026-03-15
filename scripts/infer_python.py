import argparse
import json
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer


def resolve_label(id2label: Dict[Any, str], idx: int) -> str:
    if idx in id2label:
        return id2label[idx]
    return id2label.get(str(idx), f"LABEL_{idx}")


def close_entity(buffer: Optional[Dict[str, Any]], text: str, out: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not buffer:
        return None
    start = int(buffer["start"])
    end = int(buffer["end"])
    if start < end:
        out.append(
            {
                "start": start,
                "end": end,
                "label": buffer["label"],
                "text": text[start:end],
                "score": round(buffer["score_sum"] / max(buffer["token_count"], 1), 4),
            }
        )
    return None


def decode_bio_entities(
    text: str,
    tokens: List[str],
    offsets: List[List[int]],
    special_mask: List[int],
    pred_labels: List[str],
    pred_scores: List[float],
) -> List[Dict[str, Any]]:
    entities: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None

    for i, token in enumerate(tokens):
        _ = token
        if special_mask[i] == 1:
            current = close_entity(current, text, entities)
            continue

        start, end = offsets[i]
        if start >= end:
            current = close_entity(current, text, entities)
            continue

        label = pred_labels[i]
        score = float(pred_scores[i])

        if label == "O":
            current = close_entity(current, text, entities)
            continue

        if "-" not in label:
            current = close_entity(current, text, entities)
            continue

        prefix, entity_type = label.split("-", 1)
        if prefix == "B":
            current = close_entity(current, text, entities)
            current = {
                "label": entity_type,
                "start": start,
                "end": end,
                "score_sum": score,
                "token_count": 1,
            }
            continue

        if prefix == "I":
            if current and current["label"] == entity_type and start <= current["end"]:
                current["end"] = end
                current["score_sum"] += score
                current["token_count"] += 1
            else:
                current = close_entity(current, text, entities)
                current = {
                    "label": entity_type,
                    "start": start,
                    "end": end,
                    "score_sum": score,
                    "token_count": 1,
                }
            continue

        current = close_entity(current, text, entities)

    close_entity(current, text, entities)
    return entities


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Python inference for koELECTRA-small token classification model.")
    parser.add_argument("--model_dir", default="models/koelectra-small-initial")
    parser.add_argument("--text", required=True)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    encoded = tokenizer(
        args.text,
        return_tensors="pt",
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
        truncation=True,
        max_length=args.max_length,
    )

    offsets = encoded.pop("offset_mapping")[0].tolist()
    special_mask = encoded.pop("special_tokens_mask")[0].tolist()

    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        pred_scores, pred_ids = torch.max(probs, dim=-1)

    input_ids = encoded["input_ids"][0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    pred_ids_list = pred_ids[0].tolist()
    pred_scores_list = pred_scores[0].tolist()
    pred_labels = [resolve_label(model.config.id2label, idx) for idx in pred_ids_list]

    entities = decode_bio_entities(
        text=args.text,
        tokens=tokens,
        offsets=offsets,
        special_mask=special_mask,
        pred_labels=pred_labels,
        pred_scores=pred_scores_list,
    )

    print("[Token Predictions]")
    for token, label, score, (start, end), is_special in zip(
        tokens,
        pred_labels,
        pred_scores_list,
        offsets,
        special_mask,
    ):
        if is_special == 1:
            continue
        span_text = args.text[start:end] if start < end else ""
        print(f"{token:12s} {label:24s} {score:.4f}  [{start},{end}]  {span_text}")

    print("\n[Decoded Entities]")
    print(json.dumps(entities, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
