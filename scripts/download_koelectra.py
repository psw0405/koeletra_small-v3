import argparse
import json
from pathlib import Path

from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer

from ner_labels import DATASET_LABEL_ALIASES, LABELS, build_bio_labels


def write_lines(path: Path, rows):
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download koELECTRA-small and initialize token classification head.")
    parser.add_argument("--base_model", default="monologg/koelectra-small-v3-discriminator")
    parser.add_argument("--output_dir", default="models/koelectra-small-initial")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bio_labels = build_bio_labels(LABELS)
    label2id = {label: idx for idx, label in enumerate(bio_labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    print(f"[1/3] Loading tokenizer from {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)

    print(f"[2/3] Loading base model and attaching token-classification head ({len(bio_labels)} labels)")
    config = AutoConfig.from_pretrained(args.base_model)
    config.num_labels = len(bio_labels)
    config.label2id = label2id
    config.id2label = id2label

    model = AutoModelForTokenClassification.from_pretrained(
        args.base_model,
        config=config,
        ignore_mismatched_sizes=True,
    )

    print(f"[3/3] Saving model artifacts to {out_dir}")
    tokenizer.save_pretrained(out_dir)
    model.save_pretrained(out_dir)

    write_lines(out_dir / "base_labels.txt", LABELS)
    write_lines(out_dir / "bio_labels.txt", bio_labels)
    (out_dir / "dataset_label_aliases.json").write_text(
        json.dumps(DATASET_LABEL_ALIASES, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Done.")
    print(f"- base label count: {len(LABELS)}")
    print(f"- BIO label count: {len(bio_labels)}")


if __name__ == "__main__":
    main()
