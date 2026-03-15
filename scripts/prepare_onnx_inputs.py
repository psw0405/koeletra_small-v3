import argparse
from pathlib import Path

from transformers import AutoTokenizer


def write_int_tensor(path: Path, values):
    path.write_text(" ".join(str(int(v)) for v in values) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ONNX input tensors from text.")
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_length", type=int, default=64)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    encoded = tokenizer(
        args.text,
        truncation=True,
        max_length=args.max_length,
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    token_type_ids = encoded.get("token_type_ids", [0] * len(input_ids))
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    write_int_tensor(out_dir / "input_ids.txt", input_ids)
    write_int_tensor(out_dir / "attention_mask.txt", attention_mask)
    write_int_tensor(out_dir / "token_type_ids.txt", token_type_ids)
    (out_dir / "tokens.txt").write_text("\n".join(tokens) + "\n", encoding="utf-8")
    (out_dir / "text.txt").write_text(args.text + "\n", encoding="utf-8")

    print(f"Saved ONNX inputs to {out_dir}")
    print(f"Sequence length: {len(input_ids)}")


if __name__ == "__main__":
    main()
