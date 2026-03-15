import argparse
import shutil
from pathlib import Path

from optimum.exporters.onnx import main_export


def main() -> None:
    parser = argparse.ArgumentParser(description="Export token-classification model to ONNX.")
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    main_export(
        model_name_or_path=args.model_dir,
        output=output_dir,
        task="token-classification",
        opset=args.opset,
    )

    for file_name in ["bio_labels.txt", "base_labels.txt", "dataset_label_aliases.json"]:
        src = Path(args.model_dir) / file_name
        if src.exists():
            shutil.copy2(src, output_dir / file_name)

    print(f"ONNX export complete: {output_dir}")


if __name__ == "__main__":
    main()
