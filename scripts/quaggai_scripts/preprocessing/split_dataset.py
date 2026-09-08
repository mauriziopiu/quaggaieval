# split_dataset.py
import json, random
from pathlib import Path
import argparse
import sys

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Match image/mask pairs by stem and build a structured dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset", required=True, type=Path, metavar="PATH",
        help="Folder containing dataset.json.",
    )
    parser.add_argument(
        "--split-ratio", type=float, default=0.8,
        help="Split ratio as fraction in [0, 1]. Default = 0.8",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible datast splits. Default = 42",
    )
    return parser.parse_args()

def validate_args(args: argparse.Namespace) -> None:
    for attr, label in [("dataset", "--dataset")]:
        path = getattr(args, attr)
        if not path.is_dir():
            sys.exit(f"Error: {label} path '{path}' is not a directory or does not exist.")


def run(dataset_dir: Path, split_ratio: float, seed: int) -> dict[str, int]:
    """Split dataset stems into train/val and write train.txt + val.txt.

    Returns a counts dict with keys ``train`` and ``val``.
    """
    random.seed(seed)
    manifest = json.loads(Path.joinpath(dataset_dir, "dataset.json").read_text())
    stems = [e["stem"] for e in manifest["entries"]]
    random.shuffle(stems)

    split = int(split_ratio * len(stems))
    train_stems, val_stems = stems[:split], stems[split:]

    Path.joinpath(dataset_dir, "train.txt").write_text("\n".join(train_stems))
    Path.joinpath(dataset_dir, "val.txt").write_text("\n".join(val_stems))
    return {"train": len(train_stems), "val": len(val_stems)}


def main() -> None:
    args = parse_args()
    validate_args(args)

    print(f"Dataset folder  : {args.dataset}")
    print(f"Split Ratio     : Train={args.split_ratio} / Val={1.0-args.split_ratio}")
    print(f"Random Seed     : {args.seed}")

    counts = run(args.dataset, args.split_ratio, args.seed)
    print(f"Train: {counts['train']}  Val: {counts['val']}")

if __name__ == "__main__":
    main()