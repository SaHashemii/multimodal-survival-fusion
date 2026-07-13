#!/usr/bin/env python3
"""Create stratified cross-validation fold assignments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mm_survival.data.labels import load_labels
from mm_survival.training.cross_validation import make_fold_assignments
from mm_survival.utils.config import load_yaml, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create stratified CV fold assignments from labels.")
    parser.add_argument("--data", type=Path, required=True, help="Path to data config YAML.")
    parser.add_argument("--output", type=Path, default=None, help="Output CSV path. Defaults to outputs/fold_assignments.csv.")
    parser.add_argument("--n-splits", type=int, default=None, help="Override number of CV folds.")
    parser.add_argument("--seed", type=int, default=None, help="Override CV seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.data)
    data_cfg = config.get("data", {})
    cv_cfg = config.get("cv", {})
    output_cfg = config.get("outputs", {})

    if "root" not in data_cfg:
        raise ValueError("Data config must define data.root")
    if "labels" not in data_cfg:
        raise ValueError("Data config must define data.labels")

    data_root = Path(data_cfg["root"]).expanduser()
    labels_path = resolve_path(data_root, data_cfg["labels"])
    labels = load_labels(labels_path)
    sample_ids = labels.index.astype(str).tolist()

    n_splits = args.n_splits if args.n_splits is not None else int(cv_cfg.get("n_splits", 5))
    seed = args.seed if args.seed is not None else int(cv_cfg.get("seed", 42))
    folds = make_fold_assignments(sample_ids, labels, n_splits=n_splits, seed=seed)

    if args.output is not None:
        output_path = args.output
    else:
        output_root = resolve_path(REPO_ROOT, output_cfg.get("root", "outputs"))
        output_path = output_root / "fold_assignments.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    folds.to_csv(output_path, index=False)

    print(f"Wrote {len(folds)} fold assignments to {output_path}")
    print(f"n_splits={n_splits} seed={seed}")


if __name__ == "__main__":
    main()
