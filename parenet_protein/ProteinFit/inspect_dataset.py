import argparse
import json
from pathlib import Path

import numpy as np

from dataset import (
    ProteinPairDataset,
    apply_transform,
    build_case_splits,
    discover_pairs,
    load_density_txt,
    source_overlap,
)


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--point_suffix", default="_2.00.txt")
    parser.add_argument("--min_source_points", type=int, default=128)
    parser.add_argument("--min_target_points", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7351)
    parser.add_argument("--write_splits")
    return parser


def main():
    args = make_parser().parse_args()
    pairs, skipped = discover_pairs(
        args.dataset_root,
        point_suffix=args.point_suffix,
        min_source_points=args.min_source_points,
        min_target_points=args.min_target_points,
    )
    cases = sorted({pair.case_id for pair in pairs})
    splits = build_case_splits(cases, seed=args.seed)

    print(f"valid cases: {len(cases)}")
    print(f"valid chain pairs: {len(pairs)}")
    for subset in ("train", "val", "test"):
        subset_cases = set(splits[subset])
        subset_pairs = [pair for pair in pairs if pair.case_id in subset_cases]
        print(f"{subset}: {len(subset_cases)} cases, {len(subset_pairs)} pairs")

    if pairs:
        src_counts = np.asarray([pair.src_count for pair in pairs])
        ref_counts = np.asarray([pair.ref_count for pair in pairs])
        print(
            "source points min/median/max: "
            f"{src_counts.min()}/{int(np.median(src_counts))}/{src_counts.max()}"
        )
        print(
            "target points min/median/max: "
            f"{ref_counts.min()}/{int(np.median(ref_counts))}/{ref_counts.max()}"
        )

        first = pairs[0]
        ref = load_density_txt(first.ref_path)["points"]
        src = load_density_txt(first.src_path)["points"]
        identity = np.eye(4, dtype=np.float32)
        print(
            f"first-pair identity overlap@4A: "
            f"{source_overlap(ref, src, identity, radius=4.0):.4f}"
        )

        probe = ProteinPairDataset(
            args.dataset_root,
            subset="train",
            point_suffix=args.point_suffix,
            split_seed=args.seed,
            min_source_points=args.min_source_points,
            min_target_points=args.min_target_points,
            use_augmentation=True,
            deterministic_augmentation=True,
            augmentation_noise=0.0,
            source_keep_ratio=1.0,
        )[0]
        recovered = apply_transform(probe["src_points"], probe["transform"])
        distances = np.linalg.norm(
            recovered - load_density_txt(
                next(pair.src_path for pair in pairs if pair.case_id == probe["case_id"] and pair.chain_id == probe["chain_id"])
            )["points"],
            axis=1,
        )
        print(f"transform inverse max error: {distances.max():.6f} A")

    if skipped:
        print(f"skipped entries: {len(skipped)}")
        for message in skipped[:30]:
            print(f"  - {message}")
        if len(skipped) > 30:
            print(f"  ... {len(skipped) - 30} more")

    if args.write_splits:
        output = Path(args.write_splits)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(splits, indent=2) + "\n", encoding="utf-8")
        print(f"wrote split file: {output}")


if __name__ == "__main__":
    main()
