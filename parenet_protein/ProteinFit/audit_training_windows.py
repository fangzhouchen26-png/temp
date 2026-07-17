"""Audit oracle crops at the exact coarse/fine hierarchy used by PARE-Net."""

import argparse

import torch

from config import make_cfg
from dataset import ProteinPairDataset, _dataset_kwargs
from pareconv.modules.ops import index_select, point_to_node_partition
from pareconv.modules.registration import get_node_correspondences
from pareconv.utils.data import build_dataloader_stack_mode, registration_collate_fn_stack_mode
from pareconv.utils.torch import to_cuda


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", choices=["train", "val"], default="train")
    parser.add_argument("--augmentation-repeats", type=int, default=3)
    parser.add_argument("--max-failures", type=int, default=20)
    return parser


def make_dataset(cfg, subset, seed):
    is_train = subset == "train"
    return ProteinPairDataset(
        subset=subset,
        point_limit=cfg.train.point_limit if is_train else cfg.test.point_limit,
        use_augmentation=True,
        deterministic_augmentation=True,
        augmentation_seed=seed,
        augmentation_noise=(
            cfg.train.augmentation_noise if is_train else cfg.test.augmentation_noise
        ),
        augmentation_translation=(
            cfg.train.augmentation_translation
            if is_train
            else cfg.test.augmentation_translation
        ),
        source_keep_ratio=cfg.train.source_keep_ratio if is_train else 1.0,
        crop_mode=(
            cfg.crop.train_mode if is_train else cfg.crop.val_mode
        ) if cfg.crop.enabled else "none",
        **_dataset_kwargs(cfg),
    )


def make_loader(cfg, dataset):
    return build_dataloader_stack_mode(
        dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.num_neighbors,
        cfg.backbone.subsample_ratio,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        precompute_data=True,
    )


@torch.no_grad()
def count_valid_gt_patches(cfg, data_dict):
    points = data_dict["points"]
    lengths = data_dict["lengths"]
    points_c = points[-1]
    points_f = points[1]
    ref_length_c = lengths[-1][0].item()
    ref_length_f = lengths[1][0].item()

    ref_points_c = points_c[:ref_length_c]
    src_points_c = points_c[ref_length_c:]
    ref_points_f = points_f[:ref_length_f]
    src_points_f = points_f[ref_length_f:]

    _, ref_node_masks, ref_knn_indices, ref_knn_masks = point_to_node_partition(
        ref_points_f, ref_points_c, cfg.model.num_points_in_patch
    )
    _, src_node_masks, src_knn_indices, src_knn_masks = point_to_node_partition(
        src_points_f, src_points_c, cfg.model.num_points_in_patch
    )

    ref_padded = torch.cat([ref_points_f, torch.zeros_like(ref_points_f[:1])], dim=0)
    src_padded = torch.cat([src_points_f, torch.zeros_like(src_points_f[:1])], dim=0)
    ref_knn_points = index_select(ref_padded, ref_knn_indices, dim=0)
    src_knn_points = index_select(src_padded, src_knn_indices, dim=0)

    _, overlaps = get_node_correspondences(
        ref_points_c,
        src_points_c,
        ref_knn_points,
        src_knn_points,
        data_dict["transform"],
        cfg.model.ground_truth_matching_radius,
        ref_masks=ref_node_masks,
        src_masks=src_node_masks,
        ref_knn_masks=ref_knn_masks,
        src_knn_masks=src_knn_masks,
    )
    return int((overlaps > cfg.coarse_matching.overlap_threshold).sum().item())


def main():
    args = make_parser().parse_args()
    cfg = make_cfg()
    failures = []
    total = 0
    minimum = None

    for repeat in range(args.augmentation_repeats):
        dataset = make_dataset(cfg, args.subset, cfg.seed + 1000000 * repeat)
        loader = make_loader(cfg, dataset)
        for data_dict in loader:
            data_dict = to_cuda(data_dict)
            count = count_valid_gt_patches(cfg, data_dict)
            total += 1
            minimum = count if minimum is None else min(minimum, count)
            if count == 0:
                failures.append(
                    f"{data_dict['case_id']}/chain{data_dict['chain_id']} repeat={repeat}"
                )
                if len(failures) >= args.max_failures:
                    break
        if len(failures) >= args.max_failures:
            break

    print(f"audited_samples={total}")
    print(f"minimum_valid_gt_patches={minimum}")
    print(f"failures={len(failures)}")
    for item in failures:
        print(f"FAIL {item}")
    if failures:
        raise SystemExit(2)
    print("PASS: every audited oracle crop has at least one valid GT coarse patch.")


if __name__ == "__main__":
    main()
