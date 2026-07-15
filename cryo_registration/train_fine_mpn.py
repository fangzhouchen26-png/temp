from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F

from .config import RegistrationConfig, load_config
from .fine_point_matching import FinePointMatcher
from .hierarchical import (
    HierarchicalRegistrationRefiner,
    split_chain_by_principal_axis,
)
from .model import ProteinRegistrationModel, apply_transform
from .train import _chain_subclouds, _move_to_device, build_model
from .train import _seed_everything
from .train_mpn import EarlyStopper, MpnSample, evaluate_mpn_samples
from .training import (
    compute_chain_rmse_angstrom,
    inverse_transform,
    random_rigid_transform,
    select_gt_and_positive_subclouds,
)


def build_hierarchical_refiner(
    config: RegistrationConfig,
    coarse_model: ProteinRegistrationModel,
) -> HierarchicalRegistrationRefiner:
    model = config.model
    point_matcher = None
    if model.use_fine_point_matcher:
        encoder_dim = int(coarse_model.encoder.norm.normalized_shape[0])
        point_matcher = FinePointMatcher(
            shot_dim=model.shot_dim,
            encoder_dim=encoder_dim,
            feature_dim=model.fine_point_feature_dim,
            num_heads=model.fine_point_attention_heads,
            equivariant_channels=model.equivariant_feature_dim,
            query_chunk_size=model.fine_point_attention_query_chunk,
            max_pair_elements=model.max_fine_point_pair_elements,
        )
    return HierarchicalRegistrationRefiner(
        coarse_model,
        coarse_output_topk=model.coarse_output_topk,
        fine_ops_topk=model.fine_ops_topk,
        fine_output_topk=model.fine_output_topk,
        fine_crop_diameter_factor=model.fine_crop_diameter_factor,
        fine_point_cap_factor=model.fine_point_cap_factor,
        fine_max_points_per_patch=model.fine_max_points_per_patch,
        fine_min_valid_points=model.fine_min_valid_points,
        fine_min_half_points=model.fine_min_half_points,
        pair_scoring_mode=model.pair_scoring_mode,
        fine_point_matcher=point_matcher,
        use_equivariant_pose=model.use_equivariant_hypothesis_pose,
        equivariant_feature_dim=model.equivariant_feature_dim,
        equivariant_max_hypotheses=model.equivariant_max_hypotheses,
        equivariant_acceptance_radius_angstrom=model.equivariant_acceptance_radius_angstrom,
    )


def freeze_for_fine_training(refiner: HierarchicalRegistrationRefiner) -> None:
    for parameter in refiner.coarse_model.parameters():
        parameter.requires_grad_(False)
    for parameter in refiner.fine_mpn.parameters():
        parameter.requires_grad_(True)
    refiner.coarse_model.eval()
    refiner.fine_mpn.train()


def summarize_hierarchical_rows(rows: list[dict]) -> dict[str, float | int | None]:
    if not rows:
        return {
            "chains": 0,
            "coarse_top3_region_recall": None,
            "coarse_top1_pose_recall_3A": None,
            "coarse_top3_pose_recall_3A": None,
            "coarse_top1_mean_rmse_A": None,
            "coarse_top3_oracle_mean_rmse_A": None,
            "top1_region_recall": None,
            "top3_region_recall": None,
            "top1_pose_recall_3A": None,
            "top3_pose_recall_3A": None,
            "top1_mean_rmse_A": None,
            "top3_oracle_mean_rmse_A": None,
            "mean_rre_deg": None,
            "mean_rte_A": None,
            "fallback_rate": None,
        }

    def mean(key: str) -> float:
        return float(np.mean([row[key] for row in rows]))

    total_candidates = sum(row["candidate_count"] for row in rows)
    return {
        "chains": len(rows),
        "coarse_top3_region_recall": mean("coarse_top3_region"),
        "coarse_top1_pose_recall_3A": mean("coarse_top1_pose"),
        "coarse_top3_pose_recall_3A": mean("coarse_top3_pose"),
        "coarse_top1_mean_rmse_A": mean("coarse_top1_rmse"),
        "coarse_top3_oracle_mean_rmse_A": mean("coarse_top3_rmse"),
        "top1_region_recall": mean("top1_region"),
        "top3_region_recall": mean("top3_region"),
        "top1_pose_recall_3A": mean("top1_pose"),
        "top3_pose_recall_3A": mean("top3_pose"),
        "top1_mean_rmse_A": mean("top1_rmse"),
        "top3_oracle_mean_rmse_A": mean("top3_rmse"),
        "mean_rre_deg": mean("top1_rre"),
        "mean_rte_A": mean("top1_rte"),
        "fallback_rate": (
            sum(row["fallbacks"] for row in rows) / max(1, total_candidates)
        ),
    }


def hierarchical_selection_metric(metrics: dict) -> float:
    value = metrics.get("top3_oracle_mean_rmse_A")
    if value is None:
        raise ValueError("hierarchical Top-3 RMSE is required for selection")
    return float(value)


@torch.no_grad()
def fine_samples_for_structure(
    refiner: HierarchicalRegistrationRefiner,
    structure: dict,
    generator: torch.Generator,
) -> list[MpnSample]:
    samples: list[MpnSample] = []
    scale = float(structure["normalization"]["scale"])
    refiner.eval()
    for chain_id in sorted(structure["chains"]):
        chain = structure["chains"][chain_id]
        augmentation = random_rigid_transform(
            chain["2.00"].device,
            chain["2.00"].dtype,
            generator,
        )
        ground_truth = inverse_transform(augmentation)
        output = refiner(structure, chain_id, source_transform=augmentation)
        try:
            halves = split_chain_by_principal_axis(
                chain,
                min_points=refiner.fine_min_valid_points,
            )
        except ValueError:
            continue

        for diagnostic in output["fine_diagnostics"]:
            half_outputs = diagnostic.get("halves", [])
            for half, half_output in zip(halves, half_outputs):
                augmented_half = apply_transform(half.points["2.00"], augmentation)
                errors = torch.stack(
                    [
                        compute_chain_rmse_angstrom(
                            augmented_half,
                            transform,
                            ground_truth,
                            scale,
                        )
                        for transform in half_output["candidate_transforms"]
                    ]
                )
                target_level = half_output["target_subclouds"]["2.00"]
                _, positives = select_gt_and_positive_subclouds(
                    augmented_half,
                    ground_truth,
                    target_level["points"],
                    target_level["masks"],
                    scale,
                )
                candidate_indices = half_output["candidate_indices"]
                samples.append(
                    MpnSample(
                        summaries=half_output["candidate_summaries"].detach(),
                        log_ops=torch.log(
                            half_output["ops_scores"][candidate_indices].clamp_min(
                                1e-8
                            )
                        ).detach(),
                        target=errors.argmin().detach(),
                        positive_mask=positives[candidate_indices].detach(),
                        candidate_rmse=errors.detach(),
                    )
                )
    return samples


@torch.no_grad()
def cache_fine_samples(
    refiner: HierarchicalRegistrationRefiner,
    processed_root: Path,
    structure_ids: list[str],
    device: torch.device,
    seed: int,
) -> list[MpnSample]:
    generator = torch.Generator(device=device).manual_seed(seed)
    samples: list[MpnSample] = []
    for structure_id in structure_ids:
        structure = _move_to_device(
            torch.load(
                processed_root / structure_id / "structure.pt",
                map_location="cpu",
                weights_only=False,
            ),
            device,
        )
        samples.extend(fine_samples_for_structure(refiner, structure, generator))
    return samples


@torch.no_grad()
def evaluate_hierarchical_structures(
    refiner: HierarchicalRegistrationRefiner,
    processed_root: Path,
    structure_ids: list[str],
    device: torch.device,
    seed: int,
) -> dict[str, float | int | None]:
    generator = torch.Generator(device=device).manual_seed(seed)
    rows: list[dict] = []
    refiner.eval()
    for structure_id in structure_ids:
        structure = _move_to_device(
            torch.load(
                processed_root / structure_id / "structure.pt",
                map_location="cpu",
                weights_only=False,
            ),
            device,
        )
        scale = float(structure["normalization"]["scale"])
        for chain_id in sorted(structure["chains"]):
            chain = structure["chains"][chain_id]
            augmentation = random_rigid_transform(
                chain["2.00"].device,
                chain["2.00"].dtype,
                generator,
            )
            ground_truth = inverse_transform(augmentation)
            output = refiner(structure, chain_id, source_transform=augmentation)
            augmented = apply_transform(chain["2.00"], augmentation)
            subclouds = _chain_subclouds(structure, chain_id)
            _, positives = select_gt_and_positive_subclouds(
                augmented,
                ground_truth,
                subclouds["2.00"]["points"],
                subclouds["2.00"]["masks"],
                scale,
            )
            parent_positive = positives[output["coarse_subcloud_indices"]]
            coarse_local = output["coarse_local_indices"]
            coarse_transforms = output["coarse_output"]["candidate_transforms"][
                coarse_local
            ]
            coarse_errors = torch.stack(
                [
                    compute_chain_rmse_angstrom(
                        augmented,
                        transform,
                        ground_truth,
                        scale,
                    )
                    for transform in coarse_transforms
                ]
            )
            errors = torch.stack(
                [
                    compute_chain_rmse_angstrom(
                        augmented,
                        transform,
                        ground_truth,
                        scale,
                    )
                    for transform in output["candidate_transforms"]
                ]
            )
            order = output["candidate_scores"].argsort(descending=True)
            top3 = order[: min(3, len(order))]
            top1 = int(order[0].item())
            rre, rte = _pose_errors(
                output["candidate_transforms"][top1],
                ground_truth,
                scale,
            )
            rows.append(
                {
                    "coarse_top3_region": bool(parent_positive.any()),
                    "coarse_top1_pose": bool(coarse_errors[0] <= 3.0),
                    "coarse_top3_pose": bool(coarse_errors.min() <= 3.0),
                    "coarse_top1_rmse": float(coarse_errors[0].cpu()),
                    "coarse_top3_rmse": float(coarse_errors.min().cpu()),
                    "top1_region": bool(parent_positive[top1]),
                    "top3_region": bool(parent_positive[top3].any()),
                    "top1_pose": bool(errors[top1] <= 3.0),
                    "top3_pose": bool(errors[top3].min() <= 3.0),
                    "top1_rmse": float(errors[top1].cpu()),
                    "top3_rmse": float(errors[top3].min().cpu()),
                    "top1_rre": rre,
                    "top1_rte": rte,
                    "fallbacks": sum(
                        status != "refined"
                        for status in output["refinement_status"]
                    ),
                    "candidate_count": len(order),
                }
            )
    return summarize_hierarchical_rows(rows)


def train_fine_mpn(
    checkpoint_path: Path,
    processed_root: Path,
    run_dir: Path,
    config_path: Path,
    device: torch.device,
    learning_rate: float,
    log_interval: int,
    patience: int,
    validation_structures: int,
    report_structures: int,
    max_cases: int | None,
    seed: int,
    resume: bool,
) -> None:
    _seed_everything(seed)
    config = load_config(config_path)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    coarse_model = build_model(config).to(device)
    coarse_model.load_state_dict(checkpoint["model"])
    refiner = build_hierarchical_refiner(config, coarse_model).to(device)
    freeze_for_fine_training(refiner)
    optimizer = torch.optim.AdamW(
        refiner.fine_mpn.parameters(),
        lr=learning_rate,
        weight_decay=config.training.weight_decay,
    )

    split = json.loads((processed_root / "split.json").read_text(encoding="utf-8"))
    validation_ids = list(split["val"][:validation_structures])
    remaining_validation = list(
        split["val"][
            validation_structures : validation_structures + report_structures
        ]
    )
    report_ids = remaining_validation
    if len(report_ids) < report_structures:
        report_ids.extend(
            split.get("test", [])[: report_structures - len(report_ids)]
        )
    validation_samples = cache_fine_samples(
        refiner,
        processed_root,
        validation_ids,
        device,
        seed + 10_000,
    )
    if not validation_samples:
        raise ValueError("validation produced no fine-stage samples")
    baseline_fine = evaluate_mpn_samples(refiner.fine_mpn, validation_samples)
    baseline_hierarchical = evaluate_hierarchical_structures(
        refiner,
        processed_root,
        validation_ids,
        device,
        seed + 10_000,
    )
    print(
        json.dumps(
            {
                "event": "fine_mpn_baseline",
                "fine": baseline_fine,
                "hierarchical": baseline_hierarchical,
            }
        ),
        flush=True,
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "fine_mpn_best.pt"
    recovery_path = run_dir / "fine_mpn_recovery.pt"
    if not resume:
        _save_best(
            best_path,
            refiner,
            config,
            checkpoint_path,
            0,
            baseline_fine,
            baseline_hierarchical,
        )
    train_ids = list(split["train"])
    random.Random(seed).shuffle(train_ids)
    if max_cases is not None:
        train_ids = train_ids[:max_cases]
    start = 0
    generator = torch.Generator(device=device).manual_seed(seed)
    stopper = EarlyStopper(patience, best=hierarchical_selection_metric(baseline_hierarchical))
    if resume and recovery_path.is_file():
        recovery = torch.load(
            recovery_path,
            map_location=device,
            weights_only=False,
        )
        refiner.fine_mpn.load_state_dict(recovery["fine_mpn"])
        optimizer.load_state_dict(recovery["optimizer"])
        train_ids = list(recovery["train_ids"])
        start = int(recovery["next_structure_index"])
        generator.set_state(recovery["generator_state"].to(device="cpu"))
        stopper = EarlyStopper(
            patience,
            float(recovery["best_hierarchical_top3_rmse_A"]),
            int(recovery["stale"]),
        )

    rolling_samples: list[MpnSample] = []
    rolling_losses: list[float] = []
    for index in range(start, len(train_ids)):
        structure = _move_to_device(
            torch.load(
                processed_root / train_ids[index] / "structure.pt",
                map_location="cpu",
                weights_only=False,
            ),
            device,
        )
        samples = fine_samples_for_structure(refiner, structure, generator)
        if samples:
            refiner.fine_mpn.train()
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for sample in samples:
                logits = sample.log_ops + refiner.fine_mpn(sample.summaries)
                losses.append(
                    F.cross_entropy(
                        logits.unsqueeze(0),
                        sample.target.reshape(1),
                    )
                )
            loss = torch.stack(losses).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                refiner.fine_mpn.parameters(),
                config.training.grad_clip_norm,
            )
            optimizer.step()
            rolling_samples.extend(samples)
            rolling_losses.append(float(loss.detach().cpu()))

        processed = index + 1
        if processed % log_interval != 0 and processed != len(train_ids):
            continue
        if not rolling_samples:
            continue
        train_metrics = evaluate_mpn_samples(
            refiner.fine_mpn,
            rolling_samples,
        )
        train_metrics["structure_loss"] = float(np.mean(rolling_losses))
        validation_metrics = evaluate_mpn_samples(
            refiner.fine_mpn,
            validation_samples,
        )
        hierarchical_metrics = evaluate_hierarchical_structures(
            refiner,
            processed_root,
            validation_ids,
            device,
            seed + 10_000,
        )
        improved = stopper.update(hierarchical_selection_metric(hierarchical_metrics))
        if improved:
            _save_best(
                best_path,
                refiner,
                config,
                checkpoint_path,
                processed,
                validation_metrics,
                hierarchical_metrics,
            )
        recovery = {
            "fine_mpn": refiner.fine_mpn.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_ids": train_ids,
            "next_structure_index": processed,
            "generator_state": generator.get_state(),
            "best_hierarchical_top3_rmse_A": stopper.best,
            "stale": stopper.stale,
        }
        temporary = recovery_path.with_suffix(".tmp")
        torch.save(recovery, temporary)
        temporary.replace(recovery_path)
        print(
            json.dumps(
                {
                    "event": "fine_mpn_progress",
                    "processed_structures": processed,
                    "total_structures": len(train_ids),
                    "improved": improved,
                    "best_hierarchical_top3_rmse_A": stopper.best,
                    "stale_checks": stopper.stale,
                    "train": train_metrics,
                    "validation": validation_metrics,
                    "hierarchical_validation": hierarchical_metrics,
                }
            ),
            flush=True,
        )
        rolling_samples.clear()
        rolling_losses.clear()
        if stopper.should_stop:
            print(
                json.dumps(
                    {
                        "event": "fine_mpn_early_stop",
                        "processed_structures": processed,
                        "best_hierarchical_top3_rmse_A": stopper.best,
                    }
                ),
                flush=True,
            )
            break

    if best_path.is_file() and report_ids:
        best = torch.load(best_path, map_location=device, weights_only=False)
        refiner.load_state_dict(best["model"])
        report_metrics = evaluate_hierarchical_structures(
            refiner,
            processed_root,
            report_ids,
            device,
            seed + 20_000,
        )
        print(
            json.dumps(
                {
                    "event": "fine_mpn_independent_report",
                    "structures": len(report_ids),
                    "hierarchical": report_metrics,
                }
            ),
            flush=True,
        )


def _save_best(
    path: Path,
    refiner: HierarchicalRegistrationRefiner,
    config: RegistrationConfig,
    source_checkpoint: Path,
    processed: int,
    validation_metrics: dict,
    hierarchical_metrics: dict,
) -> None:
    payload = {
        "model": refiner.state_dict(),
        "fine_mpn": refiner.fine_mpn.state_dict(),
        "config": asdict(config),
        "source_checkpoint": str(source_checkpoint),
        "processed_structures": processed,
        "validation_metrics": validation_metrics,
        "hierarchical_metrics": hierarchical_metrics,
    }
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _pose_errors(
    predicted: torch.Tensor,
    ground_truth: torch.Tensor,
    normalization_scale: float,
) -> tuple[float, float]:
    relative = predicted[:3, :3] @ ground_truth[:3, :3].T
    cosine = ((torch.trace(relative) - 1.0) / 2.0).clamp(-1.0, 1.0)
    rotation = torch.rad2deg(torch.acos(cosine))
    translation = (
        torch.linalg.norm(predicted[:3, 3] - ground_truth[:3, 3])
        * normalization_scale
    )
    return float(rotation.cpu()), float(translation.cpu())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train the hierarchical half-chain Fine-MPN"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--processed-root", required=True)
    parser.add_argument(
        "--run-dir",
        default="outputs/train_hierarchical_fine_v1",
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--validation-structures", type=int, default=10)
    parser.add_argument("--report-structures", type=int, default=10)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    train_fine_mpn(
        Path(args.checkpoint),
        Path(args.processed_root),
        Path(args.run_dir),
        Path(args.config),
        torch.device(args.device),
        args.learning_rate,
        args.log_interval,
        args.patience,
        args.validation_structures,
        args.report_structures,
        args.max_cases,
        args.seed,
        args.resume,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
