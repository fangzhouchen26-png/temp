from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from statistics import mean, median

import torch
import torch.nn.functional as F

from .config import RegistrationConfig, load_config
from .fine_point_matching import (
    FinePointMatchOutput,
    FinePointPairTooLarge,
    PointPairLabels,
    bidirectional_matchability_loss,
    fine_equivariant_alignment_loss,
    freeze_for_fine_point_training,
    make_gt_point_pairs,
    symmetric_multi_positive_descriptor_loss,
)
from .fine_shot import attach_fine_shot_4a
from .model import apply_transform, estimate_rigid_transform
from .train import (
    _make_grad_scaler,
    _move_to_device,
    _seed_everything,
    build_model,
)
from .train_fine_mpn import build_hierarchical_refiner
from .training import (
    compute_chain_rmse_angstrom,
    inverse_transform,
    random_rigid_transform,
)


def fine_point_training_loss(
    output: FinePointMatchOutput,
    labels: PointPairLabels,
    ground_truth_transform: torch.Tensor,
    descriptor_temperature: float = 0.1,
    matchability_weight: float = 0.25,
    equivariant_weight: float = 0.1,
) -> dict[str, torch.Tensor]:
    if not labels.valid:
        raise ValueError("fine point training requires a positive point pair")
    descriptor = symmetric_multi_positive_descriptor_loss(
        output.source_descriptors,
        output.target_descriptors,
        labels.positive_mask,
        temperature=descriptor_temperature,
    )
    matchability = bidirectional_matchability_loss(
        output.source_matchability_logits,
        output.target_matchability_logits,
        labels.source_matchable,
        labels.target_matchable,
    )
    equivariant = fine_equivariant_alignment_loss(
        output.source_equivariant,
        output.target_equivariant,
        labels.positive_mask,
        ground_truth_transform,
    )
    total = (
        descriptor
        + float(matchability_weight) * matchability
        + float(equivariant_weight) * equivariant
    )
    return {
        "total": total,
        "descriptor": descriptor,
        "matchability": matchability,
        "equivariant": equivariant,
    }


def point_match_metrics(
    source_descriptors: torch.Tensor,
    target_descriptors: torch.Tensor,
    positive_mask: torch.Tensor,
) -> dict[str, float | int]:
    if positive_mask.shape != (
        len(source_descriptors),
        len(target_descriptors),
    ):
        raise ValueError("positive_mask shape must match descriptor counts")
    valid = positive_mask.any(dim=1)
    valid_count = int(valid.sum().item())
    if valid_count == 0:
        return {
            "valid_source_points": 0,
            "point_recall_at_1_3A": 0.0,
            "point_recall_at_3_3A": 0.0,
        }
    similarity = (
        F.normalize(source_descriptors, dim=-1)
        @ F.normalize(target_descriptors, dim=-1).T
    )
    recalls = {}
    for k in (1, 3):
        count = min(k, similarity.shape[1])
        selected = similarity.topk(count, dim=1).indices
        hits = positive_mask.gather(1, selected).any(dim=1)
        recalls[f"point_recall_at_{k}_3A"] = float(
            hits[valid].float().mean().item()
        )
    return {
        "valid_source_points": valid_count,
        **recalls,
    }


@dataclass(frozen=True)
class FinePointSample:
    inputs: dict[str, torch.Tensor]
    labels: PointPairLabels
    ground_truth_transform: torch.Tensor
    normalization_scale: float
    structure_id: str
    chain_id: str


def _load_refiner(
    config: RegistrationConfig,
    checkpoint_path: Path,
    fine_encoder_checkpoint_path: Path | None,
    fine_mpn_checkpoint_path: Path | None,
    device: torch.device,
):
    if not config.model.use_fine_point_matcher:
        raise ValueError("model.use_fine_point_matcher must be true")
    coarse_payload = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    coarse_model = build_model(config).to(device)
    coarse_state = coarse_payload.get("model", coarse_payload)
    coarse_model.load_state_dict(coarse_state, strict=False)
    refiner = build_hierarchical_refiner(config, coarse_model).to(device)

    refiner.fine_encoder = copy.deepcopy(coarse_model.encoder).to(device)
    if fine_encoder_checkpoint_path is not None:
        payload = torch.load(
            fine_encoder_checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        state = payload.get("fine_encoder", payload)
        refiner.fine_encoder.load_state_dict(state, strict=False)

    if fine_mpn_checkpoint_path is not None:
        payload = torch.load(
            fine_mpn_checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        state = payload.get("fine_mpn")
        if state is None:
            raise KeyError("Fine-MPN checkpoint has no fine_mpn state")
        refiner.fine_mpn.load_state_dict(state, strict=False)

    freeze_for_fine_point_training(refiner)
    return refiner


@torch.no_grad()
def collect_fine_point_samples(
    refiner,
    structure: dict,
    structure_id: str,
    generator: torch.Generator,
    radius_angstrom: float,
    min_positive_pairs: int = 3,
    candidate_limit: int | None = None,
) -> tuple[list[FinePointSample], dict[str, int]]:
    matcher = refiner.fine_point_matcher
    if matcher is None:
        raise ValueError("refiner has no fine point matcher")
    if min_positive_pairs < 1:
        raise ValueError("min_positive_pairs must be positive")
    stats = {
        "chains": 0,
        "candidate_pairs": 0,
        "valid_candidates": 0,
        "no_positive_candidates": 0,
        "oversized_candidates": 0,
        "failed_chains": 0,
    }
    samples: list[FinePointSample] = []
    scale = float(structure["normalization"]["scale"])

    refiner.fine_point_matcher = None
    try:
        refiner.eval()
        for chain_id in sorted(structure["chains"]):
            stats["chains"] += 1
            chain = structure["chains"][chain_id]
            augmentation = random_rigid_transform(
                chain["2.00"].device,
                chain["2.00"].dtype,
                generator,
            )
            ground_truth = inverse_transform(augmentation)
            try:
                hierarchy = refiner(
                    structure,
                    chain_id,
                    source_transform=augmentation,
                )
            except ValueError:
                stats["failed_chains"] += 1
                continue
            for diagnostic in hierarchy.get("fine_diagnostics", []):
                for half_output in diagnostic.get("halves", []):
                    candidate_inputs = half_output.get(
                        "point_matcher_inputs", []
                    )
                    if candidate_limit is not None:
                        candidate_inputs = candidate_inputs[:candidate_limit]
                    for inputs in candidate_inputs:
                        stats["candidate_pairs"] += 1
                        pair_elements = (
                            len(inputs["source_points"])
                            * len(inputs["target_points"])
                        )
                        if pair_elements > matcher.max_pair_elements:
                            stats["oversized_candidates"] += 1
                            continue
                        labels = make_gt_point_pairs(
                            inputs["source_points"],
                            inputs["target_points"],
                            ground_truth,
                            scale,
                            radius_angstrom=radius_angstrom,
                        )
                        if labels.positive_count < min_positive_pairs:
                            stats["no_positive_candidates"] += 1
                            continue
                        samples.append(
                            FinePointSample(
                                inputs={
                                    key: value.detach()
                                    for key, value in inputs.items()
                                },
                                labels=labels,
                                ground_truth_transform=ground_truth.detach(),
                                normalization_scale=scale,
                                structure_id=structure_id,
                                chain_id=chain_id,
                            )
                        )
                        stats["valid_candidates"] += 1
    finally:
        refiner.fine_point_matcher = matcher
        freeze_for_fine_point_training(refiner)
    return samples, stats


@torch.no_grad()
def _pose_record(
    output: FinePointMatchOutput,
    sample: FinePointSample,
    config: RegistrationConfig,
) -> dict[str, float]:
    scale = sample.normalization_scale
    pose = estimate_rigid_transform(
        sample.inputs["source_points"],
        sample.inputs["target_points"],
        output.source_descriptors,
        output.target_descriptors,
        mutual_topk=config.model.mutual_topk,
        acceptance_radius=(
            config.model.equivariant_acceptance_radius_angstrom / scale
        ),
        max_hypotheses=config.model.equivariant_max_hypotheses,
        use_equivariant=True,
        src_equivariant=output.source_equivariant,
        tgt_equivariant=output.target_equivariant,
    )
    if pose.transform is None:
        return {
            "pose_valid": 0.0,
            "pose_success_3A": 0.0,
            "fallback": 1.0,
            "equivariant_support": 0.0,
            "correspondence_inlier_ratio_3A": 0.0,
        }

    predicted = pose.transform
    gt = sample.ground_truth_transform
    relative = predicted[:3, :3] @ gt[:3, :3].T
    cosine = ((torch.trace(relative) - 1.0) / 2.0).clamp(-1.0, 1.0)
    rre = float(torch.rad2deg(torch.acos(cosine)).cpu())
    rte = float(
        (
            torch.linalg.norm(predicted[:3, 3] - gt[:3, 3])
            * scale
        ).cpu()
    )
    rmse = float(
        compute_chain_rmse_angstrom(
            sample.inputs["source_points"],
            predicted,
            gt,
            scale,
        ).cpu()
    )
    if len(pose.source_correspondences):
        aligned = apply_transform(pose.source_correspondences, gt)
        distances = (
            torch.linalg.norm(aligned - pose.target_correspondences, dim=1)
            * scale
        )
        inlier_ratio = float((distances <= 3.0).float().mean().cpu())
    else:
        inlier_ratio = 0.0
    return {
        "pose_valid": 1.0,
        "pose_success_3A": float(rmse <= 3.0),
        "rre_degrees": rre,
        "rte_angstrom": rte,
        "rmse_angstrom": rmse,
        "fallback": float(pose.backend != "equivariant"),
        "equivariant_support": float(pose.equivariant_support),
        "correspondence_inlier_ratio_3A": inlier_ratio,
    }


def _sample_record(
    output: FinePointMatchOutput,
    losses: dict[str, torch.Tensor],
    sample: FinePointSample,
    config: RegistrationConfig,
) -> dict[str, float]:
    matching = point_match_metrics(
        output.source_descriptors.detach(),
        output.target_descriptors.detach(),
        sample.labels.positive_mask,
    )
    record = {
        "loss": float(losses["total"].detach().cpu()),
        "descriptor_loss": float(losses["descriptor"].detach().cpu()),
        "matchability_loss": float(losses["matchability"].detach().cpu()),
        "equivariant_loss": float(losses["equivariant"].detach().cpu()),
        "valid_source_points": float(matching["valid_source_points"]),
        "point_recall_at_1_3A": float(matching["point_recall_at_1_3A"]),
        "point_recall_at_3_3A": float(matching["point_recall_at_3_3A"]),
    }
    record.update(_pose_record(output, sample, config))
    return record


def summarize_records(
    records: list[dict[str, float]],
    sample_stats: dict[str, int],
) -> dict[str, float | int | None]:
    result: dict[str, float | int | None] = dict(sample_stats)
    result["trained_candidate_pairs"] = len(records)
    if not records:
        result.update(
            {
                "loss": None,
                "point_recall_at_1_3A": None,
                "point_recall_at_3_3A": None,
                "pose_valid_rate": 0.0,
                "pose_success_3A": 0.0,
                "mean_rre_degrees": None,
                "median_rre_degrees": None,
                "mean_rte_angstrom": None,
                "mean_rmse_angstrom": None,
                "median_rmse_angstrom": None,
                "fallback_rate": 1.0,
            }
        )
        return result

    for key in (
        "loss",
        "descriptor_loss",
        "matchability_loss",
        "equivariant_loss",
        "pose_valid",
        "pose_success_3A",
        "fallback",
        "equivariant_support",
        "correspondence_inlier_ratio_3A",
    ):
        result[key if key not in {"pose_valid", "fallback"} else {
            "pose_valid": "pose_valid_rate",
            "fallback": "fallback_rate",
        }[key]] = mean(row[key] for row in records)

    weight = sum(row["valid_source_points"] for row in records)
    for key in ("point_recall_at_1_3A", "point_recall_at_3_3A"):
        result[key] = (
            sum(row[key] * row["valid_source_points"] for row in records)
            / max(1.0, weight)
        )
    pose_rows = [row for row in records if row["pose_valid"]]
    if pose_rows:
        rre = [row["rre_degrees"] for row in pose_rows]
        rte = [row["rte_angstrom"] for row in pose_rows]
        rmse = [row["rmse_angstrom"] for row in pose_rows]
        result.update(
            {
                "mean_rre_degrees": mean(rre),
                "median_rre_degrees": median(rre),
                "mean_rte_angstrom": mean(rte),
                "mean_rmse_angstrom": mean(rmse),
                "median_rmse_angstrom": median(rmse),
            }
        )
    else:
        result.update(
            {
                "mean_rre_degrees": None,
                "median_rre_degrees": None,
                "mean_rte_angstrom": None,
                "mean_rmse_angstrom": None,
                "median_rmse_angstrom": None,
            }
        )
    return result


@torch.no_grad()
def evaluate_fine_point_matcher(
    refiner,
    config: RegistrationConfig,
    processed_root: Path,
    fine_shot_cache_root: Path,
    structure_ids: list[str],
    device: torch.device,
    seed: int,
    min_positive_pairs: int,
    candidate_limit: int | None,
) -> dict[str, float | int | None]:
    matcher = refiner.fine_point_matcher
    if matcher is None:
        raise ValueError("refiner has no fine point matcher")
    matcher.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    records: list[dict[str, float]] = []
    totals = {
        "chains": 0,
        "candidate_pairs": 0,
        "valid_candidates": 0,
        "no_positive_candidates": 0,
        "oversized_candidates": 0,
        "failed_chains": 0,
    }
    for structure_id in structure_ids:
        structure_cpu = torch.load(
            processed_root / structure_id / "structure.pt",
            map_location="cpu",
            weights_only=False,
        )
        attach_fine_shot_4a(structure_cpu, structure_id, fine_shot_cache_root)
        structure = _move_to_device(structure_cpu, device)
        samples, stats = collect_fine_point_samples(
            refiner,
            structure,
            structure_id,
            generator,
            config.model.fine_point_pair_radius_angstrom,
            min_positive_pairs,
            candidate_limit,
        )
        for key, value in stats.items():
            totals[key] += value
        matcher.eval()
        for sample in samples:
            try:
                output = matcher(**sample.inputs)
            except FinePointPairTooLarge:
                totals["oversized_candidates"] += 1
                continue
            losses = fine_point_training_loss(
                output,
                sample.labels,
                sample.ground_truth_transform,
                config.model.fine_point_descriptor_temperature,
            )
            records.append(_sample_record(output, losses, sample, config))
        del structure, samples
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return summarize_records(records, totals)


def _atomic_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _checkpoint_payload(
    refiner,
    config: RegistrationConfig,
    checkpoint_path: Path,
    fine_encoder_checkpoint_path: Path | None,
    fine_mpn_checkpoint_path: Path | None,
    processed: int,
    metrics: dict,
) -> dict:
    matcher = refiner.fine_point_matcher
    if matcher is None:
        raise ValueError("refiner has no fine point matcher")
    return {
        "fine_point_matcher": matcher.state_dict(),
        "config": asdict(config),
        "coarse_checkpoint": str(checkpoint_path),
        "fine_encoder_checkpoint": (
            None
            if fine_encoder_checkpoint_path is None
            else str(fine_encoder_checkpoint_path)
        ),
        "fine_mpn_checkpoint": (
            None
            if fine_mpn_checkpoint_path is None
            else str(fine_mpn_checkpoint_path)
        ),
        "processed_structures": processed,
        "validation_metrics": metrics,
    }


def _selection_values(metrics: dict) -> tuple[float, float]:
    recall = metrics.get("point_recall_at_3_3A")
    rre = metrics.get("median_rre_degrees")
    return (
        -1.0 if recall is None else float(recall),
        float("inf") if rre is None else float(rre),
    )


def _selection_improved(
    current: tuple[float, float],
    best: tuple[float, float],
) -> bool:
    if current[0] > best[0] + 1e-4:
        return True
    return abs(current[0] - best[0]) <= 1e-4 and current[1] < best[1] - 0.1


def _train_structure(
    refiner,
    config: RegistrationConfig,
    structure: dict,
    structure_id: str,
    generator: torch.Generator,
    optimizer: torch.optim.Optimizer,
    scaler,
    amp_enabled: bool,
    min_positive_pairs: int,
    candidate_limit: int | None,
) -> tuple[list[dict[str, float]], dict[str, int]]:
    matcher = refiner.fine_point_matcher
    if matcher is None:
        raise ValueError("refiner has no fine point matcher")
    samples, stats = collect_fine_point_samples(
        refiner,
        structure,
        structure_id,
        generator,
        config.model.fine_point_pair_radius_angstrom,
        min_positive_pairs,
        candidate_limit,
    )
    if not samples:
        return [], stats

    matcher.train()
    optimizer.zero_grad(set_to_none=True)
    records: list[dict[str, float]] = []
    sample_count = len(samples)
    for sample in samples:
        with torch.autocast(
            device_type=sample.inputs["source_points"].device.type,
            enabled=amp_enabled,
        ):
            output = matcher(**sample.inputs)
            losses = fine_point_training_loss(
                output,
                sample.labels,
                sample.ground_truth_transform,
                config.model.fine_point_descriptor_temperature,
            )
            scaled_loss = losses["total"] / sample_count
        scaler.scale(scaled_loss).backward()
        records.append(_sample_record(output, losses, sample, config))
        del output, losses, scaled_loss

    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(
        matcher.parameters(),
        config.training.grad_clip_norm,
    )
    scaler.step(optimizer)
    scaler.update()
    return records, stats


def train_fine_point_matcher(
    config_path: Path,
    checkpoint_path: Path,
    fine_encoder_checkpoint_path: Path | None,
    fine_mpn_checkpoint_path: Path | None,
    processed_root: Path,
    fine_shot_cache_root: Path,
    run_dir: Path,
    device: torch.device,
    max_cases: int | None,
    validation_structures: int,
    report_structures: int,
    log_interval: int,
    patience: int,
    min_positive_pairs: int,
    candidate_limit: int | None,
    seed: int,
    resume: bool,
) -> None:
    _seed_everything(seed)
    config = load_config(config_path)
    refiner = _load_refiner(
        config,
        checkpoint_path,
        fine_encoder_checkpoint_path,
        fine_mpn_checkpoint_path,
        device,
    )
    matcher = refiner.fine_point_matcher
    assert matcher is not None
    optimizer = torch.optim.AdamW(
        matcher.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    amp_enabled = config.training.mixed_precision and device.type == "cuda"
    scaler = _make_grad_scaler(device.type, amp_enabled)

    split = json.loads(
        (processed_root / "split.json").read_text(encoding="utf-8")
    )
    train_ids = list(split["train"])
    random.Random(seed).shuffle(train_ids)
    if max_cases is not None:
        train_ids = train_ids[:max_cases]
    validation_ids = list(split.get("val", [])[:validation_structures])
    if not train_ids:
        raise ValueError("training split is empty")
    if not validation_ids:
        raise ValueError("validation split is empty")

    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "fine_point_matcher_best.pt"
    recovery_path = run_dir / "fine_point_matcher_recovery.pt"
    generator = torch.Generator(device=device).manual_seed(seed)
    start = 0
    stale = 0

    if resume:
        if not recovery_path.is_file():
            raise FileNotFoundError(recovery_path)
        recovery = torch.load(
            recovery_path,
            map_location=device,
            weights_only=False,
        )
        matcher.load_state_dict(recovery["fine_point_matcher"])
        optimizer.load_state_dict(recovery["optimizer"])
        scaler.load_state_dict(recovery["scaler"])
        train_ids = list(recovery["train_ids"])
        start = int(recovery["next_structure_index"])
        generator.set_state(recovery["generator_state"].cpu())
        best_values = tuple(recovery["best_values"])
        stale = int(recovery["stale"])
    else:
        baseline = evaluate_fine_point_matcher(
            refiner,
            config,
            processed_root,
            fine_shot_cache_root,
            validation_ids,
            device,
            seed + 10_000,
            min_positive_pairs,
            candidate_limit,
        )
        best_values = _selection_values(baseline)
        _atomic_save(
            _checkpoint_payload(
                refiner,
                config,
                checkpoint_path,
                fine_encoder_checkpoint_path,
                fine_mpn_checkpoint_path,
                0,
                baseline,
            ),
            best_path,
        )
        print(
            json.dumps(
                {
                    "event": "fine_point_baseline",
                    "validation_structures": len(validation_ids),
                    "validation": baseline,
                }
            ),
            flush=True,
        )

    rolling_records: list[dict[str, float]] = []
    rolling_stats = {
        "chains": 0,
        "candidate_pairs": 0,
        "valid_candidates": 0,
        "no_positive_candidates": 0,
        "oversized_candidates": 0,
        "failed_chains": 0,
    }
    for index in range(start, len(train_ids)):
        structure_id = train_ids[index]
        structure_cpu = torch.load(
            processed_root / structure_id / "structure.pt",
            map_location="cpu",
            weights_only=False,
        )
        attach_fine_shot_4a(structure_cpu, structure_id, fine_shot_cache_root)
        structure = _move_to_device(structure_cpu, device)
        records, stats = _train_structure(
            refiner,
            config,
            structure,
            structure_id,
            generator,
            optimizer,
            scaler,
            amp_enabled,
            min_positive_pairs,
            candidate_limit,
        )
        rolling_records.extend(records)
        for key, value in stats.items():
            rolling_stats[key] += value
        del structure, records
        if device.type == "cuda":
            torch.cuda.empty_cache()

        processed = index + 1
        should_log = (
            processed % log_interval == 0 or processed == len(train_ids)
        )
        if not should_log:
            continue

        train_metrics = summarize_records(
            rolling_records,
            rolling_stats,
        )
        validation = evaluate_fine_point_matcher(
            refiner,
            config,
            processed_root,
            fine_shot_cache_root,
            validation_ids,
            device,
            seed + 10_000,
            min_positive_pairs,
            candidate_limit,
        )
        current_values = _selection_values(validation)
        improved = _selection_improved(current_values, best_values)
        if improved:
            best_values = current_values
            stale = 0
            _atomic_save(
                _checkpoint_payload(
                    refiner,
                    config,
                    checkpoint_path,
                    fine_encoder_checkpoint_path,
                    fine_mpn_checkpoint_path,
                    processed,
                    validation,
                ),
                best_path,
            )
        else:
            stale += 1

        recovery = {
            "fine_point_matcher": matcher.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "train_ids": train_ids,
            "next_structure_index": processed,
            "generator_state": generator.get_state(),
            "best_values": best_values,
            "stale": stale,
        }
        _atomic_save(recovery, recovery_path)
        print(
            json.dumps(
                {
                    "event": "fine_point_progress",
                    "processed_structures": processed,
                    "total_structures": len(train_ids),
                    "improved": improved,
                    "best_point_recall_at_3_3A": best_values[0],
                    "best_median_rre_degrees": best_values[1],
                    "stale_checks": stale,
                    "train": train_metrics,
                    "validation": validation,
                }
            ),
            flush=True,
        )
        rolling_records.clear()
        for key in rolling_stats:
            rolling_stats[key] = 0

        if stale >= patience:
            print(
                json.dumps(
                    {
                        "event": "fine_point_early_stop",
                        "processed_structures": processed,
                        "stale_checks": stale,
                    }
                ),
                flush=True,
            )
            break

    if not best_path.is_file():
        return
    best = torch.load(best_path, map_location=device, weights_only=False)
    matcher.load_state_dict(best["fine_point_matcher"])
    remaining_val = split.get("val", [])[
        validation_structures : validation_structures + report_structures
    ]
    report_ids = list(remaining_val)
    if len(report_ids) < report_structures:
        report_ids.extend(
            split.get("test", [])[: report_structures - len(report_ids)]
        )
    if report_ids:
        report = evaluate_fine_point_matcher(
            refiner,
            config,
            processed_root,
            fine_shot_cache_root,
            report_ids,
            device,
            seed + 20_000,
            min_positive_pairs,
            candidate_limit,
        )
        print(
            json.dumps(
                {
                    "event": "fine_point_independent_report",
                    "structures": len(report_ids),
                    "metrics": report,
                }
            ),
            flush=True,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train candidate-local downstream 4 A point matching"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fine-encoder-checkpoint")
    parser.add_argument("--fine-mpn-checkpoint")
    parser.add_argument("--processed-root", required=True)
    parser.add_argument(
        "--fine-shot-cache-root",
        default="/fangzhouc/alignmodel702/outputs/fine_shot_4a_cache_v1",
    )
    parser.add_argument(
        "--run-dir",
        default="outputs/train_fine_point_matcher_4a_v1",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--validation-structures", type=int, default=10)
    parser.add_argument("--report-structures", type=int, default=10)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-positive-pairs", type=int, default=3)
    parser.add_argument(
        "--candidate-limit-per-half",
        type=int,
        default=0,
        help="0 keeps all Fine OPS candidates",
    )
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    train_fine_point_matcher(
        Path(args.config),
        Path(args.checkpoint),
        (
            None
            if args.fine_encoder_checkpoint is None
            else Path(args.fine_encoder_checkpoint)
        ),
        (
            None
            if args.fine_mpn_checkpoint is None
            else Path(args.fine_mpn_checkpoint)
        ),
        Path(args.processed_root),
        Path(args.fine_shot_cache_root),
        Path(args.run_dir),
        device,
        args.max_cases,
        args.validation_structures,
        args.report_structures,
        args.log_interval,
        args.patience,
        args.min_positive_pairs,
        (
            None
            if args.candidate_limit_per_half == 0
            else args.candidate_limit_per_half
        ),
        args.seed,
        args.resume,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
