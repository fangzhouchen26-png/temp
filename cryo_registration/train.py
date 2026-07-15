from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import random

import numpy as np
import torch

from .config import RegistrationConfig, load_config
from .metrics import registration_metrics
from .model import ProteinRegistrationModel, apply_transform
from .training import (
    MPN_START_EPOCH,
    inverse_transform,
    random_rigid_transform,
    select_fine_gt_and_positive_subclouds,
    training_loss,
)


def build_model(config: RegistrationConfig) -> ProteinRegistrationModel:
    model = config.model
    return ProteinRegistrationModel(
        shot_dim=model.shot_dim,
        feature_dim=model.feature_dim,
        num_heads=model.num_heads,
        kernel_points=model.kernel_points,
        ops_topk=model.ops_topk,
        mutual_topk=model.mutual_topk,
        max_points_per_patch=model.max_points_per_patch,
        max_dense_points_per_patch=model.max_dense_points_per_patch,
        use_compatibility_graph=model.use_compatibility_graph,
        compatibility_distance_tolerance_angstrom=(
            model.compatibility_distance_tolerance_angstrom
        ),
        compatibility_max_nodes=model.compatibility_max_nodes,
        compatibility_min_clique_size=model.compatibility_min_clique_size,
        use_multiscale_pose_refinement=model.use_multiscale_pose_refinement,
        use_fusion_mlp=model.use_fusion_mlp,
        fusion_mlp_hidden_dim=model.fusion_mlp_hidden_dim,
        use_equivariant_pose=model.use_equivariant_hypothesis_pose,
        use_learned_equivariant_features=model.use_learned_equivariant_features,
        equivariant_feature_dim=model.equivariant_feature_dim,
        equivariant_max_hypotheses=model.equivariant_max_hypotheses,
        equivariant_acceptance_radius_angstrom=model.equivariant_acceptance_radius_angstrom,
    )


def _make_grad_scaler(device_type: str, enabled: bool):
    """Construct a GradScaler across supported PyTorch AMP APIs."""
    amp = getattr(torch, "amp", None)
    scaler_cls = getattr(amp, "GradScaler", None) if amp is not None else None
    if scaler_cls is not None:
        try:
            return scaler_cls(device_type, enabled=enabled)
        except TypeError:
            return scaler_cls(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled and device_type == "cuda")


def _backward_chain_loss(loss: torch.Tensor, chain_count: int, scaler) -> None:
    if chain_count <= 0:
        raise ValueError("chain_count must be positive")
    scaler.scale(loss / chain_count).backward()


def _release_cuda_cache_if_needed(
    device: torch.device, threshold: float = 0.0
) -> bool:
    if device.type != "cuda":
        return False
    total_memory = torch.cuda.get_device_properties(device).total_memory
    reserved_memory = torch.cuda.memory_reserved(device)
    if total_memory <= 0 or reserved_memory / total_memory < threshold:
        return False
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    return True


def _coarse_workload_elements(structure: dict, chain_id: str) -> int:
    source_count = len(structure["chains"][chain_id]["6.00"])
    coarse_points = _chain_subclouds(structure, chain_id)["6.00"]["points"]
    if coarse_points.ndim != 3:
        raise ValueError("coarse subcloud points must have shape (candidates, points, 3)")
    candidate_count, padded_target_count = coarse_points.shape[:2]
    return int(source_count * padded_target_count)


def train(
    config: RegistrationConfig,
    processed_root: str | Path,
    run_dir: str | Path,
    max_structures: int | None = None,
    max_chains: int | None = None,
    resume: bool = False,
    init_checkpoint: str | Path | None = None,
) -> list[dict[str, float]]:
    _seed_everything(config.training.seed)
    processed_root = Path(processed_root)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    split = json.loads((processed_root / "split.json").read_text(encoding="utf-8"))
    structure_ids = split["train"][:max_structures]
    validation_ids = split.get("val", [])[:max_structures]
    test_ids = split.get("test", [])[:max_structures]
    if not structure_ids:
        raise ValueError("training split is empty")
    device = _resolve_device(config.training.device)
    model = build_model(config).to(device)
    if init_checkpoint is not None:
        checkpoint = torch.load(
            init_checkpoint, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model"], strict=False)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    amp_enabled = config.training.mixed_precision and device.type == "cuda"
    scaler = _make_grad_scaler(device.type, enabled=amp_enabled)
    generator = torch.Generator(device=device).manual_seed(config.training.seed)
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    stale_epochs = 0
    start_epoch = 0
    recovery_path = run_dir / "recovery.pt"
    resume_structure_index = 0
    resume_in_epoch = False
    if resume:
        checkpoint_path = (
            recovery_path if recovery_path.is_file() else run_dir / "last.pt"
        )
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"resume checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"])
        resume_in_epoch = bool(checkpoint.get("in_epoch", False))
        if resume_in_epoch:
            start_epoch = int(checkpoint["epoch_index"])
            resume_structure_index = int(checkpoint["next_structure_index"])
            structure_ids = list(checkpoint["structure_ids"])
            if "generator_state" in checkpoint:
                _restore_generator_state(generator, checkpoint["generator_state"])
        history_path = run_dir / "history.json"
        if history_path.is_file():
            history = json.loads(history_path.read_text(encoding="utf-8"))
        best_path = run_dir / "best.pt"
        if best_path.is_file():
            best_checkpoint = torch.load(
                best_path, map_location="cpu", weights_only=False
            )
            best_loss = float(
                best_checkpoint.get("best_loss", best_checkpoint["selection_metric"])
            )

    for epoch in range(start_epoch, config.training.epochs):
        model.train()
        if epoch == config.training.mpn_start_epoch:
            best_loss = float("inf")
            stale_epochs = 0
        epoch_losses: list[float] = []
        epoch_metrics: list[dict[str, float]] = []
        rolling_structure_metrics: list[dict[str, float]] = []
        if resume_in_epoch and epoch == start_epoch:
            structure_start = resume_structure_index
        else:
            random.shuffle(structure_ids)
            structure_start = 0
        for structure_index in range(structure_start, len(structure_ids)):
            structure_id = structure_ids[structure_index]
            structure_metrics: list[dict[str, float]] = []
            structure = torch.load(
                processed_root / structure_id / "structure.pt", weights_only=False
            )
            chain_ids = sorted(structure["chains"])[:max_chains]
            workload_limit = config.training.max_coarse_workload_elements
            chain_workloads = {
                chain_id: _coarse_workload_elements(structure, chain_id)
                for chain_id in chain_ids
            }
            if workload_limit is not None and any(
                workload > workload_limit for workload in chain_workloads.values()
            ):
                print(
                    json.dumps(
                        {
                            "event": "structure_skipped_memory_budget",
                            "structure_id": structure_id,
                            "structure_index": structure_index,
                            "limit": workload_limit,
                            "chain_workloads": chain_workloads,
                        }
                    ),
                    flush=True,
                )
                del structure
                continue
            structure = _move_to_device(structure, device)
            optimizer.zero_grad(set_to_none=True)
            chain_loss_values: list[float] = []
            for chain_id in chain_ids:
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                    encoded_target = model.encode_target(structure, chain_id)
                    chain_subclouds = _chain_subclouds(structure, chain_id)
                    augmentation = random_rigid_transform(device, structure["chains"][chain_id]["2.00"].dtype, generator)
                    ground_truth = inverse_transform(augmentation)
                    try:
                        output = model(
                            structure,
                            chain_id,
                            source_transform=augmentation,
                            encoded_target=encoded_target,
                            differentiable_pose=config.training.sup_awl,
                            use_fcw=config.training.sup_awl,
                        )
                    except RuntimeError:
                        subcloud_shapes = {
                            key: list(chain_subclouds[key]["points"].shape)
                            for key in model.scale_keys
                        }
                        source_sizes = {
                            key: len(structure["chains"][chain_id][key])
                            for key in model.scale_keys
                        }
                        print(
                            json.dumps(
                                {
                                    "event": "train_failure_context",
                                    "structure_id": structure_id,
                                    "structure_index": structure_index,
                                    "chain_id": chain_id,
                                    "source_sizes": source_sizes,
                                    "subcloud_shapes": subcloud_shapes,
                                    "cuda_allocated_gb": (
                                        torch.cuda.memory_allocated(device) / 1e9
                                    ),
                                    "cuda_reserved_gb": (
                                        torch.cuda.memory_reserved(device) / 1e9
                                    ),
                                }
                            ),
                            flush=True,
                        )
                        raise
                    augmented_fine = apply_transform(structure["chains"][chain_id]["2.00"], augmentation)
                    augmented_coarse = apply_transform(structure["chains"][chain_id]["6.00"], augmentation)
                    gt_subcloud, positive_subclouds = select_fine_gt_and_positive_subclouds(
                        {"2.00": augmented_fine, "6.00": augmented_coarse}, ground_truth, chain_subclouds, float(structure["normalization"]["scale"])
                    )
                    loss = training_loss(
                        output, augmented_fine, ground_truth, gt_subcloud, float(structure["normalization"]["scale"]), epoch,
                        correspondence_start_epoch=config.training.correspondence_start_epoch,
                        mpn_start_epoch=config.training.mpn_start_epoch,
                        positive_subcloud_mask=positive_subclouds,
                        sup_awl=config.training.sup_awl,
                        equivariant_loss_weight=config.training.equivariant_loss_weight,
                        equivariant_loss_start_epoch=config.training.equivariant_loss_start_epoch,
                    )
                _backward_chain_loss(loss["total"], len(chain_ids), scaler)
                chain_loss_values.append(float(loss["total"].detach().cpu()))
                metric = registration_metrics(
                    output, augmented_fine, ground_truth, gt_subcloud, float(structure["normalization"]["scale"]),
                    positive_subcloud_mask=positive_subclouds,
                )
                epoch_metrics.append(metric)
                structure_metrics.append(metric)
                del (
                    encoded_target,
                    chain_subclouds,
                    augmentation,
                    ground_truth,
                    output,
                    augmented_fine,
                    augmented_coarse,
                    gt_subcloud,
                    positive_subclouds,
                    loss,
                    metric,
                )
                _release_cuda_cache_if_needed(device)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            epoch_losses.append(float(np.mean(chain_loss_values)))
            rolling_structure_metrics.append(
                {
                    key: float(np.mean([metric[key] for metric in structure_metrics]))
                    for key in structure_metrics[0]
                }
            )
            del structure
            _release_cuda_cache_if_needed(device)
            processed_structures = structure_index + 1
            log_interval = config.training.log_interval_structures
            if log_interval > 0 and processed_structures % log_interval == 0:
                record = _progress_log_record(
                    epoch + 1,
                    processed_structures,
                    len(structure_ids),
                    epoch_losses,
                    rolling_structure_metrics,
                    log_interval,
                )
                print(
                    json.dumps({"event": "train_progress", **record}), flush=True
                )
            checkpoint_interval = config.training.checkpoint_interval_structures
            if checkpoint_interval > 0 and (
                processed_structures % checkpoint_interval == 0
                or processed_structures == len(structure_ids)
            ):
                recovery_payload = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": epoch,
                    "epoch_index": epoch,
                    "next_structure_index": processed_structures,
                    "structure_ids": structure_ids,
                    "generator_state": generator.get_state(),
                    "in_epoch": True,
                    "config": asdict(config),
                }
                recovery_tmp = run_dir / "recovery.tmp"
                torch.save(recovery_payload, recovery_tmp)
                recovery_tmp.replace(recovery_path)
                (run_dir / "progress.json").write_text(
                    json.dumps(
                        {
                            "status": "training",
                            "epoch": epoch + 1,
                            "processed_structures": processed_structures,
                            "total_structures": len(structure_ids),
                            "last_structure_id": structure_id,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )

        mean_loss = float(np.mean(epoch_losses))
        metric_means = {
            key: float(np.mean([metrics[key] for metrics in epoch_metrics]))
            for key in epoch_metrics[0]
        }
        validation_metrics = _evaluate(
            model,
            processed_root,
            validation_ids,
            device,
            config.training.seed,
            max_chains,
        )
        history.append(
            {"epoch": epoch + 1, "train_loss": mean_loss, **metric_means, **validation_metrics}
        )
        (run_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        monitor = _selection_metric(validation_metrics, metric_means, mean_loss)
        checkpoint_payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch + 1,
            "selection_metric": monitor,
            "config": asdict(config),
        }
        torch.save(checkpoint_payload, run_dir / "last.pt")
        recovery_path.unlink(missing_ok=True)
        resume_in_epoch = False
        (run_dir / "progress.json").write_text(
            json.dumps(
                {
                    "status": "epoch_complete",
                    "epoch": epoch + 1,
                    "validation_metrics": validation_metrics,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if monitor < best_loss:
            best_loss = monitor
            stale_epochs = 0
            checkpoint_payload["best_loss"] = best_loss
            torch.save(checkpoint_payload, run_dir / "best.pt")
        elif _early_stopping_active(epoch, config.training.mpn_start_epoch):
            stale_epochs += 1
            if stale_epochs >= config.training.early_stopping_patience:
                break
    if test_ids:
        best_checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(best_checkpoint["model"])
        test_metrics = _evaluate(
            model,
            processed_root,
            test_ids,
            device,
            config.training.seed + 1,
            max_chains,
            prefix="test",
        )
        (run_dir / "test_metrics.json").write_text(
            json.dumps(test_metrics, indent=2), encoding="utf-8"
        )
    return history


@torch.no_grad()
def _evaluate(
    model: ProteinRegistrationModel,
    processed_root: Path,
    structure_ids: list[str],
    device: torch.device,
    seed: int,
    max_chains: int | None,
    prefix: str = "val",
) -> dict[str, float]:
    if not structure_ids:
        return {}
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed + 10_000)
    collected: list[dict[str, float]] = []
    for structure_id in structure_ids:
        structure = _move_to_device(
            torch.load(processed_root / structure_id / "structure.pt", weights_only=False),
            device,
        )
        for chain_id in sorted(structure["chains"])[:max_chains]:
            encoded_target = model.encode_target(structure, chain_id)
            chain_subclouds = _chain_subclouds(structure, chain_id)
            augmentation = random_rigid_transform(
                device, structure["chains"][chain_id]["2.00"].dtype, generator
            )
            ground_truth = inverse_transform(augmentation)
            output = model(
                structure,
                chain_id,
                source_transform=augmentation,
                encoded_target=encoded_target,
            )
            augmented_fine = apply_transform(
                structure["chains"][chain_id]["2.00"], augmentation
            )
            augmented_coarse = apply_transform(
                structure["chains"][chain_id]["6.00"], augmentation
            )
            gt_subcloud, positive_subclouds = select_fine_gt_and_positive_subclouds(
                {"2.00": augmented_fine, "6.00": augmented_coarse},
                ground_truth,
                chain_subclouds,
                float(structure["normalization"]["scale"]),
            )
            collected.append(
                registration_metrics(
                    output,
                    augmented_fine,
                    ground_truth,
                    gt_subcloud,
                    float(structure["normalization"]["scale"]),
                    positive_subcloud_mask=positive_subclouds,
                )
            )
    return {
        f"{prefix}_{key}": float(np.mean([metrics[key] for metrics in collected]))
        for key in collected[0]
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train protein-chain registration")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--processed-root")
    parser.add_argument("--run-dir")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device")
    parser.add_argument("--max-structures", type=int)
    parser.add_argument("--max-chains", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--init-checkpoint")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.epochs is not None or args.device is not None:
        config = replace(
            config,
            training=replace(
                config.training,
                epochs=args.epochs or config.training.epochs,
                device=args.device or config.training.device,
            ),
        )
    train(
        config,
        args.processed_root or config.data.processed_root,
        args.run_dir or config.output.run_dir,
        max_structures=args.max_structures,
        max_chains=args.max_chains,
        resume=args.resume,
        init_checkpoint=args.init_checkpoint,
    )
    return 0


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _chain_subclouds(structure: dict, chain_id: str) -> dict:
    subclouds = structure["subclouds"]
    return subclouds if "6.00" in subclouds else subclouds[chain_id]



def _early_stopping_active(
    epoch: int, mpn_start_epoch: int = MPN_START_EPOCH
) -> bool:
    return epoch >= mpn_start_epoch

def _selection_metric(
    validation_metrics: dict[str, float],
    training_metrics: dict[str, float],
    training_loss_value: float,
) -> float:
    return validation_metrics.get(
        "val_rmse_angstrom",
        training_metrics.get("rmse_angstrom", training_loss_value),
    )


def _move_to_device(value, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    return value


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _progress_log_record(
    epoch: int,
    processed_structures: int,
    total_structures: int,
    losses: list[float],
    metrics: list[dict[str, float]],
    window: int,
) -> dict[str, float | int]:
    if window <= 0:
        raise ValueError("window must be positive")
    count = min(window, len(losses), len(metrics))
    if count == 0:
        raise ValueError("losses and metrics must not be empty")
    recent_metrics = metrics[-count:]
    record: dict[str, float | int] = {
        "epoch": epoch,
        "processed_structures": processed_structures,
        "total_structures": total_structures,
        "window_structures": count,
        "loss": float(np.mean(losses[-count:])),
    }
    record.update(
        {
            key: float(np.mean([metric[key] for metric in recent_metrics]))
            for key in recent_metrics[0]
        }
    )
    return record


def _restore_generator_state(
    generator: torch.Generator, state: torch.Tensor
) -> None:
    generator.set_state(state.cpu())


if __name__ == "__main__":
    raise SystemExit(main())
