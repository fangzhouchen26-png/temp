from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .model import apply_transform
from .overlap import compute_overlap, symmetric_overlap_rate

CORRESPONDENCE_START_EPOCH = 20
MPN_START_EPOCH = 50



def random_rigid_transform(
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
    translation_limit: float = 0.5,
) -> torch.Tensor:
    """Draw a uniform SO(3) rotation and bounded translation."""
    u1, u2, u3 = torch.rand(3, device=device, dtype=dtype, generator=generator)
    quaternion = torch.stack(
        [
            torch.sqrt(1 - u1) * torch.sin(2 * math.pi * u2),
            torch.sqrt(1 - u1) * torch.cos(2 * math.pi * u2),
            torch.sqrt(u1) * torch.sin(2 * math.pi * u3),
            torch.sqrt(u1) * torch.cos(2 * math.pi * u3),
        ]
    )
    x, y, z, w = quaternion
    rotation = torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ]
    ).reshape(3, 3)
    translation = (torch.rand(3, device=device, dtype=dtype, generator=generator) * 2 - 1) * translation_limit
    transform = torch.eye(4, device=device, dtype=dtype)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def inverse_transform(transform: torch.Tensor) -> torch.Tensor:
    inverse = torch.eye(4, dtype=transform.dtype, device=transform.device)
    rotation = transform[:3, :3]
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -(rotation.T @ transform[:3, 3])
    return inverse


def compute_chain_rmse_angstrom(
    source_points: torch.Tensor,
    predicted_transform: torch.Tensor,
    ground_truth_transform: torch.Tensor,
    normalization_scale: float,
) -> torch.Tensor:
    predicted = apply_transform(source_points, predicted_transform)
    expected = apply_transform(source_points, ground_truth_transform)
    return torch.sqrt(torch.sum((predicted - expected) ** 2, dim=1).mean()) * normalization_scale


def select_gt_subcloud(
    augmented_source_points: torch.Tensor,
    ground_truth_transform: torch.Tensor,
    target_patches: torch.Tensor,
    target_masks: torch.Tensor,
    max_source_points: int = 512,
    max_target_points: int = 4096,
    source_chunk_size: int = 128,
) -> int:
    nearest_distances = _subcloud_nearest_distances(
        augmented_source_points, ground_truth_transform, target_patches, target_masks,
        max_source_points, max_target_points, source_chunk_size,
    )
    errors = [torch.sqrt((nearest * nearest).mean()) for nearest in nearest_distances]
    return int(torch.stack(errors).argmin().item())


def select_gt_and_positive_subclouds(
    augmented_source_points: torch.Tensor,
    ground_truth_transform: torch.Tensor,
    target_patches: torch.Tensor,
    target_masks: torch.Tensor,
    normalization_scale: float,
    normalization_center: torch.Tensor | None = None,
    max_distance_angstrom: float = 3.0,
    min_coverage: float = 0.6,
    max_source_points: int = 512,
    max_target_points: int = 4096,
    source_chunk_size: int = 128,
) -> tuple[int, torch.Tensor]:
    """Return the closest patch and every patch that covers the aligned chain."""
    nearest_distances = _subcloud_nearest_distances(
        augmented_source_points, ground_truth_transform, target_patches, target_masks,
        max_source_points, max_target_points, source_chunk_size,
    )
    errors = torch.stack(
        [torch.sqrt((nearest * nearest).mean()) for nearest in nearest_distances]
    )
    center = (
        augmented_source_points.new_zeros(3)
        if normalization_center is None
        else torch.as_tensor(
            normalization_center,
            device=augmented_source_points.device,
            dtype=augmented_source_points.dtype,
        )
    )
    aligned_source = _uniform_limit(
        apply_transform(augmented_source_points, ground_truth_transform),
        max_source_points,
    )
    source_angstrom = (
        aligned_source * normalization_scale + center
    ).detach().cpu().numpy()
    overlap_rates = []
    for patch, mask in zip(target_patches, target_masks):
        valid_patch = _uniform_limit(patch[mask], max_target_points)
        target_angstrom = (
            valid_patch * normalization_scale + center
        ).detach().cpu().numpy()
        has_src, has_tgt, _ = compute_overlap(
            source_angstrom, target_angstrom, max_distance_angstrom
        )
        overlap_rates.append(symmetric_overlap_rate(has_src, has_tgt))
    coverage = errors.new_tensor(overlap_rates)

    selected = int(errors.argmin().item())
    positives = coverage >= min_coverage
    if not positives.any():
        positives[selected] = True
    return selected, positives


def _subcloud_nearest_distances(
    augmented_source_points: torch.Tensor,
    ground_truth_transform: torch.Tensor,
    target_patches: torch.Tensor,
    target_masks: torch.Tensor,
    max_source_points: int,
    max_target_points: int,
    source_chunk_size: int,
) -> list[torch.Tensor]:
    aligned_source = _uniform_limit(
        apply_transform(augmented_source_points, ground_truth_transform), max_source_points
    )
    nearest_distances = []
    for patch, mask in zip(target_patches, target_masks):
        valid_patch = _uniform_limit(patch[mask], max_target_points)
        nearest_distances.append(
            torch.cat(
                [
                    torch.cdist(chunk, valid_patch).min(dim=1).values
                    for chunk in aligned_source.split(source_chunk_size)
                ]
            )
        )
    return nearest_distances


def _uniform_limit(points: torch.Tensor, limit: int) -> torch.Tensor:
    if limit <= 0:
        raise ValueError("point limit must be positive")
    if len(points) <= limit:
        return points
    indices = torch.linspace(
        0, len(points) - 1, limit, device=points.device
    ).long()
    return points[indices]


def select_fine_gt_subcloud(
    augmented_source_levels: dict[str, torch.Tensor],
    ground_truth_transform: torch.Tensor,
    target_subcloud_levels: dict[str, dict[str, torch.Tensor]],
) -> int:
    """Choose the training target using the finest available 2 Å geometry."""
    fine_source = augmented_source_levels["2.00"]
    fine_target = target_subcloud_levels["2.00"]
    return select_gt_subcloud(
        fine_source,
        ground_truth_transform,
        fine_target["points"],
        fine_target["masks"],
    )


def select_fine_gt_and_positive_subclouds(
    augmented_source_levels: dict[str, torch.Tensor],
    ground_truth_transform: torch.Tensor,
    target_subcloud_levels: dict[str, dict[str, torch.Tensor]],
    normalization_scale: float,
) -> tuple[int, torch.Tensor]:
    fine_source = augmented_source_levels["2.00"]
    fine_target = target_subcloud_levels["2.00"]
    return select_gt_and_positive_subclouds(
        fine_source,
        ground_truth_transform,
        fine_target["points"],
        fine_target["masks"],
        normalization_scale,
    )



def sup_alignment_aware_weighting_loss(
    candidate_rmse_angstrom: torch.Tensor,
    predicted_weights: torch.Tensor,
    normalization_scale: float,
    acceptance_rmse_normalized: float = 0.2,
    reasonable_rmse_normalized: float = 0.3,
) -> torch.Tensor:
    """SuP FeatureConsistencyLoss with all reported errors in angstroms."""
    if candidate_rmse_angstrom.ndim != 1:
        raise ValueError("candidate_rmse_angstrom must be one-dimensional")
    if predicted_weights.shape != candidate_rmse_angstrom.shape:
        raise ValueError("predicted_weights must match candidate_rmse_angstrom")
    if normalization_scale <= 0:
        raise ValueError("normalization_scale must be positive")
    if not (0 < acceptance_rmse_normalized < reasonable_rmse_normalized):
        raise ValueError("RMSE thresholds must satisfy 0 < acceptance < reasonable")
    normalized_rmse = candidate_rmse_angstrom / normalization_scale
    acceptance_angstrom = acceptance_rmse_normalized * normalization_scale
    reasonable_angstrom = reasonable_rmse_normalized * normalization_scale
    weights = 1.0 - normalized_rmse
    success = candidate_rmse_angstrom < acceptance_angstrom
    weights = torch.where(
        success | (candidate_rmse_angstrom > reasonable_angstrom),
        torch.ones_like(weights),
        weights,
    )
    labels = success.to(predicted_weights.dtype)
    pointwise = weights * (predicted_weights - labels).abs()
    terms: list[torch.Tensor] = []
    if success.any():
        terms.append(pointwise[success].mean())
    if (~success).any():
        terms.append(pointwise[~success].mean())
    if not terms:
        return candidate_rmse_angstrom.sum() * 0.0
    return torch.stack(terms).sum()


def descriptor_correspondence_loss(
    encoded_levels: dict[str, dict[str, torch.Tensor]],
    gt_subcloud_index: int,
    ground_truth_transform: torch.Tensor,
    normalization_scale: float,
    temperature: float = 0.1,
    max_distance_angstrom: float = 3.0,
) -> torch.Tensor:
    """Supervise rotation-invariant point features with geometric matches."""
    if not encoded_levels:
        raise ValueError("encoded_levels must not be empty")
    if temperature <= 0 or max_distance_angstrom <= 0:
        raise ValueError("temperature and max_distance_angstrom must be positive")
    terms: list[torch.Tensor] = []
    zero: torch.Tensor | None = None
    for level in encoded_levels.values():
        source_points = level["source_points"]
        source_features = level["source_features"]
        target_mask = level["target_masks"][gt_subcloud_index]
        target_points = level["target_points"][gt_subcloud_index][target_mask]
        target_features = level["target_features"][gt_subcloud_index][target_mask]
        if zero is None:
            zero = source_features.sum() * 0.0
        if len(source_points) == 0 or len(target_points) == 0:
            continue

        aligned_source = apply_transform(source_points, ground_truth_transform)
        distances = torch.cdist(aligned_source, target_points)
        logits = (
            F.normalize(source_features, dim=-1)
            @ F.normalize(target_features, dim=-1).T
        ) / temperature

        source_distance, source_labels = distances.min(dim=1)
        valid_source = source_distance * normalization_scale <= max_distance_angstrom
        if valid_source.any():
            terms.append(F.cross_entropy(logits[valid_source], source_labels[valid_source]))

        target_distance, target_labels = distances.min(dim=0)
        valid_target = target_distance * normalization_scale <= max_distance_angstrom
        if valid_target.any():
            terms.append(
                F.cross_entropy(logits.T[valid_target], target_labels[valid_target])
            )

    if zero is None:
        raise ValueError("encoded_levels must contain point features")
    return torch.stack(terms).mean() if terms else zero


def equivariant_feature_alignment_loss(
    encoded_levels: dict[str, dict[str, torch.Tensor]],
    gt_subcloud_index: int,
    ground_truth_transform: torch.Tensor,
    normalization_scale: float,
    max_distance_angstrom: float = 3.0,
) -> torch.Tensor:
    """Train vector features with the known rigid rotation.

    This loss is applied before hard Sinkhorn/top-k/SVD operations. Pose
    hypothesis selection remains an inference-only operation.
    """
    level = encoded_levels.get("6.00")
    if level is None:
        raise ValueError("encoded_levels must contain 6.00")
    source_points = level["source_points"]
    source_vectors = level.get("source_equivariant")
    target_vectors_all = level.get("target_equivariant")
    zero = source_points.sum() * 0.0
    if source_vectors is None or target_vectors_all is None:
        return zero
    target_mask = level["target_masks"][gt_subcloud_index]
    target_points = level["target_points"][gt_subcloud_index][target_mask]
    target_vectors = target_vectors_all[gt_subcloud_index][target_mask]
    if len(source_points) == 0 or len(target_points) == 0:
        return zero

    rotation = ground_truth_transform[:3, :3]
    aligned_source = apply_transform(source_points, ground_truth_transform)
    distances = torch.cdist(aligned_source.float(), target_points.float())
    source_distance, source_indices = distances.min(dim=1)
    target_distance, target_indices = distances.min(dim=0)
    threshold = max_distance_angstrom / normalization_scale
    terms: list[torch.Tensor] = []

    valid_source = source_distance <= threshold
    if valid_source.any():
        transformed = torch.matmul(
            source_vectors[valid_source].float(), rotation.float().T
        )
        matched = target_vectors[source_indices[valid_source]].float()
        cosine = (
            F.normalize(transformed, dim=-1, eps=1e-6)
            * F.normalize(matched, dim=-1, eps=1e-6)
        ).sum(dim=-1)
        terms.append((1.0 - cosine).mean())

    valid_target = target_distance <= threshold
    if valid_target.any():
        inverse_rotation = rotation.T
        transformed = torch.matmul(
            target_vectors[valid_target].float(), inverse_rotation.float().T
        )
        matched = source_vectors[target_indices[valid_target]].float()
        cosine = (
            F.normalize(transformed, dim=-1, eps=1e-6)
            * F.normalize(matched, dim=-1, eps=1e-6)
        ).sum(dim=-1)
        terms.append((1.0 - cosine).mean())
    return torch.stack(terms).mean().to(source_vectors.dtype) if terms else zero


def training_loss(
    output: dict,
    augmented_source_points: torch.Tensor,
    ground_truth_transform: torch.Tensor,
    gt_subcloud_index: int,
    normalization_scale: float,
    epoch: int,
    correspondence_start_epoch: int = CORRESPONDENCE_START_EPOCH,
    mpn_start_epoch: int = MPN_START_EPOCH,
    positive_subcloud_mask: torch.Tensor | None = None,
    sup_awl: bool = False,
    equivariant_loss_weight: float = 0.0,
    equivariant_loss_start_epoch: int = 1,
) -> dict[str, torch.Tensor]:
    eps = torch.finfo(output["ops_scores"].dtype).eps
    ops_log_prob = torch.log(output["ops_scores"].clamp_min(eps))
    ops_log_prob = F.log_softmax(ops_log_prob, dim=0)
    if positive_subcloud_mask is None:
        positive_subcloud_mask = torch.zeros_like(output["ops_scores"], dtype=torch.bool)
        positive_subcloud_mask[gt_subcloud_index] = True
    else:
        positive_subcloud_mask = positive_subcloud_mask.to(
            device=output["ops_scores"].device, dtype=torch.bool
        )
    if positive_subcloud_mask.shape != output["ops_scores"].shape:
        raise ValueError("positive_subcloud_mask must match ops_scores")
    ops_loss = -torch.logsumexp(ops_log_prob[positive_subcloud_mask], dim=0)

    if sup_awl:
        errors_angstrom = torch.stack(
            [
                compute_chain_rmse_angstrom(
                    augmented_source_points,
                    transform,
                    ground_truth_transform,
                    normalization_scale,
                )
                for transform in output["candidate_transforms"]
            ]
        )
        predicted_weights = output.get("fcw_weights")
        if predicted_weights is None:
            predicted_logits = output.get("mpn_logits", output["final_logits"])
            predicted_weights = torch.sigmoid(predicted_logits)
        facw_loss = sup_alignment_aware_weighting_loss(
            errors_angstrom,
            predicted_weights,
            normalization_scale,
        )
        zero = facw_loss.new_zeros(())
        return {
            "total": facw_loss,
            "ops": zero,
            "descriptor": zero,
            "correspondence": zero,
            "pose": zero,
            "mpn": zero,
            "facw": facw_loss,
            "equivariant": facw_loss.new_zeros(()),
            "candidate_rmse_angstrom": errors_angstrom,
        }

    errors = torch.stack(
        [
            compute_chain_rmse_angstrom(
                augmented_source_points, transform, ground_truth_transform, normalization_scale
            )
            for transform in output["candidate_transforms"]
        ]
    ).detach()
    zero = ops_loss.new_zeros(())
    descriptor_loss = (
        descriptor_correspondence_loss(
            output["encoded_levels"],
            gt_subcloud_index,
            ground_truth_transform,
            normalization_scale,
        )
        if "encoded_levels" in output else zero
    )
    pose_loss = errors.min() / 3.0 if epoch >= correspondence_start_epoch else zero

    correspondence_terms: list[torch.Tensor] = []
    if epoch >= correspondence_start_epoch:
        for correspondence in output["candidate_correspondences"]:
            if len(correspondence["source"]) == 0:
                continue
            expected = apply_transform(correspondence["source"], ground_truth_transform)
            residual = torch.linalg.norm(expected - correspondence["target"], dim=1)
            inlier_target = (residual * normalization_scale <= 3.0).to(
                correspondence["scores"].dtype
            )
            probabilities = correspondence["scores"].clamp(
                min=torch.finfo(correspondence["scores"].dtype).eps,
                max=1.0 - torch.finfo(correspondence["scores"].dtype).eps,
            )
            with torch.autocast(device_type=probabilities.device.type, enabled=False):
                correspondence_terms.append(
                    F.binary_cross_entropy(probabilities.float(), inlier_target.float())
                )
    correspondence_loss = (
        torch.stack(correspondence_terms).mean() if correspondence_terms else zero
    )

    if epoch >= equivariant_loss_start_epoch and equivariant_loss_weight > 0:
        equivariant_loss = equivariant_feature_alignment_loss(
            output["encoded_levels"],
            gt_subcloud_index,
            ground_truth_transform,
            normalization_scale,
        )
    else:
        equivariant_loss = zero

    if epoch >= mpn_start_epoch:
        best_candidate = errors.argmin().reshape(1)
        mpn_loss = F.cross_entropy(output["final_logits"].unsqueeze(0), best_candidate)
    else:
        mpn_loss = zero
    total = (
        ops_loss
        + descriptor_loss
        + correspondence_loss
        + 0.1 * pose_loss
        + mpn_loss
        + equivariant_loss_weight * equivariant_loss
    )
    return {
        "total": total,
        "ops": ops_loss,
        "descriptor": descriptor_loss,
        "correspondence": correspondence_loss,
        "pose": pose_loss,
        "mpn": mpn_loss,
        "equivariant": equivariant_loss,
        "candidate_rmse_angstrom": errors.detach(),
    }
