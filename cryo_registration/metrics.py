from __future__ import annotations

import math

import torch

from .training import compute_chain_rmse_angstrom


def registration_metrics(
    output: dict,
    augmented_source_points: torch.Tensor,
    ground_truth_transform: torch.Tensor,
    gt_subcloud_index: int,
    normalization_scale: float,
    positive_subcloud_mask: torch.Tensor | None = None,
) -> dict[str, float]:
    predicted = output["best_transform"]
    relative_rotation = predicted[:3, :3] @ ground_truth_transform[:3, :3].T
    cosine = ((torch.trace(relative_rotation) - 1.0) / 2.0).clamp(-1.0, 1.0)
    rotation_error = torch.rad2deg(torch.acos(cosine))
    translation_error = (
        torch.linalg.norm(predicted[:3, 3] - ground_truth_transform[:3, 3])
        * normalization_scale
    )
    rmse = compute_chain_rmse_angstrom(
        augmented_source_points,
        predicted,
        ground_truth_transform,
        normalization_scale,
    )
    candidates = output["candidate_indices"].tolist()
    if positive_subcloud_mask is None:
        positive_subcloud_mask = torch.zeros(
            len(output["ops_scores"]) if "ops_scores" in output else max(candidates) + 1,
            dtype=torch.bool,
        )
        positive_subcloud_mask[gt_subcloud_index] = True
    positive_subcloud_mask = positive_subcloud_mask.cpu()
    candidate_hit = any(bool(positive_subcloud_mask[index]) for index in candidates)
    best_hit = bool(positive_subcloud_mask[int(output["best_subcloud_index"].item())])
    return {
        "ops_topk_recall": float(candidate_hit),
        "mpn_top1_accuracy": float(best_hit),
        "rotation_error_degrees": float(rotation_error.detach().cpu()),
        "translation_error_angstrom": float(translation_error.detach().cpu()),
        "rmse_angstrom": float(rmse.detach().cpu()),
        "registration_success": float(rmse.detach().cpu() <= 3.0),
    }
