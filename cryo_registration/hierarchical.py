from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from .fine_point_matching import FinePointPairTooLarge
from .fine_shot import select_fine_shot_levels
from .subclouds import build_target_subclouds, furthest_point_sample_by_coverage
from .model import _local_global_refinement, apply_transform
from .model import (
    MultiScalePostWeightingNetwork,
    ProteinRegistrationModel,
    SubcloudFeatureEncoder,
    _chain_subclouds,
    _limit_indices,
    _limit_points,
    compute_ops_scores,
    estimate_rigid_transform,
    compute_local_frames,
    build_equivariant_features,
)
from .turboclique import estimate_turboclique_from_correspondences


SCALE_KEYS = ("2.00", "4.00", "6.00")


@dataclass(frozen=True)
class ChainHalf:
    points: dict[str, torch.Tensor]
    indices: dict[str, torch.Tensor]



@dataclass(frozen=True)
class HalfGeometry:
    points: torch.Tensor
    valid_point_count: int


@dataclass(frozen=True)
class FineCandidate:
    subcloud_index: int
    transform: torch.Tensor
    source_correspondences: torch.Tensor
    target_correspondences: torch.Tensor
    correspondence_scores: torch.Tensor
    final_score: torch.Tensor
    source_points_full: dict[str, torch.Tensor] | None = None
    target_points_full: dict[str, torch.Tensor] | None = None



@dataclass(frozen=True)
class PairScoreComponents:
    total_score: torch.Tensor
    local_score: torch.Tensor
    adjacency_score: torch.Tensor
    pose_score: torch.Tensor
    left_point_weight: float
    right_point_weight: float
    topology_balance: float
    center_distance_error_A: float
    direction_error_deg: float
    interface_distance_error_A: float

@dataclass(frozen=True)
class PairFusion:
    transform: torch.Tensor
    score: torch.Tensor
    left_subcloud_index: int
    right_subcloud_index: int
    residual_A: float
    inlier_ratio_3A: float
    rotation_disagreement_deg: float
    translation_disagreement_A: float
    components: PairScoreComponents | None = None
    pose_backend: str = "lgr"
    left_final_score: torch.Tensor | None = None
    right_final_score: torch.Tensor | None = None

def split_chain_by_principal_axis(
    levels: dict[str, torch.Tensor],
    min_points: int = 3,
) -> tuple[ChainHalf, ChainHalf]:
    """Split a multi-scale chain with one plane fitted on the 6 A points."""
    _validate_levels(levels)
    if min_points <= 0:
        raise ValueError("min_points must be positive")
    coarse = levels["6.00"]
    if len(coarse) < 2 * min_points:
        raise ValueError("chain needs enough 6 A points for two halves")

    centered = coarse - coarse.mean(dim=0)
    _, _, vh = torch.linalg.svd(centered.to(torch.float32), full_matrices=False)
    axis = vh[0].to(dtype=coarse.dtype)
    projections = centered @ axis
    order = projections.argsort(stable=True)
    split = len(coarse) // 2
    coarse_left = order[:split]
    coarse_right = order[split:]
    threshold = 0.5 * (
        projections[coarse_left[-1]] + projections[coarse_right[0]]
    )
    origin = coarse.mean(dim=0)

    left_indices: dict[str, torch.Tensor] = {}
    right_indices: dict[str, torch.Tensor] = {}
    for key in SCALE_KEYS:
        points = levels[key]
        if key == "6.00":
            left, right = coarse_left, coarse_right
        else:
            projected = (points - origin) @ axis
            left = torch.nonzero(projected <= threshold, as_tuple=False).flatten()
            right = torch.nonzero(projected > threshold, as_tuple=False).flatten()
            if len(left) == 0 or len(right) == 0:
                level_order = projected.argsort(stable=True)
                level_split = max(1, min(len(points) - 1, len(points) // 2))
                left, right = level_order[:level_split], level_order[level_split:]
        left_indices[key] = left
        right_indices[key] = right

    return (
        ChainHalf(
            points={key: levels[key][left_indices[key]] for key in SCALE_KEYS},
            indices=left_indices,
        ),
        ChainHalf(
            points={key: levels[key][right_indices[key]] for key in SCALE_KEYS},
            indices=right_indices,
        ),
    )


def build_fine_target_subclouds(
    parent_points: dict[str, torch.Tensor],
    parent_indices: dict[str, torch.Tensor],
    half_points: dict[str, torch.Tensor],
    crop_diameter_factor: float = 1.25,
    point_cap_factor: float = 1.25,
) -> dict[str, dict[str, torch.Tensor]]:
    """Cover one coarse target patch with half-chain-sized fine patches."""
    _validate_levels(parent_points)
    _validate_levels(half_points)
    if crop_diameter_factor <= 0 or point_cap_factor <= 0:
        raise ValueError("fine subcloud factors must be positive")
    for key in SCALE_KEYS:
        indices = parent_indices[key]
        if indices.ndim != 1 or len(indices) != len(parent_points[key]):
            raise ValueError("parent indices must align with parent points")

    half_diameter = _point_cloud_diameter(half_points["6.00"])
    crop_diameter = crop_diameter_factor * half_diameter
    anchor_indices = furthest_point_sample_by_coverage(
        parent_points["6.00"],
        coverage_radius=crop_diameter / 2.0,
    )
    shared_anchors = parent_points["6.00"][anchor_indices]

    result: dict[str, dict[str, torch.Tensor]] = {}
    for key in SCALE_KEYS:
        point_cap = max(1, math.ceil(point_cap_factor * len(half_points[key])))
        sampled = build_target_subclouds(
            parent_points[key],
            center_count=len(shared_anchors),
            crop_diameter=crop_diameter,
            point_cap=point_cap,
            anchor_points=shared_anchors,
        )
        padded_global = torch.cat(
            [
                parent_indices[key],
                parent_indices[key].new_full((1,), -1),
            ]
        )
        result[key] = {
            "anchor_indices": sampled.anchor_indices,
            "anchors": sampled.anchors,
            "local_indices": sampled.indices,
            "indices": padded_global[sampled.indices],
            "masks": sampled.masks,
            "points": sampled.points,
        }
    return result

PAIR_SCORING_MODES = {
    "legacy",
    "point_weighted",
    "center",
    "direction",
    "topology",
}


def point_count_weights(
    left_count: int,
    right_count: int,
    minimum: int = 8,
) -> tuple[float, float, float]:
    if minimum <= 0:
        raise ValueError("minimum must be positive")
    if left_count < minimum or right_count < minimum:
        raise ValueError("each half needs at least eight valid 6 A points")
    total = left_count + right_count
    left_weight = left_count / total
    right_weight = right_count / total
    return left_weight, right_weight, 2.0 * min(left_weight, right_weight)


def normalize_half_correspondence_weights(
    left: torch.Tensor,
    right: torch.Tensor,
    left_mass: float,
    right_mass: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if left.numel() == 0 or right.numel() == 0:
        raise ValueError("both halves need correspondence weights")
    if left_mass <= 0 or right_mass <= 0:
        raise ValueError("half masses must be positive")

    def normalize(weights: torch.Tensor, mass: float) -> torch.Tensor:
        weights = weights.clamp_min(0)
        total = weights.sum()
        if float(total.detach().cpu()) <= torch.finfo(weights.dtype).eps:
            weights = torch.ones_like(weights)
            total = weights.sum()
        return weights / total * mass

    return normalize(left, left_mass), normalize(right, right_mass)


def score_topology_pair(
    left: FineCandidate,
    right: FineCandidate,
    left_geometry: HalfGeometry,
    right_geometry: HalfGeometry,
    fused_transform: torch.Tensor,
    residual_A: torch.Tensor,
    inlier_ratio: torch.Tensor,
    normalization_scale: float,
    scoring_mode: str = "topology",
) -> PairScoreComponents:
    if scoring_mode not in PAIR_SCORING_MODES - {"legacy"}:
        raise ValueError(f"unsupported topology scoring mode: {scoring_mode}")
    if normalization_scale <= 0:
        raise ValueError("normalization_scale must be positive")
    left_weight, right_weight, balance = point_count_weights(
        left_geometry.valid_point_count,
        right_geometry.valid_point_count,
    )
    local_score = (
        left_weight * left.final_score + right_weight * right.final_score
    )

    left_points = left_geometry.points
    right_points = right_geometry.points
    left_center = left_points.mean(dim=0)
    right_center = right_points.mean(dim=0)
    original_vector = right_center - left_center
    predicted_left = apply_transform(left_points, left.transform)
    predicted_right = apply_transform(right_points, right.transform)
    predicted_vector = predicted_right.mean(dim=0) - predicted_left.mean(dim=0)
    center_error = (
        torch.abs(
            torch.linalg.norm(predicted_vector)
            - torch.linalg.norm(original_vector)
        )
        * normalization_scale
    )
    expected_vector = fused_transform[:3, :3] @ original_vector
    direction_error = _vector_angle_degrees(predicted_vector, expected_vector)
    original_interface = _interface_distance(left_points, right_points)
    predicted_interface = _interface_distance(predicted_left, predicted_right)
    interface_error = (
        torch.abs(predicted_interface - original_interface)
        * normalization_scale
    )

    adjacency_score = local_score.new_tensor(0.0)
    if scoring_mode in {"center", "direction", "topology"}:
        adjacency_score = adjacency_score - center_error / 3.0
    if scoring_mode in {"direction", "topology"}:
        adjacency_score = adjacency_score - direction_error / 30.0
    if scoring_mode == "topology":
        adjacency_score = adjacency_score - interface_error / 3.0

    relative_rotation = left.transform[:3, :3].T @ right.transform[:3, :3]
    cosine = ((torch.trace(relative_rotation) - 1.0) / 2.0).clamp(-1.0, 1.0)
    rotation_disagreement = torch.rad2deg(torch.acos(cosine))
    translation_disagreement = (
        torch.linalg.norm(left.transform[:3, 3] - right.transform[:3, 3])
        * normalization_scale
    )
    residual_tensor = torch.as_tensor(
        residual_A,
        dtype=local_score.dtype,
        device=local_score.device,
    )
    inlier_tensor = torch.as_tensor(
        inlier_ratio,
        dtype=local_score.dtype,
        device=local_score.device,
    )
    pose_score = (
        -rotation_disagreement / 30.0
        - translation_disagreement / 6.0
        - residual_tensor / 3.0
        + torch.log(inlier_tensor.clamp_min(1e-6))
    )
    total_score = local_score + balance * adjacency_score + pose_score
    return PairScoreComponents(
        total_score=total_score,
        local_score=local_score,
        adjacency_score=adjacency_score,
        pose_score=pose_score,
        left_point_weight=left_weight,
        right_point_weight=right_weight,
        topology_balance=balance,
        center_distance_error_A=float(center_error.detach().cpu()),
        direction_error_deg=float(direction_error.detach().cpu()),
        interface_distance_error_A=float(interface_error.detach().cpu()),
    )


def _interface_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    distances = torch.cdist(left, right).flatten()
    smaller = min(len(left), len(right))
    count = min(16, max(3, math.ceil(0.1 * smaller)), len(distances))
    return distances.topk(count, largest=False).values.mean()


def _vector_angle_degrees(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(first.dtype).eps
    first_norm = torch.linalg.norm(first)
    second_norm = torch.linalg.norm(second)
    if float(torch.minimum(first_norm, second_norm).detach().cpu()) <= eps:
        return first.new_tensor(180.0)
    first_unit = first / first_norm
    second_unit = second / second_norm
    sine = torch.linalg.norm(torch.linalg.cross(first_unit, second_unit))
    cosine = torch.dot(first_unit, second_unit).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.atan2(sine, cosine))


def fuse_half_candidate_pairs(
    left_candidates: list[FineCandidate],
    right_candidates: list[FineCandidate],
    normalization_scale: float,
    left_geometry: HalfGeometry | None = None,
    right_geometry: HalfGeometry | None = None,
    scoring_mode: str = "legacy",
    pose_backend: str = "lgr",
) -> PairFusion:
    """Fuse all left/right Top-k pairs and return the best whole-chain pose."""
    if normalization_scale <= 0:
        raise ValueError("normalization_scale must be positive")
    if not left_candidates or not right_candidates:
        raise ValueError("at least one candidate per half is required for fusion")
    if scoring_mode not in PAIR_SCORING_MODES:
        raise ValueError(f"unsupported pair scoring mode: {scoring_mode}")
    if pose_backend not in {"lgr", "turboclique"}:
        raise ValueError(f"unsupported fusion pose backend: {pose_backend}")
    if scoring_mode != "legacy" and (
        left_geometry is None or right_geometry is None
    ):
        raise ValueError("topology scoring requires both half geometries")
    point_masses: tuple[float, float, float] | None = None
    if scoring_mode != "legacy":
        point_masses = point_count_weights(
            left_geometry.valid_point_count,
            right_geometry.valid_point_count,
        )

    best: PairFusion | None = None
    best_value = -math.inf
    for left in left_candidates:
        for right in right_candidates:
            source = torch.cat(
                [left.source_correspondences, right.source_correspondences], dim=0
            )
            target = torch.cat(
                [left.target_correspondences, right.target_correspondences], dim=0
            )
            left_weights = left.correspondence_scores
            right_weights = right.correspondence_scores
            if point_masses is not None:
                if left_weights.numel() == 0 or right_weights.numel() == 0:
                    continue
                left_weights, right_weights = normalize_half_correspondence_weights(
                    left_weights,
                    right_weights,
                    point_masses[0],
                    point_masses[1],
                )
            weights = torch.cat([left_weights, right_weights], dim=0)
            if pose_backend == "turboclique" and len(weights) > 512:
                keep = weights.topk(512).indices
                source = source[keep]
                target = target[keep]
                weights = weights[keep]
            if len(source) < 3 or len(source) != len(target):
                continue
            try:
                if pose_backend == "lgr":
                    transform = _local_global_refinement(
                        source,
                        target,
                        weights,
                        acceptance_radius=3.0 / normalization_scale,
                        max_hypotheses=32,
                        refinement_steps=5,
                    )
                else:
                    transform = estimate_turboclique_from_correspondences(
                        source,
                        target,
                        weights,
                        distance_tolerance=6.0 / normalization_scale,
                        minimum_baseline=12.0 / normalization_scale,
                        inlier_threshold=9.0 / normalization_scale,
                        max_pivots=2000,
                        refinement_steps=20,
                    ).transform
            except (RuntimeError, ValueError):
                continue

            residuals_A = (
                torch.linalg.norm(apply_transform(source, transform) - target, dim=1)
                * normalization_scale
            )
            residual_A = residuals_A.mean()
            inlier_ratio = (residuals_A <= 3.0).float().mean()
            relative_rotation = left.transform[:3, :3].T @ right.transform[:3, :3]
            cosine = ((torch.trace(relative_rotation) - 1.0) / 2.0).clamp(-1.0, 1.0)
            rotation_disagreement = torch.rad2deg(torch.acos(cosine))
            translation_disagreement = (
                torch.linalg.norm(left.transform[:3, 3] - right.transform[:3, 3])
                * normalization_scale
            )
            components = None
            if scoring_mode == "legacy":
                score = (
                    left.final_score
                    + right.final_score
                    - rotation_disagreement / 30.0
                    - translation_disagreement / 6.0
                    - residual_A / 3.0
                    + torch.log(inlier_ratio.clamp_min(1e-6))
                )
            else:
                components = score_topology_pair(
                    left,
                    right,
                    left_geometry,
                    right_geometry,
                    transform,
                    residual_A,
                    inlier_ratio,
                    normalization_scale,
                    scoring_mode,
                )
                score = components.total_score
            value = float(score.detach().cpu())
            if value > best_value:
                best_value = value
                best = PairFusion(
                    transform=transform,
                    score=score,
                    left_subcloud_index=left.subcloud_index,
                    right_subcloud_index=right.subcloud_index,
                    residual_A=float(residual_A.detach().cpu()),
                    inlier_ratio_3A=float(inlier_ratio.detach().cpu()),
                    rotation_disagreement_deg=float(
                        rotation_disagreement.detach().cpu()
                    ),
                    translation_disagreement_A=float(
                        translation_disagreement.detach().cpu()
                    ),
                    components=components,
                    pose_backend=pose_backend,
                    left_final_score=left.final_score.detach(),
                    right_final_score=right.final_score.detach(),
                )
    if best is None:
        raise ValueError("unable to fuse half-chain correspondences")
    return best



def _point_cloud_diameter(points: torch.Tensor) -> float:
    if len(points) < 2:
        raise ValueError("at least two half-chain points are required")
    maximum = points.new_tensor(0.0)
    for chunk in points.split(1024):
        maximum = torch.maximum(maximum, torch.cdist(chunk, points).max())
    diameter = float(maximum.item())
    if not math.isfinite(diameter) or diameter <= 0:
        raise ValueError("half-chain diameter must be finite and positive")
    return diameter


def _validate_levels(levels: dict[str, torch.Tensor]) -> None:
    missing = set(SCALE_KEYS) - set(levels)
    if missing:
        raise ValueError(f"missing point-cloud levels: {sorted(missing)}")
    for key in SCALE_KEYS:
        points = levels[key]
        if (
            not isinstance(points, torch.Tensor)
            or points.ndim != 2
            or points.shape[1] != 3
            or len(points) == 0
        ):
            raise ValueError(f"{key} points must have shape (N, 3)")


class HierarchicalRegistrationRefiner(nn.Module):
    """Run half-chain fine registration inside each coarse Top-k region."""

    def __init__(
        self,
        coarse_model: ProteinRegistrationModel,
        coarse_output_topk: int = 3,
        fine_ops_topk: int = 6,
        fine_output_topk: int = 6,
        fine_crop_diameter_factor: float = 1.25,
        fine_point_cap_factor: float = 1.25,
        fine_max_points_per_patch: int | None = 3000,
        fine_min_valid_points: int = 3,
        fine_min_half_points: int = 8,
        pair_scoring_mode: str = "legacy",
        fine_encoder: SubcloudFeatureEncoder | None = None,
        fine_cross_attention: nn.Module | None = None,
        fine_point_matcher: nn.Module | None = None,
        fine_learned_feature_weight: float = 0.25,
        use_equivariant_pose: bool | None = None,
        equivariant_feature_dim: int | None = None,
        equivariant_max_hypotheses: int | None = None,
        equivariant_acceptance_radius_angstrom: float | None = None,
    ) -> None:
        super().__init__()
        if coarse_output_topk <= 0 or fine_ops_topk <= 0 or fine_output_topk <= 0:
            raise ValueError("hierarchical Top-k values must be positive")
        if fine_output_topk > fine_ops_topk:
            raise ValueError("fine_output_topk cannot exceed fine_ops_topk")
        if fine_min_valid_points < 3:
            raise ValueError("fine_min_valid_points must be at least three")
        if fine_min_half_points < 8:
            raise ValueError("fine_min_half_points must be at least eight")
        if fine_max_points_per_patch is not None and fine_max_points_per_patch < 1:
            raise ValueError("fine_max_points_per_patch must be positive")
        if pair_scoring_mode not in PAIR_SCORING_MODES:
            raise ValueError("unsupported pair scoring mode")
        self.coarse_model = coarse_model
        hidden_dim = coarse_model.mpn.input[0].out_features
        self.fine_mpn = MultiScalePostWeightingNetwork(
            input_dim=8,
            hidden_dim=hidden_dim,
            num_heads=coarse_model.mpn.context.num_heads,
        )
        self.coarse_output_topk = coarse_output_topk
        self.fine_ops_topk = fine_ops_topk
        self.fine_output_topk = fine_output_topk
        self.fine_crop_diameter_factor = fine_crop_diameter_factor
        self.fine_point_cap_factor = fine_point_cap_factor
        self.fine_max_points_per_patch = fine_max_points_per_patch
        self.fine_min_valid_points = fine_min_valid_points
        self.fine_min_half_points = fine_min_half_points
        self.pair_scoring_mode = pair_scoring_mode
        self.fine_encoder = fine_encoder
        self.fine_cross_attention = fine_cross_attention
        self.fine_point_matcher = fine_point_matcher
        self.fine_learned_feature_weight = float(fine_learned_feature_weight)
        self.use_equivariant_pose = (
            coarse_model.use_equivariant_pose
            if use_equivariant_pose is None else use_equivariant_pose
        )
        self.equivariant_feature_dim = (
            coarse_model.equivariant_feature_dim
            if equivariant_feature_dim is None else equivariant_feature_dim
        )
        self.equivariant_max_hypotheses = (
            coarse_model.equivariant_max_hypotheses
            if equivariant_max_hypotheses is None else equivariant_max_hypotheses
        )
        self.equivariant_acceptance_radius_angstrom = (
            coarse_model.equivariant_acceptance_radius_angstrom
            if equivariant_acceptance_radius_angstrom is None
            else equivariant_acceptance_radius_angstrom
        )

    def forward(
        self,
        structure: dict,
        chain_id: str,
        source_transform: torch.Tensor | None = None,
    ) -> dict:
        encoded_target = self.coarse_model.encode_target(structure, chain_id)
        coarse = self.coarse_model(
            structure,
            chain_id,
            source_transform=source_transform,
            encoded_target=encoded_target,
        )
        coarse_count = min(self.coarse_output_topk, len(coarse["final_logits"]))
        coarse_local_indices = coarse["final_logits"].topk(coarse_count).indices
        if self.pair_scoring_mode == "legacy":
            coarse_scores = coarse["final_logits"][coarse_local_indices]
        else:
            coarse_scores = F.log_softmax(
                coarse["final_logits"][coarse_local_indices], dim=0
            )
        scale = float(structure["normalization"]["scale"])
        subclouds = _chain_subclouds(structure, chain_id)
        try:
            halves = split_chain_by_principal_axis(
                structure["chains"][chain_id],
                min_points=self.fine_min_valid_points,
            )
            split_error = None
        except ValueError as error:
            halves = None
            split_error = str(error)

        transforms: list[torch.Tensor] = []
        scores: list[torch.Tensor] = []
        statuses: list[str] = []
        diagnostics: list[dict] = []
        parent_indices: list[int] = []
        coarse_scores_list: list[torch.Tensor] = []
        fusion_scores_list: list[torch.Tensor] = []
        local_scores_list: list[torch.Tensor] = []
        refined_mask_list: list[torch.Tensor] = []
        for coarse_rank, coarse_local_tensor in enumerate(coarse_local_indices):
            coarse_local = int(coarse_local_tensor.item())
            parent_index = int(coarse["candidate_indices"][coarse_local].item())
            parent_indices.append(parent_index)
            coarse_transform = coarse["candidate_transforms"][coarse_local]
            coarse_score = coarse_scores[coarse_rank]
            if halves is None:
                transforms.append(coarse_transform)
                scores.append(coarse_score)
                coarse_scores_list.append(coarse_score.detach())
                fusion_scores_list.append(coarse_score.new_zeros(()))
                local_scores_list.append(coarse_score.new_zeros(()))
                refined_mask_list.append(torch.tensor(False, device=coarse_score.device))
                statuses.append(f"fallback:{split_error}")
                diagnostics.append({"halves": [], "error": split_error})
                continue
            half_outputs: list[dict] = []
            try:
                parent_points, parent_global_indices = self._parent_region(
                    structure, subclouds, parent_index
                )
                fine_subclouds = [
                    build_fine_target_subclouds(
                        parent_points,
                        parent_global_indices,
                        half.points,
                        crop_diameter_factor=self.fine_crop_diameter_factor,
                        point_cap_factor=self.fine_point_cap_factor,
                    )
                    for half in halves
                ]
                half_outputs = [
                    self._run_fine_stage(
                        structure,
                        chain_id,
                        half,
                        fine,
                        scale,
                        source_transform,
                    )
                    for half, fine in zip(halves, fine_subclouds)
                ]
                fusion = fuse_half_candidate_pairs(
                    half_outputs[0]["top_candidates"],
                    half_outputs[1]["top_candidates"],
                    normalization_scale=scale,
                    left_geometry=half_outputs[0]["source_geometry"],
                    right_geometry=half_outputs[1]["source_geometry"],
                    scoring_mode=self.pair_scoring_mode,
                )
            except (RuntimeError, ValueError) as error:
                transforms.append(coarse_transform)
                scores.append(coarse_score)
                coarse_scores_list.append(coarse_score.detach())
                fusion_scores_list.append(coarse_score.new_zeros(()))
                local_scores_list.append(coarse_score.new_zeros(()))
                refined_mask_list.append(torch.tensor(False, device=coarse_score.device))
                statuses.append(f"fallback:{error}")
                diagnostics.append({"halves": half_outputs, "error": str(error)})
                continue

            transforms.append(fusion.transform)
            scores.append(coarse_score + fusion.score)
            coarse_scores_list.append(coarse_score.detach())
            fusion_scores_list.append(fusion.score.detach())
            local_score = fusion.left_final_score + fusion.right_final_score
            local_scores_list.append(local_score.detach())
            refined_mask_list.append(torch.tensor(True, device=coarse_score.device))
            statuses.append("refined")
            diagnostics.append(
                {
                    "halves": half_outputs,
                    "fusion": fusion,
                }
            )

        candidate_transforms = torch.stack(transforms)
        candidate_scores = torch.stack(scores)
        best_index = candidate_scores.argmax()
        parent_tensor = torch.as_tensor(
            parent_indices,
            dtype=torch.long,
            device=candidate_scores.device,
        )
        return {
            "candidate_transforms": candidate_transforms,
            "candidate_scores": candidate_scores,
            "candidate_coarse_scores": torch.stack(coarse_scores_list),
            "candidate_fusion_scores": torch.stack(fusion_scores_list),
            "candidate_local_scores": torch.stack(local_scores_list),
            "candidate_refined_mask": torch.stack(refined_mask_list),
            "coarse_subcloud_indices": parent_tensor,
            "coarse_local_indices": coarse_local_indices,
            "refinement_status": statuses,
            "fine_diagnostics": diagnostics,
            "best_candidate_index": best_index,
            "best_subcloud_index": parent_tensor[best_index],
            "best_transform": candidate_transforms[best_index],
            "coarse_output": coarse,
            "pose_scale": "6.00",
        }

    def _parent_region(
        self,
        structure: dict,
        subclouds: dict[str, dict[str, torch.Tensor]],
        parent_index: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        points: dict[str, torch.Tensor] = {}
        indices: dict[str, torch.Tensor] = {}
        for key in SCALE_KEYS:
            level = subclouds[key]
            mask = level["masks"][parent_index]
            parent_points = level["points"][parent_index][mask]
            if len(parent_points) == 0:
                raise ValueError("coarse parent region is empty")
            points[key] = parent_points
            if "indices" in level:
                indices[key] = level["indices"][parent_index][mask]
            else:
                indices[key] = torch.cdist(
                    parent_points, structure["target"][key]
                ).argmin(dim=1)
        return points, indices

    def _run_fine_stage(
        self,
        structure: dict,
        chain_id: str,
        half: ChainHalf,
        fine_subclouds: dict[str, dict[str, torch.Tensor]],
        normalization_scale: float,
        source_transform: torch.Tensor | None,
    ) -> dict:
        geometry: dict[str, dict[str, torch.Tensor]] = {}
        source_original_indices: dict[str, torch.Tensor] = {}
        for key in SCALE_KEYS:
            points = half.points[key]
            keep = _limit_indices(
                len(points),
                self.fine_max_points_per_patch,
                points.device,
            )
            source = points[keep]
            if source_transform is not None:
                source = apply_transform(source, source_transform)
            source_original_indices[key] = half.indices[key][keep]
            geometry[key] = {
                "source_points": source,
                "target_points": _limit_points(
                    fine_subclouds[key]["points"],
                    self.fine_max_points_per_patch,
                ),
                "target_masks": _limit_points(
                    fine_subclouds[key]["masks"],
                    self.fine_max_points_per_patch,
                ),
            }

        source_shot_level, target_shot = select_fine_shot_levels(
            structure, chain_id
        )
        source_shot_features = source_shot_level["features"][
            source_original_indices["4.00"]
        ]
        source_shot_valid = source_shot_level["valid_mask"][
            source_original_indices["4.00"]
        ]
        source_valid_point_count = int(source_shot_valid.sum().item())
        if (
            self.pair_scoring_mode != "legacy"
            and source_valid_point_count < self.fine_min_half_points
        ):
            raise ValueError("each half needs at least eight valid 6 A points")
        if source_valid_point_count < self.fine_min_valid_points:
            source_shot_features = torch.zeros_like(source_shot_features)
            source_shot_valid = torch.ones_like(source_shot_valid)
        source_mask = torch.ones_like(source_shot_valid).unsqueeze(0)
        enc = self.fine_encoder if self.fine_encoder is not None else self.coarse_model.encoder
        source_encoded, _ = enc(
            geometry["4.00"]["source_points"].unsqueeze(0)
            * (normalization_scale / 4.0),
            source_mask,
            source_shot_features.unsqueeze(0),
            source_shot_valid.unsqueeze(0),
        )
        source_features = torch.cat(
            [
                F.normalize(source_shot_features[source_shot_valid], dim=-1),
                self.fine_learned_feature_weight
                * F.normalize(
                    source_encoded.squeeze(0)[source_shot_valid], dim=-1
                ),
            ],
            dim=-1,
        )
        source_points = geometry["4.00"]["source_points"][source_shot_valid]

        source_geometry = HalfGeometry(
            points=source_points,
            valid_point_count=source_valid_point_count,
        )
        target_indices = _limit_points(
            fine_subclouds["4.00"]["indices"],
            self.fine_max_points_per_patch,
        )
        padded_features = torch.cat(
            [
                target_shot["features"],
                target_shot["features"].new_zeros(
                    1, target_shot["features"].shape[1]
                ),
            ],
            dim=0,
        )
        padded_valid = torch.cat(
            [
                target_shot["valid_mask"],
                target_shot["valid_mask"].new_zeros(1),
            ],
            dim=0,
        )
        safe_indices = target_indices.masked_fill(
            target_indices < 0, len(target_shot["features"])
        )
        target_shot_features = padded_features[safe_indices]
        target_shot_valid = padded_valid[safe_indices]
        target_masks = geometry["4.00"]["target_masks"]
        target_encoded, _ = enc(
            geometry["4.00"]["target_points"] * (normalization_scale / 4.0),
            target_masks,
            target_shot_features,
            target_shot_valid,
        )
        target_features = torch.cat(
            [
                F.normalize(target_shot_features, dim=-1),
                self.fine_learned_feature_weight * F.normalize(target_encoded, dim=-1),
            ],
            dim=-1,
        )
        matching_masks = target_masks & target_shot_valid
        if self.fine_cross_attention is None:
            pair_source_features = source_features.unsqueeze(0).expand(
                target_features.shape[0], -1, -1
            )
        else:
            pair_source_features, target_features = self.fine_cross_attention(
                source_features,
                target_features,
                matching_masks,
            )
        source_equivariant = None
        if self.use_equivariant_pose and len(source_points) >= 3:
            source_scalar_features = source_encoded.squeeze(0)[source_shot_valid]
            source_equivariant = self.coarse_model._equivariant_vectors(
                source_points, source_scalar_features
            )
        target_equivariant = None
        if self.use_equivariant_pose:
            target_equivariant_rows = []
            for patch_points, patch_mask, patch_scalar_features in zip(
                geometry["4.00"]["target_points"],
                matching_masks,
                target_encoded,
            ):
                full_vectors = patch_points.new_zeros(
                    patch_points.shape[0], self.equivariant_feature_dim, 3
                )
                valid_points = patch_points[patch_mask]
                if len(valid_points) >= 3:
                    vectors = self.coarse_model._equivariant_vectors(
                        valid_points, patch_scalar_features[patch_mask]
                    )
                    if vectors is not None:
                        full_vectors[patch_mask] = vectors
                target_equivariant_rows.append(full_vectors)
            target_equivariant = torch.stack(target_equivariant_rows)
        coarse = {
            "source_points": source_points,
            "source_features": source_features,
            "source_equivariant": source_equivariant,
            "target_points": geometry["4.00"]["target_points"],
            "target_features": target_features,
            "target_equivariant": target_equivariant,
            "target_masks": matching_masks,
        }
        ops_scores = torch.stack(
            [
                compute_ops_scores(
                    pair_source_features[index],
                    target_features[index : index + 1],
                    matching_masks[index : index + 1],
                )[0]
                for index in range(target_features.shape[0])
            ]
        )
        eligible = matching_masks.sum(dim=1) >= self.fine_min_valid_points
        eligible_count = int(eligible.sum().item())
        if eligible_count == 0:
            raise ValueError("no fine target patch can support LGR")
        ops_scores = ops_scores.masked_fill(~eligible, -1e4)
        candidate_count = min(self.fine_ops_topk, eligible_count)
        candidate_indices = ops_scores.topk(candidate_count).indices
        transforms: list[torch.Tensor] = []
        correspondences: list[dict[str, torch.Tensor]] = []
        summaries: list[torch.Tensor] = []
        point_matcher_statuses: list[str] = []
        point_matcher_outputs: list[object | None] = []
        point_matcher_inputs: list[dict[str, torch.Tensor]] = []
        for candidate_index in candidate_indices.tolist():
            valid = matching_masks[candidate_index]
            point_output = None
            point_status = "disabled"
            pose_source_features = pair_source_features[candidate_index]
            pose_target_features = target_features[candidate_index][valid]
            pose_source_equivariant = source_equivariant
            pose_target_equivariant = None
            if coarse["target_equivariant"] is not None:
                pose_target_equivariant = coarse["target_equivariant"][
                    candidate_index
                ][valid]
            point_matcher_input = {
                "source_points": source_points,
                "target_points": coarse["target_points"][candidate_index][valid],
                "source_shot": source_shot_features[source_shot_valid],
                "target_shot": target_shot_features[candidate_index][valid],
                "source_encoded": source_encoded.squeeze(0)[source_shot_valid],
                "target_encoded": target_encoded[candidate_index][valid],
            }
            point_matcher_inputs.append(point_matcher_input)
            if self.fine_point_matcher is not None:
                try:
                    point_output = self.fine_point_matcher(
                        source_points,
                        coarse["target_points"][candidate_index][valid],
                        source_shot_features[source_shot_valid],
                        target_shot_features[candidate_index][valid],
                        source_encoded.squeeze(0)[source_shot_valid],
                        target_encoded[candidate_index][valid],
                    )
                except FinePointPairTooLarge as error:
                    point_status = (
                        f"oversized:{error.pair_elements}>{error.limit}"
                    )
                else:
                    point_status = "ok"
                    pose_source_features = point_output.source_descriptors
                    pose_target_features = point_output.target_descriptors
                    pose_source_equivariant = point_output.source_equivariant
                    pose_target_equivariant = point_output.target_equivariant
            point_matcher_statuses.append(point_status)
            point_matcher_outputs.append(point_output)
            try:
                if int(valid.sum().item()) < self.fine_min_valid_points:
                    raise ValueError("fine target patch has too few valid points")
                pose = estimate_rigid_transform(
                    source_points,
                    coarse["target_points"][candidate_index][valid],
                    pose_source_features,
                    pose_target_features,
                    mutual_topk=self.coarse_model.mutual_topk,
                    acceptance_radius=(
                        self.equivariant_acceptance_radius_angstrom
                        / normalization_scale
                    ),
                    max_hypotheses=self.equivariant_max_hypotheses,
                    use_equivariant=(
                        self.use_equivariant_pose
                        and pose_source_equivariant is not None
                        and pose_target_equivariant is not None
                    ),
                    src_equivariant=pose_source_equivariant,
                    tgt_equivariant=pose_target_equivariant,
                )
                if pose.transform is None:
                    raise ValueError(pose.fallback_reason or "invalid_pose")
                transform = pose.transform.detach()
                correspondence = {
                    "source": pose.source_correspondences,
                    "target": pose.target_correspondences,
                    "scores": pose.correspondence_scores,
                }
            except (RuntimeError, ValueError):
                transform = torch.eye(
                    4,
                    dtype=source_points.dtype,
                    device=source_points.device,
                )
                correspondence = {
                    "source": source_points[:0],
                    "target": source_points[:0],
                    "scores": source_points.new_zeros(0),
                }
            transforms.append(transform)
            correspondences.append(correspondence)
            if valid.any():
                summaries.append(
                    ProteinRegistrationModel._candidate_summary(
                        geometry,
                        {**coarse, "source_features": pair_source_features[candidate_index]},
                        candidate_index,
                        transform,
                        ops_scores[candidate_index],
                        normalization_scale,
                        len(correspondence["source"]),
                    )
                )
            else:
                summaries.append(
                    ops_scores.new_tensor(
                        [ops_scores[candidate_index], 20, 20, 20, 0, 0, 0, 0]
                    )
                )

        candidate_transforms = torch.stack(transforms)
        summary_tensor = torch.stack(summaries)
        mpn_logits = self.fine_mpn(summary_tensor)
        final_logits = (
            torch.log(ops_scores[candidate_indices].clamp_min(1e-8)) + mpn_logits
        )
        top_count = min(self.fine_output_topk, len(final_logits))
        top_local = final_logits.topk(top_count).indices
        top_candidates = []
        if self.pair_scoring_mode == "legacy":
            top_scores = final_logits[top_local]
        else:
            top_scores = F.log_softmax(final_logits[top_local], dim=0)
        for rank, local_tensor in enumerate(top_local):
            local = int(local_tensor.item())
            correspondence = correspondences[local]
            top_candidates.append(
                FineCandidate(
                    subcloud_index=int(candidate_indices[local].item()),
                    transform=candidate_transforms[local],
                    source_correspondences=correspondence["source"],
                    target_correspondences=correspondence["target"],
                    correspondence_scores=correspondence["scores"],
                    final_score=top_scores[rank],
                    source_points_full={
                        key: geometry[key]["source_points"]
                        for key in SCALE_KEYS
                    },
                    target_points_full={
                        key: geometry[key]["target_points"][int(candidate_indices[local].item())]
                        for key in SCALE_KEYS
                    },
                )
            )
        return {
            "source_geometry": source_geometry,
            "target_subclouds": fine_subclouds,
            "ops_scores": ops_scores,
            "candidate_indices": candidate_indices,
            "candidate_transforms": candidate_transforms,
            "candidate_correspondences": correspondences,
            "point_matcher_statuses": point_matcher_statuses,
            "point_matcher_outputs": point_matcher_outputs,
            "point_matcher_inputs": point_matcher_inputs,
            "candidate_summaries": summary_tensor,
            "mpn_logits": mpn_logits,
            "final_logits": final_logits,
            "top_local_indices": top_local,
            "top_candidates": top_candidates,
            "encoded_levels": {"4.00": coarse},
        }
