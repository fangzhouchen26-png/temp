from __future__ import annotations

from dataclasses import dataclass
import itertools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fcw import FeatureConsistencyWeighting


class KernelPointConv(nn.Module):
    """Device-safe rigid KPConv adapted from SuP's KPConv implementation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        radius: float,
        kernel_points: int = 15,
        max_neighbors: int = 32,
    ) -> None:
        super().__init__()
        if kernel_points < 2 or radius <= 0:
            raise ValueError("kernel_points must be >= 2 and radius must be positive")
        self.radius = radius
        self.sigma = 1.5 * radius / (kernel_points - 1)
        self.max_neighbors = max_neighbors
        self.weights = nn.Parameter(torch.empty(kernel_points, in_channels, out_channels))
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        self.register_buffer("kernel_radii", torch.linspace(0.0, radius, kernel_points))

    def forward(
        self, points: torch.Tensor, features: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        batch_size, point_count, _ = points.shape
        neighbor_count = min(self.max_neighbors, point_count)
        distances = torch.cdist(points, points)
        valid_pairs = mask.unsqueeze(2) & mask.unsqueeze(1) & (distances <= self.radius)
        ranked = distances.masked_fill(~valid_pairs, torch.inf)
        neighbor_distances, neighbor_indices = ranked.topk(
            neighbor_count, dim=2, largest=False
        )
        neighbor_valid = torch.isfinite(neighbor_distances)
        batch_indices = torch.arange(batch_size, device=points.device)[:, None, None]
        neighbor_features = features[batch_indices, neighbor_indices]
        radial_delta = (neighbor_distances.unsqueeze(-1) - self.kernel_radii).abs()
        influence = (1.0 - radial_delta / self.sigma).clamp_min(0.0)
        influence = influence * neighbor_valid.unsqueeze(-1)
        aggregated = torch.einsum("bnhk,bnhc->bnkc", influence, neighbor_features)
        output = torch.einsum("bnkc,kco->bno", aggregated, self.weights)
        normalizer = neighbor_valid.sum(dim=2).clamp_min(1).unsqueeze(-1)
        return (output / normalizer) * mask.unsqueeze(-1)


class SubcloudFeatureEncoder(nn.Module):
    """SHOT projection -> radial KPConv -> coordinate-free self-attention."""

    def __init__(
        self,
        input_dim: int = 352,
        feature_dim: int = 64,
        num_heads: int = 4,
        radius: float = 0.1,
        kernel_points: int = 15,
        self_layers: int = 2,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
        )
        self.kpconv = KernelPointConv(
            feature_dim, feature_dim, radius, kernel_points
        )
        layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 2,
            dropout=0.0,
            batch_first=True,
            norm_first=False,
        )
        self.attention = nn.TransformerEncoder(
            layer, num_layers=self_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(feature_dim)

    def forward(
        self,
        points: torch.Tensor,
        mask: torch.Tensor,
        input_features: torch.Tensor,
        feature_valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if points.ndim != 3 or points.shape[-1] != 3 or mask.shape != points.shape[:2]:
            raise ValueError("points/mask must have shapes (B, N, 3) and (B, N)")
        if input_features.shape[:2] != points.shape[:2] or input_features.ndim != 3:
            raise ValueError("input_features must have shape (B, N, C)")
        if feature_valid_mask is None:
            feature_valid_mask = mask
        if feature_valid_mask.shape != mask.shape:
            raise ValueError("feature_valid_mask must have shape (B, N)")
        effective_mask = mask & feature_valid_mask
        safe_mask = effective_mask.clone()
        empty_rows = ~safe_mask.any(dim=1)
        safe_mask[empty_rows, 0] = True
        projected = self.input_projection(input_features)
        projected = projected * effective_mask.unsqueeze(-1)
        features = self.kpconv(points, projected, effective_mask)
        features = self.attention(features, src_key_padding_mask=~safe_mask)
        features = self.norm(features + projected) * effective_mask.unsqueeze(-1)
        descriptor = features.sum(dim=1) / effective_mask.sum(
            dim=1, keepdim=True
        ).clamp_min(1)
        return features, descriptor




class EquivariantVectorHead(nn.Module):
    """Learn vector features from invariant features and local relative vectors.

    Pair weights are predicted only from scalar features and pairwise
    distances. The aggregated values are relative coordinate vectors, so the
    output transforms as v -> Rv under a rigid rotation.
    """

    def __init__(
        self,
        scalar_dim: int,
        channels: int = 8,
        hidden_dim: int | None = None,
        neighbor_k: int = 16,
    ) -> None:
        super().__init__()
        if scalar_dim <= 0 or channels <= 0 or neighbor_k <= 0:
            raise ValueError("scalar_dim, channels and neighbor_k must be positive")
        hidden_dim = hidden_dim or max(32, scalar_dim)
        self.channels = channels
        self.neighbor_k = neighbor_k
        self.pair_mlp = nn.Sequential(
            nn.Linear(2 * scalar_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, channels),
        )

    def forward(
        self, points: torch.Tensor, scalar_features: torch.Tensor
    ) -> torch.Tensor:
        if points.ndim != 2 or points.shape[-1] != 3:
            raise ValueError("points must have shape (N, 3)")
        if scalar_features.ndim != 2 or scalar_features.shape[0] != len(points):
            raise ValueError("scalar_features must have shape (N, D)")
        n = len(points)
        output = scalar_features.new_zeros(n, self.channels, 3)
        if n < 2:
            return output

        geometry = points.float()
        scalars = scalar_features.float()
        k = min(self.neighbor_k, n - 1)
        distances = torch.cdist(geometry, geometry)
        distances.fill_diagonal_(torch.inf)
        neighbor_distances, neighbor_indices = distances.topk(
            k, dim=1, largest=False
        )
        neighbor_points = geometry[neighbor_indices]
        relative = neighbor_points - geometry[:, None, :]
        neighbor_scalars = scalars[neighbor_indices]
        center_scalars = scalars[:, None, :].expand(-1, k, -1)
        pair_input = torch.cat(
            [center_scalars, neighbor_scalars, neighbor_distances.unsqueeze(-1)],
            dim=-1,
        )
        logits = self.pair_mlp(pair_input)
        weights = torch.softmax(logits, dim=1)
        vectors = torch.einsum("nkc,nkd->ncd", weights, relative)
        vectors = F.normalize(vectors, dim=-1, eps=1e-6)
        return vectors.to(scalar_features.dtype)


def compute_local_frames(
    points: torch.Tensor, k: int = 16, chunk_size: int = 512
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute PCA frames with chunked kNN and explicit degeneracy flags."""
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if k < 1 or chunk_size < 1:
        raise ValueError("k and chunk_size must be positive")
    n = len(points)
    eye = torch.eye(3, dtype=points.dtype, device=points.device)
    if n < 3:
        return eye.expand(n, 3, 3).clone(), torch.ones(n, dtype=torch.bool, device=points.device)
    neighbor_count = min(k, n - 1)
    frames = eye.expand(n, 3, 3).clone()
    degenerate = torch.ones(n, dtype=torch.bool, device=points.device)
    for start in range(0, n, chunk_size):
        stop = min(n, start + chunk_size)
        distances = torch.cdist(points[start:stop], points)
        local_rows = torch.arange(stop - start, device=points.device)
        global_cols = torch.arange(start, stop, device=points.device)
        distances[local_rows, global_cols] = torch.inf
        indices = distances.topk(neighbor_count, dim=1, largest=False).indices
        for local_index, global_index in enumerate(range(start, stop)):
            neighbors = points[indices[local_index]]
            centered = neighbors - neighbors.mean(dim=0, keepdim=True)
            try:
                _, singular_values, vh = torch.linalg.svd(
                    centered.to(torch.float32), full_matrices=False
                )
            except RuntimeError:
                continue
            frame = vh[:3].T.to(points.dtype)
            ratio = singular_values[-1] / singular_values[0].clamp_min(1e-12)
            is_degenerate = bool(ratio < 1e-4)
            direction = neighbors.mean(dim=0) - points[global_index]
            for axis in range(3):
                dot = torch.dot(frame[:, axis], direction)
                if float(dot.abs()) < 1e-6:
                    is_degenerate = True
                elif float(dot) < 0:
                    frame[:, axis] = -frame[:, axis]
            if torch.det(frame) < 0:
                frame[:, 2] = -frame[:, 2]
            if not is_degenerate:
                frames[global_index] = frame
                degenerate[global_index] = False
    return frames, degenerate


def build_equivariant_features(
    frames: torch.Tensor, num_channels: int = 8
) -> torch.Tensor:
    """Build deterministic C-channel 3D vectors from a local PCA frame."""
    if frames.ndim != 3 or frames.shape[1:] != (3, 3):
        raise ValueError("frames must have shape (N, 3, 3)")
    if num_channels < 3:
        raise ValueError("num_channels must be at least three")
    e0, e1, e2 = frames[:, :, 0], frames[:, :, 1], frames[:, :, 2]
    channels = []
    for channel in range(num_channels):
        angle = 2.0 * math.pi * channel / num_channels
        vector = (
            math.cos(angle) * e0
            + math.sin(angle) * e1
            + 0.5 * math.cos(2.0 * angle) * e2
        )
        channels.append(F.normalize(vector, dim=-1, eps=1e-6))
    return torch.stack(channels, dim=1)


def _equivariant_svd_rotation(
    source_vectors: torch.Tensor, target_vectors: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    covariance = source_vectors.T @ target_vectors
    u, singular_values, vh = torch.linalg.svd(covariance.to(torch.float32))
    correction = torch.eye(3, dtype=torch.float32, device=covariance.device)
    correction[-1, -1] = torch.sign(torch.det(vh.T @ u.T))
    rotation = vh.T @ correction @ u.T
    return rotation.to(source_vectors.dtype), singular_values


@torch.no_grad()
def equivariant_pose_hypotheses(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    source_indices: torch.Tensor,
    target_indices: torch.Tensor,
    source_equivariant: torch.Tensor,
    target_equivariant: torch.Tensor,
    acceptance_radius: float,
    max_hypotheses: int = 32,
) -> tuple[torch.Tensor | None, int, int, str | None]:
    """Create one feature-direction SVD pose hypothesis per correspondence."""
    if len(source_indices) != len(target_indices) or len(source_indices) < 3:
        return None, 0, 0, "too_few_correspondences"
    if source_equivariant.ndim != 3 or target_equivariant.ndim != 3:
        return None, 0, 0, "invalid_equivariant_shape"
    if source_equivariant.shape[1:] != target_equivariant.shape[1:]:
        return None, 0, 0, "equivariant_shape_mismatch"
    if acceptance_radius <= 0:
        return None, 0, 0, "invalid_acceptance_radius"
    count = min(len(source_indices), max_hypotheses)
    hypothesis_indices = torch.linspace(
        0, len(source_indices) - 1, count, device=source_points.device
    ).long().unique()
    source_correspondences = source_points[source_indices]
    target_correspondences = target_points[target_indices]
    best_transform = None
    best_support = 0
    best_residual = torch.tensor(float("inf"), device=source_points.device)
    for hypothesis_index in hypothesis_indices.tolist():
        try:
            rotation, singular_values = _equivariant_svd_rotation(
                source_equivariant[source_indices[hypothesis_index]],
                target_equivariant[target_indices[hypothesis_index]],
            )
        except RuntimeError:
            continue
        if not torch.isfinite(singular_values).all():
            continue
        if singular_values[0] <= 1e-8 or singular_values[-1] <= 1e-8:
            continue
        translation = (
            target_points[target_indices[hypothesis_index]]
            - rotation @ source_points[source_indices[hypothesis_index]]
        )
        transform = torch.eye(
            4, dtype=source_points.dtype, device=source_points.device
        )
        transform[:3, :3] = rotation
        transform[:3, 3] = translation
        residuals = torch.linalg.norm(
            apply_transform(source_correspondences, transform)
            - target_correspondences,
            dim=1,
        )
        inliers = residuals <= acceptance_radius
        support = int(inliers.sum().item())
        mean_residual = residuals[inliers].mean() if support else residuals.mean()
        if support > best_support or (
            support == best_support and mean_residual < best_residual
        ):
            best_transform = transform
            best_support = support
            best_residual = mean_residual
    if best_transform is None:
        return None, 0, len(hypothesis_indices), "equivariant_hypothesis_failed"
    if best_support < 3:
        return None, best_support, len(hypothesis_indices), "insufficient_inliers"
    return best_transform, best_support, len(hypothesis_indices), None


def compute_ops_scores(
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    target_masks: torch.Tensor,
    local_topk: int = 3,
) -> torch.Tensor:
    """SuP-style overlap prior from dual-normalized feature similarities."""
    source = F.normalize(source_features, dim=-1)
    targets = F.normalize(target_features, dim=-1)
    scores: list[torch.Tensor] = []
    for patch, mask in zip(targets, target_masks):
        patch = patch[mask]
        if patch.numel() == 0:
            scores.append(source.new_tensor(0.0))
            continue
        similarity = source @ patch.T
        dual = torch.softmax(similarity, dim=1) * torch.softmax(similarity, dim=0)
        count = min(local_topk, dual.numel())
        scores.append(dual.flatten().topk(count).values.mean())
    return torch.stack(scores)


def sinkhorn_transport(logits: torch.Tensor, iterations: int = 50) -> torch.Tensor:
    if logits.ndim != 2 or logits.numel() == 0:
        raise ValueError("logits must be a non-empty matrix")
    log_transport = logits
    for _ in range(iterations):
        log_transport = log_transport - torch.logsumexp(log_transport, dim=1, keepdim=True)
        log_transport = log_transport - torch.logsumexp(log_transport, dim=0, keepdim=True)
    return log_transport.exp()


@dataclass(frozen=True)
class PoseEstimate:
    transform: torch.Tensor | None
    source_correspondences: torch.Tensor
    target_correspondences: torch.Tensor
    correspondence_scores: torch.Tensor
    residual: torch.Tensor
    backend: str = "lgr"
    equivariant_support: int = 0
    fallback_reason: str | None = None


def estimate_rigid_transform(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    mutual_topk: int = 3,
    sinkhorn_iterations: int = 50,
    acceptance_radius: float = 0.1,
    max_hypotheses: int = 32,
    refinement_steps: int = 5,
    compatibility_graph: bool = False,
    compatibility_tolerance: float = 0.1,
    compatibility_max_nodes: int = 512,
    compatibility_min_clique_size: int = 3,
    use_equivariant: bool = False,
    src_equivariant: torch.Tensor | None = None,
    tgt_equivariant: torch.Tensor | None = None,
) -> PoseEstimate:
    """Estimate a rigid transform, optionally using deterministic PARE-style hypotheses."""
    source = F.normalize(source_features, dim=-1)
    target = F.normalize(target_features, dim=-1)
    logits = (source @ target.T) / 0.1
    transport = sinkhorn_transport(logits, sinkhorn_iterations)
    source_indices, target_indices = _mutual_correspondences(
        transport, mutual_topk
    )
    if source_indices.numel() < 3:
        return PoseEstimate(
            transform=None,
            source_correspondences=source_points[:0],
            target_correspondences=target_points[:0],
            correspondence_scores=source_points.new_zeros(0),
            residual=source_points.new_tensor(float("inf")),
            backend="invalid",
            fallback_reason="too_few_correspondences",
        )
    weights = transport[source_indices, target_indices]
    source_corr = source_points[source_indices]
    target_corr = target_points[target_indices]
    eq_source = None
    eq_target = None
    if use_equivariant and src_equivariant is not None and tgt_equivariant is not None:
        if len(src_equivariant) == len(source_points) and len(tgt_equivariant) == len(target_points):
            eq_source = src_equivariant[source_indices]
            eq_target = tgt_equivariant[target_indices]
        else:
            use_equivariant = False

    if compatibility_graph:
        keep = select_compatibility_clique(
            source_corr,
            target_corr,
            weights,
            tolerance=compatibility_tolerance,
            max_nodes=compatibility_max_nodes,
            min_clique_size=compatibility_min_clique_size,
        )
        if int(keep.sum().item()) >= compatibility_min_clique_size:
            source_corr = source_corr[keep]
            target_corr = target_corr[keep]
            weights = weights[keep]
            if eq_source is not None:
                eq_source = eq_source[keep]
                eq_target = eq_target[keep]

    fallback_reason = None
    if use_equivariant and eq_source is not None and eq_target is not None:
        best_hypothesis, support, _, fallback_reason = equivariant_pose_hypotheses(
            source_corr,
            target_corr,
            torch.arange(len(source_corr), device=source_corr.device),
            torch.arange(len(target_corr), device=target_corr.device),
            eq_source,
            eq_target,
            acceptance_radius,
            max_hypotheses,
        )
        if best_hypothesis is not None and fallback_reason is None:
            residuals = torch.linalg.norm(
                apply_transform(source_corr, best_hypothesis) - target_corr, dim=1
            )
            inliers = residuals <= acceptance_radius
            if int(inliers.sum().item()) >= 3:
                try:
                    refined = _local_global_refinement(
                        source_corr[inliers],
                        target_corr[inliers],
                        weights[inliers],
                        acceptance_radius,
                        max_hypotheses,
                        refinement_steps,
                    )
                    aligned = apply_transform(source_corr, refined)
                    residuals_refined = torch.linalg.norm(
                        aligned - target_corr, dim=1
                    )
                    residual = residuals_refined.mean()
                    try:
                        lgr_compare = _local_global_refinement(
                            source_corr,
                            target_corr,
                            weights,
                            acceptance_radius,
                            max_hypotheses,
                            refinement_steps,
                        )
                        lgr_residuals = torch.linalg.norm(
                            apply_transform(source_corr, lgr_compare)
                            - target_corr,
                            dim=1,
                        )
                        eq_support = int(
                            (residuals_refined <= acceptance_radius).sum().item()
                        )
                        lgr_support = int(
                            (lgr_residuals <= acceptance_radius).sum().item()
                        )
                        if (
                            eq_support < lgr_support
                            or (
                                eq_support == lgr_support
                                and residual > lgr_residuals.mean()
                            )
                        ):
                            fallback_reason = "equivariant_worse_than_lgr"
                        else:
                            return PoseEstimate(
                                refined,
                                source_corr,
                                target_corr,
                                weights,
                                residual,
                                backend="equivariant",
                                equivariant_support=eq_support,
                            )
                    except (RuntimeError, ValueError):
                        return PoseEstimate(
                            refined,
                            source_corr,
                            target_corr,
                            weights,
                            residual,
                            backend="equivariant",
                            equivariant_support=int(
                                (residuals_refined <= acceptance_radius).sum().item()
                            ),
                        )
                except (RuntimeError, ValueError):
                    fallback_reason = "refinement_failure"
            else:
                fallback_reason = "insufficient_inliers"
        elif fallback_reason is None:
            fallback_reason = "equivariant_hypothesis_failed"

    try:
        transform = _local_global_refinement(
            source_corr,
            target_corr,
            weights,
            acceptance_radius,
            max_hypotheses,
            refinement_steps,
        )
        aligned = apply_transform(source_corr, transform)
        residual = torch.linalg.norm(aligned - target_corr, dim=1).mean()
        return PoseEstimate(
            transform,
            source_corr,
            target_corr,
            weights,
            residual,
            backend="lgr" if fallback_reason is None else "lgr",
            equivariant_support=0 if fallback_reason is None else int(
                support if "support" in locals() else 0
            ),
            fallback_reason=fallback_reason,
        )
    except (RuntimeError, ValueError) as error:
        return PoseEstimate(
            transform=None,
            source_correspondences=source_corr,
            target_correspondences=target_corr,
            correspondence_scores=weights,
            residual=source_points.new_tensor(float("inf")),
            backend="invalid",
            equivariant_support=0,
            fallback_reason=f"lgr_failure:{error}",
        )


def select_compatibility_clique(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    weights: torch.Tensor,
    *,
    tolerance: float,
    max_nodes: int = 512,
    min_clique_size: int = 3,
) -> torch.Tensor:
    """Select a high-weight rigidly compatible correspondence clique."""
    if source_points.shape != target_points.shape:
        raise ValueError("source and target correspondence shapes must match")
    if source_points.ndim != 2 or source_points.shape[1] != 3:
        raise ValueError("correspondence points must have shape (N, 3)")
    if weights.ndim != 1 or len(weights) != len(source_points):
        raise ValueError("weights must have shape (N,)")
    if tolerance <= 0 or max_nodes < 1 or min_clique_size < 1:
        raise ValueError("compatibility parameters must be positive")
    count = len(source_points)
    empty = torch.zeros(count, dtype=torch.bool, device=source_points.device)
    if count < min_clique_size:
        return empty

    ranked = weights.argsort(descending=True)[: min(count, max_nodes)]
    source = source_points[ranked]
    target = target_points[ranked]
    pairwise_error = (
        torch.cdist(source, source) - torch.cdist(target, target)
    ).abs()
    compatible = pairwise_error <= tolerance
    compatible.fill_diagonal_(False)

    best: list[int] = []
    best_weight = weights.new_tensor(-torch.inf)
    seed_count = min(len(ranked), 64)
    for seed in range(seed_count):
        clique = [seed]
        candidates = torch.nonzero(compatible[seed], as_tuple=False).flatten()
        while candidates.numel():
            jointly_compatible = compatible[candidates][:, clique].all(dim=1)
            candidates = candidates[jointly_compatible]
            if candidates.numel() == 0:
                break
            next_index = candidates[
                weights[ranked[candidates]].argmax()
            ]
            clique.append(int(next_index.item()))
            candidates = candidates[candidates != next_index]

        clique_weight = weights[ranked[clique]].sum()
        if len(clique) > len(best) or (
            len(clique) == len(best) and clique_weight > best_weight
        ):
            best = clique
            best_weight = clique_weight

    if len(best) < min_clique_size:
        return empty
    keep = torch.zeros(count, dtype=torch.bool, device=source_points.device)
    keep[ranked[torch.as_tensor(best, device=ranked.device)]] = True
    return keep


def weighted_procrustes(
    source_points: torch.Tensor, target_points: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    output_dtype = source_points.dtype
    work_dtype = torch.float32 if output_dtype in (torch.float16, torch.bfloat16) else output_dtype
    with torch.autocast(device_type=source_points.device.type, enabled=False):
        source_work = source_points.to(work_dtype)
        target_work = target_points.to(work_dtype)
        weight_work = weights.to(work_dtype).clamp_min(0)
        weight_work = weight_work / weight_work.sum().clamp_min(torch.finfo(work_dtype).eps)
        source_center = (source_work * weight_work[:, None]).sum(dim=0)
        target_center = (target_work * weight_work[:, None]).sum(dim=0)
        source_centered = source_work - source_center
        target_centered = target_work - target_center
        covariance = source_centered.T @ (weight_work[:, None] * target_centered)
        u, _, vh = torch.linalg.svd(covariance)
        correction = torch.eye(3, dtype=work_dtype, device=source_points.device)
        correction[-1, -1] = torch.sign(torch.det(vh.T @ u.T))
        rotation = vh.T @ correction @ u.T
        translation = target_center - rotation @ source_center
        transform = torch.eye(4, dtype=work_dtype, device=source_points.device)
        transform[:3, :3] = rotation
        transform[:3, 3] = translation
    return transform.to(output_dtype)


def apply_transform(points: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    return points @ transform[:3, :3].T + transform[:3, 3]


def multiscale_lgr_refine(
    source_levels: dict[str, torch.Tensor],
    target_levels: dict[str, torch.Tensor],
    initial_transform: torch.Tensor,
    normalization_scale: float,
    acceptance_radius_angstrom: float = 3.0,
    refinement_steps: int = 5,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Refine a 6 A pose geometrically at 4 A and 2 A in one candidate.

    The point tensors are in the same normalized coordinate system.  Each
    finer level is restricted to the already selected target subcloud.  A
    level with fewer than three mutual radius correspondences safely falls
    back to the transform from the previous level.
    """
    if normalization_scale <= 0 or acceptance_radius_angstrom <= 0:
        raise ValueError("normalization_scale and acceptance radius must be positive")
    if refinement_steps < 1:
        raise ValueError("refinement_steps must be positive")
    current = initial_transform
    radius = acceptance_radius_angstrom / normalization_scale
    counts: dict[str, int] = {}
    for key in ("4.00", "2.00"):
        source = source_levels.get(key)
        target = target_levels.get(key)
        if source is None or target is None or len(source) < 3 or len(target) < 3:
            counts[key] = 0
            continue
        count = 0

        def chunked_nearest(
            query: torch.Tensor,
            reference: torch.Tensor,
            chunk_size: int = 512,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            distances: list[torch.Tensor] = []
            indices: list[torch.Tensor] = []
            for query_chunk in query.split(chunk_size):
                block = torch.cdist(query_chunk, reference)
                chunk_distances, chunk_indices = block.min(dim=1)
                distances.append(chunk_distances)
                indices.append(chunk_indices)
            return torch.cat(distances), torch.cat(indices)

        for _ in range(refinement_steps):
            aligned = apply_transform(source, current)
            source_distances, target_indices = chunked_nearest(aligned, target)
            _, source_indices = chunked_nearest(target, aligned)
            source_ids = torch.arange(len(source), device=source.device)
            mutual = source_indices[target_indices] == source_ids
            keep = mutual & (source_distances <= radius)
            count = int(keep.sum().item())
            if count < 3:
                break
            weights = (1.0 - source_distances[keep] / radius).clamp_min(0.05)
            current = weighted_procrustes(
                source[keep], target[target_indices[keep]], weights
            )
        counts[key] = count
    return current, counts


class MultiScalePostWeightingNetwork(nn.Module):
    def __init__(self, input_dim: int = 8, hidden_dim: int = 64, num_heads: int = 4) -> None:
        super().__init__()
        self.input = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU())
        self.context = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, summaries: torch.Tensor) -> torch.Tensor:
        tokens = self.input(summaries).unsqueeze(0)
        context, _ = self.context(tokens, tokens, tokens, need_weights=False)
        return self.output(tokens + context).squeeze(0).squeeze(-1)


class ProteinRegistrationModel(nn.Module):
    """Per-chain OPS -> pose -> multi-scale post-weighting pipeline."""

    scale_keys = ("2.00", "4.00", "6.00")

    def __init__(
        self,
        shot_dim: int = 352,
        feature_dim: int = 64,
        num_heads: int = 4,
        kernel_points: int = 15,
        ops_topk: int = 6,
        mutual_topk: int = 3,
        max_points_per_patch: int | None = None,
        max_dense_points_per_patch: int | None = None,
        use_compatibility_graph: bool = False,
        compatibility_distance_tolerance_angstrom: float = 12.0,
        compatibility_max_nodes: int = 512,
        compatibility_min_clique_size: int = 3,
        use_multiscale_pose_refinement: bool = True,
        use_fusion_mlp: bool = False,
        fusion_mlp_hidden_dim: int = 256,
        use_equivariant_pose: bool = False,
        use_learned_equivariant_features: bool = False,
        equivariant_feature_dim: int = 8,
        equivariant_max_hypotheses: int = 32,
        equivariant_acceptance_radius_angstrom: float = 3.0,
    ) -> None:
        super().__init__()
        self.ops_topk = ops_topk
        self.mutual_topk = mutual_topk
        self.max_points_per_patch = max_points_per_patch
        self.max_dense_points_per_patch = max_dense_points_per_patch
        self.use_compatibility_graph = use_compatibility_graph
        self.compatibility_distance_tolerance_angstrom = (
            compatibility_distance_tolerance_angstrom
        )
        self.compatibility_max_nodes = compatibility_max_nodes
        self.compatibility_min_clique_size = compatibility_min_clique_size
        self.use_multiscale_pose_refinement = use_multiscale_pose_refinement
        self.use_fusion_mlp = use_fusion_mlp
        self.use_equivariant_pose = use_equivariant_pose
        self.use_learned_equivariant_features = use_learned_equivariant_features
        self.equivariant_feature_dim = equivariant_feature_dim
        self.equivariant_max_hypotheses = equivariant_max_hypotheses
        self.equivariant_acceptance_radius_angstrom = equivariant_acceptance_radius_angstrom
        matching_dim = shot_dim + feature_dim
        self.fusion_mlp = nn.Sequential(
            nn.LayerNorm(matching_dim),
            nn.Linear(matching_dim, fusion_mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(fusion_mlp_hidden_dim, matching_dim),
        )
        nn.init.zeros_(self.fusion_mlp[-1].weight)
        nn.init.zeros_(self.fusion_mlp[-1].bias)
        self.shot_fusion_logit = nn.Parameter(torch.tensor(0.0))
        self.encoder_fusion_logit = nn.Parameter(torch.tensor(math.log(0.25)))
        self.encoder = SubcloudFeatureEncoder(
            input_dim=shot_dim,
            feature_dim=feature_dim,
            num_heads=num_heads,
            radius=2.5,
            kernel_points=kernel_points,
        )
        self.equivariant_head = EquivariantVectorHead(
            feature_dim, equivariant_feature_dim
        )
        self.mpn = MultiScalePostWeightingNetwork(input_dim=8, hidden_dim=feature_dim, num_heads=num_heads)
        self.fcw = FeatureConsistencyWeighting(
            feature_dim=shot_dim + feature_dim,
            hidden_dim=feature_dim,
            num_heads=num_heads,
        )

    def _point_limit_for_scale(self, scale_key: str) -> int | None:
        if scale_key in {"2.00", "4.00"} and self.max_dense_points_per_patch is not None:
            return self.max_dense_points_per_patch
        return self.max_points_per_patch

    def _base_matching_features(
        self, shot_features: torch.Tensor, encoder_features: torch.Tensor
    ) -> torch.Tensor:
        shot = F.normalize(shot_features, dim=-1) * self.shot_fusion_logit.exp()
        encoded = F.normalize(encoder_features, dim=-1) * self.encoder_fusion_logit.exp()
        return torch.cat([shot, encoded], dim=-1)

    def _fuse_matching_features(
        self, shot_features: torch.Tensor, encoder_features: torch.Tensor
    ) -> torch.Tensor:
        matching = self._base_matching_features(shot_features, encoder_features)
        if self.use_fusion_mlp:
            matching = F.normalize(matching + self.fusion_mlp(matching), dim=-1)
        return matching

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        # Older coarse checkpoints predate learnable feature-fusion gates.
        # Fill them with the behavior-equivalent initialization before loading.
        state_dict = dict(state_dict)
        state_dict.setdefault(
            "shot_fusion_logit", self.shot_fusion_logit.detach().clone()
        )
        for name, value in self.fusion_mlp.state_dict().items():
            state_dict.setdefault(f"fusion_mlp.{name}", value.detach().clone())
        state_dict.setdefault(
            "encoder_fusion_logit", self.encoder_fusion_logit.detach().clone()
        )
        for name, value in self.equivariant_head.state_dict().items():
            state_dict.setdefault(
                f"equivariant_head.{name}", value.detach().clone()
            )
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def _equivariant_vectors(
        self, points: torch.Tensor, scalar_features: torch.Tensor
    ) -> torch.Tensor | None:
        if self.use_learned_equivariant_features:
            return self.equivariant_head(points, scalar_features)
        if len(points) < 3:
            return None
        frames, degenerate = compute_local_frames(points)
        if float(degenerate.float().mean()) > 0.5:
            return None
        return build_equivariant_features(frames, self.equivariant_feature_dim)

    def forward(
        self,
        structure: dict,
        chain_id: str,
        source_transform: torch.Tensor | None = None,
        encoded_target: dict[str, dict[str, torch.Tensor]] | None = None,
        differentiable_pose: bool = False,
        use_fcw: bool = False,
    ) -> dict:
        normalization_scale = float(structure["normalization"]["scale"])
        if encoded_target is None:
            encoded_target = self.encode_target(structure, chain_id)

        subclouds = _chain_subclouds(structure, chain_id)
        geometry: dict[str, dict[str, torch.Tensor]] = {}
        for key in self.scale_keys:
            source_points = structure["chains"][chain_id][key]
            if source_transform is not None:
                source_points = apply_transform(source_points, source_transform)
            point_limit = self._point_limit_for_scale(key)
            source_indices = _limit_indices(
                len(source_points), point_limit, source_points.device
            )
            geometry[key] = {
                "source_points": source_points[source_indices],
                "source_indices": source_indices,
                "target_points": _limit_points(subclouds[key]["points"], point_limit),
                "target_masks": _limit_points(subclouds[key]["masks"], point_limit),
            }

        source_shot = structure["chain_shot"][chain_id]
        coarse_geometry = geometry["6.00"]
        source_shot_features = source_shot["features"][coarse_geometry["source_indices"]]
        source_shot_valid = source_shot["valid_mask"][coarse_geometry["source_indices"]]
        if not source_shot_valid.any():
            source_shot_features = torch.zeros_like(source_shot_features)
            source_shot_valid = torch.ones_like(source_shot_valid)
        source_mask = torch.ones_like(source_shot_valid).unsqueeze(0)
        source_features, _ = self.encoder(
            coarse_geometry["source_points"].unsqueeze(0) * (normalization_scale / 6.0),
            source_mask,
            source_shot_features.unsqueeze(0),
            source_shot_valid.unsqueeze(0),
        )
        source_matching_features = self._fuse_matching_features(
            source_shot_features[source_shot_valid],
            source_features.squeeze(0)[source_shot_valid],
        )
        source_pose_features = self._base_matching_features(
            source_shot_features[source_shot_valid],
            source_features.squeeze(0)[source_shot_valid],
        )
        source_equivariant = None
        if self.use_equivariant_pose and len(source_pose_features) >= 3:
            source_points_valid = coarse_geometry["source_points"][source_shot_valid]
            source_scalar_features = source_features.squeeze(0)[source_shot_valid]
            source_equivariant = self._equivariant_vectors(
                source_points_valid, source_scalar_features
            )
        encoded = {
            "6.00": {
                "source_points": coarse_geometry["source_points"][source_shot_valid],
                "source_features": source_pose_features,
                "source_equivariant": source_equivariant,
                "ops_source_features": source_matching_features,
                **encoded_target["6.00"],
            }
        }

        coarse = encoded["6.00"]
        ops_scores = compute_ops_scores(
            coarse["ops_source_features"], coarse["ops_target_features"], coarse["target_masks"]
        )
        candidate_count = min(self.ops_topk, ops_scores.numel())
        candidate_indices = ops_scores.topk(candidate_count).indices
        transforms: list[torch.Tensor] = []
        correspondences: list[dict[str, torch.Tensor]] = []
        refinement_counts: list[dict[str, int]] = []
        summaries: list[torch.Tensor] = []
        pose_backends: list[str] = []
        equivariant_supports: list[int] = []
        fallback_reasons: list[str | None] = []
        for candidate_index in candidate_indices.tolist():
            valid = coarse["target_masks"][candidate_index]
            try:
                target_equivariant = None
                if self.use_equivariant_pose:
                    target_equivariant_all = coarse.get("target_equivariant")
                    if target_equivariant_all is not None:
                        target_equivariant = target_equivariant_all[candidate_index][valid]
                pose = estimate_rigid_transform(
                    coarse["source_points"],
                    coarse["target_points"][candidate_index][valid],
                    coarse["source_features"],
                    coarse["target_features"][candidate_index][valid],
                    mutual_topk=self.mutual_topk,
                    acceptance_radius=(
                        self.equivariant_acceptance_radius_angstrom
                        / normalization_scale
                    ),
                    compatibility_graph=self.use_compatibility_graph,
                    compatibility_tolerance=(
                        self.compatibility_distance_tolerance_angstrom
                        / normalization_scale
                    ),
                    compatibility_max_nodes=self.compatibility_max_nodes,
                    compatibility_min_clique_size=self.compatibility_min_clique_size,
                    use_equivariant=(
                        self.use_equivariant_pose
                        and coarse.get("source_equivariant") is not None
                        and target_equivariant is not None
                    ),
                    src_equivariant=coarse.get("source_equivariant"),
                    tgt_equivariant=target_equivariant,
                    max_hypotheses=self.equivariant_max_hypotheses,
                )
                # Formal inference keeps the historical detached LGR path.
                # Invalid correspondence sets remain excluded by the caller.
                if pose.transform is None:
                    raise ValueError(pose.fallback_reason or "invalid_pose")
                transform = pose.transform if differentiable_pose else pose.transform.detach()
                if self.use_multiscale_pose_refinement:
                    fine_sources = {
                        key: geometry[key]["source_points"]
                        for key in ("4.00", "2.00")
                    }
                    fine_targets = {
                        key: geometry[key]["target_points"][candidate_index][
                            geometry[key]["target_masks"][candidate_index]
                        ]
                        for key in ("4.00", "2.00")
                    }
                    transform, level_counts = multiscale_lgr_refine(
                        fine_sources,
                        fine_targets,
                        transform,
                        normalization_scale,
                    )
                    refinement_counts.append(level_counts)
                else:
                    refinement_counts.append({"4.00": 0, "2.00": 0})
                correspondences.append(
                    {
                        "source": pose.source_correspondences,
                        "target": pose.target_correspondences,
                        "scores": pose.correspondence_scores,
                    }
                )
                pose_backends.append(pose.backend)
                equivariant_supports.append(pose.equivariant_support)
                fallback_reasons.append(pose.fallback_reason)
            except ValueError:
                transform = torch.eye(
                    4,
                    dtype=coarse["source_points"].dtype,
                    device=coarse["source_points"].device,
                )
                correspondences.append(
                    {
                        "source": coarse["source_points"][:0],
                        "target": coarse["source_points"][:0],
                        "scores": coarse["source_points"].new_zeros(0),
                    }
                )
                pose_backends.append("invalid")
                equivariant_supports.append(0)
                fallback_reasons.append("estimate_rigid_transform_error")
            transforms.append(transform)
            summaries.append(
                self._candidate_summary(
                    geometry,
                    coarse,
                    candidate_index,
                    transform,
                    ops_scores[candidate_index],
                    normalization_scale,
                    len(correspondences[-1]["source"]),
                )
            )

        candidate_transforms = torch.stack(transforms)
        summary_tensor = torch.stack(summaries)
        mpn_logits = self.mpn(summary_tensor)
        fcw_weights = None
        if use_fcw:
            fcw_weights = torch.stack(
                [
                    self.fcw(
                        coarse["source_points"],
                        coarse["target_points"][candidate_index][coarse["target_masks"][candidate_index]],
                        coarse["source_features"],
                        coarse["target_features"][candidate_index][coarse["target_masks"][candidate_index]],
                        transform.unsqueeze(0),
                        radius=3.0 / normalization_scale,
                    )[0]
                    for candidate_index, transform in zip(
                        candidate_indices.tolist(), candidate_transforms
                    )
                ]
            )
            final_logits = (
                torch.log(ops_scores[candidate_indices].clamp_min(1e-8))
                + torch.log(fcw_weights.clamp_min(1e-6))
            )
        else:
            final_logits = torch.log(ops_scores[candidate_indices].clamp_min(1e-8)) + mpn_logits
        best_local_index = final_logits.argmax()
        return {
            "ops_scores": ops_scores,
            "encoded_levels": encoded,
            "candidate_indices": candidate_indices,
            "candidate_transforms": candidate_transforms,
            "candidate_correspondences": correspondences,
            "multiscale_refinement_counts": refinement_counts,
            "candidate_summaries": summary_tensor,
            "mpn_logits": mpn_logits,
            "fcw_weights": fcw_weights,
            "final_logits": final_logits,
            "best_subcloud_index": candidate_indices[best_local_index],
            "best_transform": candidate_transforms[best_local_index],
            "pose_scale": "6.00",
            "pose_backends": pose_backends,
            "equivariant_supports": equivariant_supports,
            "fallback_reasons": fallback_reasons,
        }

    def encode_target(
        self, structure: dict, chain_id: str | None = None
    ) -> dict[str, dict[str, torch.Tensor]]:
        normalization_scale = float(structure["normalization"]["scale"])
        subclouds = _chain_subclouds(structure, chain_id)
        coarse = subclouds["6.00"]
        target_points = _limit_points(
            coarse["points"], self.max_points_per_patch
        )
        target_masks = _limit_points(
            coarse["masks"], self.max_points_per_patch
        )
        target_indices = _limit_points(
            coarse["indices"], self.max_points_per_patch
        )
        target_shot = structure["target_shot"]
        padded_features = torch.cat(
            [
                target_shot["features"],
                target_shot["features"].new_zeros(1, target_shot["features"].shape[1]),
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
        shot_features = padded_features[target_indices]
        shot_valid = padded_valid[target_indices]
        if chain_id is not None and not structure["chain_shot"][chain_id][
            "valid_mask"
        ].any():
            shot_features = torch.zeros_like(shot_features)
            shot_valid = target_masks.clone()
        target_features, _ = self.encoder(
            target_points * (normalization_scale / 6.0),
            target_masks,
            shot_features,
            shot_valid,
        )
        target_pose_features = self._base_matching_features(
            shot_features, target_features
        )
        matching_features = self._fuse_matching_features(
            shot_features, target_features
        )
        target_equivariant = None
        if self.use_equivariant_pose:
            target_equivariant_rows = []
            effective_masks = target_masks & shot_valid
            for patch_points, patch_mask, patch_features in zip(
                target_points, effective_masks, target_features
            ):
                full_vectors = patch_points.new_zeros(
                    patch_points.shape[0], self.equivariant_feature_dim, 3
                )
                valid_points = patch_points[patch_mask]
                if len(valid_points) >= 3:
                    vectors = self._equivariant_vectors(
                        valid_points, patch_features[patch_mask]
                    )
                    if vectors is not None:
                        full_vectors[patch_mask] = vectors
                target_equivariant_rows.append(full_vectors)
            target_equivariant = torch.stack(target_equivariant_rows)
        return {
            "6.00": {
                "target_points": target_points,
                "target_masks": target_masks & shot_valid,
                "target_features": target_pose_features,
                "ops_target_features": matching_features,
                "target_equivariant": target_equivariant,
            }
        }

    @staticmethod
    def _chunked_nearest(
        query: torch.Tensor,
        reference: torch.Tensor,
        chunk_size: int = 128,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return nearest-reference distances/indices without full cdist."""
        if query.numel() == 0 or reference.numel() == 0:
            return (
                query.new_empty((0,)),
                torch.empty((0,), dtype=torch.long, device=query.device),
            )
        distances: list[torch.Tensor] = []
        indices: list[torch.Tensor] = []
        for start in range(0, query.shape[0], chunk_size):
            block = torch.cdist(query[start : start + chunk_size], reference)
            block_distances, block_indices = block.min(dim=1)
            distances.append(block_distances)
            indices.append(block_indices)
        return torch.cat(distances), torch.cat(indices)

    @staticmethod
    def _candidate_summary(
        geometry: dict[str, dict[str, torch.Tensor]],
        coarse: dict[str, torch.Tensor],
        candidate_index: int,
        transform: torch.Tensor,
        ops_score: torch.Tensor,
        normalization_scale: float,
        correspondence_count: int,
    ) -> torch.Tensor:
        residuals: list[torch.Tensor] = []
        inlier_ratios: dict[str, torch.Tensor] = {}
        for key in ProteinRegistrationModel.scale_keys:
            level = geometry[key]
            valid = level["target_masks"][candidate_index]
            target_points = level["target_points"][candidate_index][valid]
            aligned = apply_transform(level["source_points"], transform)
            nearest_distance, _ = ProteinRegistrationModel._chunked_nearest(
                aligned, target_points
            )
            residual_angstrom = nearest_distance.mean() * normalization_scale
            residuals.append((residual_angstrom / 3.0).clamp(max=20.0))
            if key in ("2.00", "6.00"):
                inlier_ratios[key] = (
                    nearest_distance * normalization_scale <= 3.0
                ).float().mean()

        valid = coarse["target_masks"][candidate_index]
        target_points = coarse["target_points"][candidate_index][valid]
        target_features = coarse["target_features"][candidate_index][valid]
        aligned = apply_transform(coarse["source_points"], transform)
        _, nearest_index = ProteinRegistrationModel._chunked_nearest(
            aligned, target_points
        )
        source_features = F.normalize(coarse["source_features"], dim=-1)
        nearest_features = F.normalize(target_features[nearest_index], dim=-1)
        consistency = (source_features * nearest_features).sum(dim=-1).mean()
        possible_correspondences = max(
            1, min(len(coarse["source_points"]), len(target_points))
        )
        correspondence_ratio = ops_score.new_tensor(
            min(1.0, correspondence_count / possible_correspondences)
        )
        return torch.stack(
            [
                ops_score,
                *residuals,
                consistency,
                inlier_ratios["2.00"],
                inlier_ratios["6.00"],
                correspondence_ratio,
            ]
        )


def _mutual_correspondences(scores: torch.Tensor, topk: int) -> tuple[torch.Tensor, torch.Tensor]:
    row_k = min(topk, scores.shape[1])
    col_k = min(topk, scores.shape[0])
    row_indices = scores.topk(row_k, dim=1).indices
    col_indices = scores.topk(col_k, dim=0).indices
    row_mask = torch.zeros_like(scores, dtype=torch.bool)
    col_mask = torch.zeros_like(scores, dtype=torch.bool)
    rows = torch.arange(scores.shape[0], device=scores.device)[:, None].expand_as(row_indices)
    cols = torch.arange(scores.shape[1], device=scores.device)[None, :].expand_as(col_indices)
    row_mask[rows, row_indices] = True
    col_mask[col_indices, cols] = True
    return torch.nonzero(row_mask & col_mask, as_tuple=True)


def _local_global_refinement(
    source: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    acceptance_radius: float,
    max_hypotheses: int,
    refinement_steps: int,
) -> torch.Tensor:
    """SuP-style local hypotheses followed by global inlier refinement."""
    if acceptance_radius <= 0:
        raise ValueError("acceptance_radius must be positive")
    ranked = weights.argsort(descending=True)[: min(len(weights), 8)].tolist()
    triplets = list(itertools.islice(itertools.combinations(ranked, 3), max_hypotheses))
    if not triplets:
        return weighted_procrustes(source, target, weights)
    best_transform = weighted_procrustes(source, target, weights)
    best_score = weights.new_tensor(-1.0)
    for triplet in triplets:
        indices = torch.as_tensor(triplet, dtype=torch.long, device=source.device)
        hypothesis = weighted_procrustes(source[indices], target[indices], weights[indices])
        residual = torch.linalg.norm(apply_transform(source, hypothesis) - target, dim=1)
        score = weights[residual <= acceptance_radius].sum()
        if score > best_score:
            best_score = score
            best_transform = hypothesis
    for _ in range(refinement_steps):
        residual = torch.linalg.norm(apply_transform(source, best_transform) - target, dim=1)
        inliers = residual <= acceptance_radius
        if int(inliers.sum().item()) < 3:
            break
        best_transform = weighted_procrustes(source[inliers], target[inliers], weights[inliers])
    return best_transform


def _chain_subclouds(structure: dict, chain_id: str | None) -> dict:
    subclouds = structure["subclouds"]
    if "6.00" in subclouds:
        return subclouds
    if chain_id is None:
        raise ValueError("chain_id is required for chain-specific target subclouds")
    return subclouds[chain_id]


def _limit_indices(
    length: int, limit: int | None, device: torch.device
) -> torch.Tensor:
    if limit is None or length <= limit:
        return torch.arange(length, device=device)
    return torch.linspace(0, length - 1, limit, device=device).long()


def _limit_points(points: torch.Tensor, limit: int | None) -> torch.Tensor:
    if limit is None:
        return points
    if points.ndim < 2:
        return points[:limit]
    return points[:, :limit, ...]


def _make_kernel_points(count: int, radius: float) -> torch.Tensor:
    points = torch.zeros(count, 3, dtype=torch.float32)
    if count == 1:
        return points
    indices = torch.arange(count - 1, dtype=torch.float32)
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    z = 1.0 - 2.0 * (indices + 0.5) / (count - 1)
    radial = torch.sqrt((1.0 - z * z).clamp_min(0.0))
    points[1:, 0] = radial * torch.cos(indices * golden_angle)
    points[1:, 1] = radial * torch.sin(indices * golden_angle)
    points[1:, 2] = z
    points[1:] *= radius * 0.67
    return points
