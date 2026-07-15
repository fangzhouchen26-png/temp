from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class TargetSubclouds:
    anchor_indices: torch.Tensor
    anchors: torch.Tensor
    indices: torch.Tensor
    masks: torch.Tensor
    points: torch.Tensor


def furthest_point_sample(
    points: torch.Tensor, count: int, start_index: int | None = None
) -> torch.Tensor:
    _validate_points(points)
    point_count = points.shape[0]
    if count <= 0 or count > point_count:
        raise ValueError("count must be between 1 and the number of points")
    if start_index is not None and not 0 <= start_index < point_count:
        raise ValueError("start_index is outside the point cloud")

    selected = torch.empty(count, dtype=torch.long, device=points.device)
    min_distance = torch.full((point_count,), torch.inf, device=points.device)
    current = (
        int(start_index)
        if start_index is not None
        else int(torch.randint(point_count, (1,), device=points.device).item())
    )
    for index in range(count):
        selected[index] = current
        squared_distance = torch.sum((points - points[current]) ** 2, dim=1)
        min_distance = torch.minimum(min_distance, squared_distance)
        current = int(torch.argmax(min_distance).item())
    return selected


def furthest_point_sample_by_coverage(
    points: torch.Tensor,
    coverage_radius: float,
    start_index: int | None = None,
) -> torch.Tensor:
    """Select FPS centers until every point lies within coverage_radius."""
    _validate_points(points)
    if coverage_radius <= 0 or not math.isfinite(float(coverage_radius)):
        raise ValueError("coverage_radius must be finite and positive")
    if start_index is not None and not 0 <= start_index < len(points):
        raise ValueError("start_index is outside the point cloud")
    if start_index is None:
        center = points.mean(dim=0)
        current = int(torch.sum((points - center) ** 2, dim=1).argmax().item())
    else:
        current = int(start_index)
    selected: list[int] = []
    min_distance = torch.full((len(points),), torch.inf, device=points.device)
    radius_squared = float(coverage_radius) ** 2
    while True:
        selected.append(current)
        distance = torch.sum((points - points[current]) ** 2, dim=1)
        min_distance = torch.minimum(min_distance, distance)
        if float(min_distance.max().item()) <= radius_squared:
            break
        current = int(min_distance.argmax().item())
    return torch.as_tensor(selected, dtype=torch.long, device=points.device)


def build_target_subclouds(
    points: torch.Tensor,
    chain_count: int | None = None,
    longest_chain_diameter: float | None = None,
    longest_chain_points: int | None = None,
    start_index: int | None = None,
    anchor_points: torch.Tensor | None = None,
    center_count: int | None = None,
    crop_diameter: float | None = None,
    point_cap: int | None = None,
) -> TargetSubclouds:
    """Create FPS-centered target patches with an explicit crop diameter."""
    _validate_points(points)
    legacy_center_count = center_count is None
    if center_count is None:
        if chain_count is None or chain_count <= 0:
            raise ValueError("chain_count must be positive")
        center_count = 3 * chain_count
    if crop_diameter is None:
        if longest_chain_diameter is None:
            raise ValueError("crop_diameter is required")
        crop_diameter = longest_chain_diameter
    if point_cap is None:
        if longest_chain_points is None:
            raise ValueError("point_cap is required")
        point_cap = math.ceil(1.25 * longest_chain_points)
    if center_count <= 0 or points.shape[0] < center_count:
        if legacy_center_count:
            raise ValueError("target must contain at least 3 * chain_count points")
        raise ValueError("center_count must be between 1 and the target point count")
    if crop_diameter <= 0 or point_cap <= 0:
        raise ValueError("crop_diameter and point_cap must be positive")
    cap = min(point_cap, points.shape[0])
    if anchor_points is None:
        anchor_indices = furthest_point_sample(points, center_count, start_index)
    else:
        _validate_points(anchor_points)
        if len(anchor_points) != center_count:
            raise ValueError("anchor_points must match center_count")
        anchor_indices = torch.cdist(
            anchor_points.to(device=points.device, dtype=points.dtype), points
        ).argmin(dim=1)
    anchors = points[anchor_indices]
    distances = torch.cdist(anchors, points)
    nearest_distance, nearest_indices = distances.topk(cap, dim=1, largest=False)
    masks = nearest_distance <= crop_diameter / 2.0
    sentinel = points.shape[0]
    indices = nearest_indices.masked_fill(~masks, sentinel)
    padded = torch.cat([points, torch.zeros_like(points[:1])], dim=0)
    subcloud_points = padded[indices]
    return TargetSubclouds(anchor_indices, anchors, indices, masks, subcloud_points)


def _validate_points(points: torch.Tensor) -> None:
    if not isinstance(points, torch.Tensor) or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be a torch.Tensor with shape (N, 3)")
    if points.shape[0] == 0:
        raise ValueError("points cannot be empty")
