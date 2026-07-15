from __future__ import annotations

import torch
import torch.nn.functional as F

from cryo_registration.fine_point_matching import (
    EquivariantVectorHead,
    equivariant_channel_conditioning_loss,
    fine_equivariant_alignment_loss,
)


def _rotation_z(angle: float) -> torch.Tensor:
    value = torch.tensor(angle)
    return torch.tensor(
        [
            [torch.cos(value), -torch.sin(value), 0.0],
            [torch.sin(value), torch.cos(value), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def _channel_stats(vectors: torch.Tensor) -> tuple[float, float, float]:
    normalized = F.normalize(vectors.float(), dim=-1)
    channels = normalized.shape[1]
    upper = torch.triu(
        torch.ones(channels, channels, dtype=torch.bool),
        diagonal=1,
    )
    cosine = torch.einsum("nic,njc->nij", normalized, normalized).abs()
    singular = torch.linalg.svdvals(normalized)
    return (
        float(cosine[:, upper].mean().detach()),
        float((singular[:, 1] / singular[:, 0]).mean().detach()),
        float((singular[:, 2] / singular[:, 0]).mean().detach()),
    )


def test_equivariant_head_random_initialization_is_not_rank_one() -> None:
    torch.manual_seed(5)
    parameter = torch.linspace(0.0, 12.0, 128)
    points = torch.stack(
        [parameter, torch.sin(parameter), torch.cos(parameter)],
        dim=1,
    )
    points = points + 0.05 * torch.randn_like(points)
    scalars = torch.randn(len(points), 32)
    head = EquivariantVectorHead(32, channels=8, neighbor_k=16).eval()

    vectors = head(points, scalars)
    mean_abs_cosine, sigma2_ratio, sigma3_ratio = _channel_stats(vectors)

    assert mean_abs_cosine < 0.9
    assert sigma2_ratio > 0.2
    assert sigma3_ratio > 0.05


def test_equivariant_head_preserves_rotation_equivariance() -> None:
    torch.manual_seed(11)
    points = torch.randn(48, 3)
    scalars = torch.randn(48, 16)
    rotation = _rotation_z(0.73)
    head = EquivariantVectorHead(16, channels=8).eval()

    baseline = head(points, scalars)
    rotated = head(points @ rotation.T, scalars)

    assert torch.allclose(rotated, baseline @ rotation.T, atol=2e-4, rtol=2e-4)


def test_conditioning_loss_penalizes_collapsed_channels() -> None:
    frame = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    ).unsqueeze(0)
    collapsed = torch.tensor([1.0, 0.0, 0.0]).expand(1, 6, 3).clone()

    assert equivariant_channel_conditioning_loss(frame) < 1e-6
    assert equivariant_channel_conditioning_loss(collapsed) > 0.5


def test_alignment_loss_rejects_aligned_but_collapsed_solution() -> None:
    frame = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    ).unsqueeze(0)
    collapsed = torch.tensor([1.0, 0.0, 0.0]).expand(1, 6, 3).clone()
    positives = torch.ones(1, 1, dtype=torch.bool)

    conditioned = fine_equivariant_alignment_loss(
        frame,
        frame.clone(),
        positives,
        torch.eye(4),
    )
    degenerate = fine_equivariant_alignment_loss(
        collapsed,
        collapsed.clone(),
        positives,
        torch.eye(4),
    )

    assert conditioned < 1e-6
    assert degenerate > conditioned + 0.05


def test_joint_equivariant_hypothesis_recovers_rotation_from_rank_one_points() -> None:
    from cryo_registration.model import equivariant_pose_hypotheses

    torch.manual_seed(19)
    source_points = torch.randn(12, 3)
    rotation = _rotation_z(0.61)
    translation = torch.tensor([0.4, -0.2, 0.3])
    target_points = source_points @ rotation.T + translation
    directions = F.normalize(torch.randn(12, 3), dim=-1)
    source_vectors = directions[:, None, :].expand(-1, 8, -1).clone()
    target_vectors = source_vectors @ rotation.T
    indices = torch.arange(len(source_points))

    transform, support, _, reason = equivariant_pose_hypotheses(
        source_points,
        target_points,
        indices,
        indices,
        source_vectors,
        target_vectors,
        acceptance_radius=1e-3,
        max_hypotheses=12,
    )

    assert reason is None
    assert transform is not None
    assert support == len(source_points)
    assert torch.allclose(transform[:3, :3], rotation, atol=1e-4)
    assert torch.allclose(transform[:3, 3], translation, atol=1e-4)
