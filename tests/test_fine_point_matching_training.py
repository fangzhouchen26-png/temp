from __future__ import annotations

import torch

from cryo_registration.fine_point_matching import (
    FinePointMatcher,
    make_gt_point_pairs,
)
from cryo_registration.train_fine_point_matching import (
    fine_point_training_loss,
    point_match_metrics,
)


def _matching_case() -> tuple[FinePointMatcher, tuple[torch.Tensor, ...]]:
    torch.manual_seed(17)
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.0, 0.2, 0.0],
            [0.0, 0.0, 0.2],
        ]
    )
    shot = torch.randn(4, 12)
    encoded = torch.randn(4, 8)
    matcher = FinePointMatcher(
        shot_dim=12,
        encoder_dim=8,
        feature_dim=8,
        num_heads=2,
        equivariant_channels=4,
        max_pair_elements=100,
    )
    inputs = (points, points.clone(), shot, shot.clone(), encoded, encoded.clone())
    return matcher, inputs


def test_fine_point_training_loss_backpropagates_all_trainable_heads() -> None:
    matcher, inputs = _matching_case()
    output = matcher(*inputs)
    labels = make_gt_point_pairs(
        inputs[0],
        inputs[1],
        torch.eye(4),
        normalization_scale=10.0,
        radius_angstrom=0.5,
    )

    losses = fine_point_training_loss(
        output,
        labels,
        torch.eye(4),
        descriptor_temperature=0.1,
        matchability_weight=0.25,
        equivariant_weight=0.1,
    )
    losses["total"].backward()

    assert set(losses) == {
        "total",
        "descriptor",
        "matchability",
        "equivariant",
    }
    assert torch.isfinite(losses["total"])
    assert matcher.shot_projection[0].weight.grad is not None
    assert matcher.encoder_projection[0].weight.grad is not None
    assert matcher.source_matchability.weight.grad is not None
    assert matcher.equivariant_head.pair_mlp[0].weight.grad is not None


def test_point_match_metrics_report_top1_and_top3_physical_recall() -> None:
    source = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    target = torch.tensor(
        [
            [1.0, 0.0],
            [0.8, 0.2],
            [0.0, 1.0],
            [0.2, 0.8],
        ]
    )
    positives = torch.tensor(
        [
            [True, True, False, False],
            [False, False, True, True],
        ]
    )

    metrics = point_match_metrics(source, target, positives)

    assert metrics["valid_source_points"] == 2
    assert metrics["point_recall_at_1_3A"] == 1.0
    assert metrics["point_recall_at_3_3A"] == 1.0

