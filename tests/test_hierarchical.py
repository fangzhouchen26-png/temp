import math

import torch

from cryo_registration.fine_point_matching import FinePointMatcher
from cryo_registration.hierarchical import (
    FineCandidate,
    HierarchicalRegistrationRefiner,
    build_fine_target_subclouds,
    fuse_half_candidate_pairs,
    split_chain_by_principal_axis,
)
from cryo_registration.model import apply_transform


def _line_points(count: int, step: float = 1.0) -> torch.Tensor:
    x = torch.arange(count, dtype=torch.float32) * step
    return torch.stack([x, torch.zeros_like(x), torch.zeros_like(x)], dim=1)


def test_principal_axis_split_is_equal_at_6A_and_consistent_across_scales() -> None:
    levels = {
        "2.00": _line_points(12, 0.5),
        "4.00": _line_points(8, 0.75),
        "6.00": _line_points(6, 1.0),
    }

    left, right = split_chain_by_principal_axis(levels, min_points=3)

    assert len(left.points["6.00"]) == len(right.points["6.00"]) == 3
    for key, points in levels.items():
        recovered = torch.cat([left.indices[key], right.indices[key]]).sort().values
        assert torch.equal(recovered, torch.arange(len(points)))
        assert set(left.indices[key].tolist()).isdisjoint(right.indices[key].tolist())
        assert len(left.points[key]) > 0
        assert len(right.points[key]) > 0


def test_fine_target_subclouds_cover_parent_and_preserve_global_indices() -> None:
    parent_points = {
        "2.00": _line_points(17, 0.5),
        "4.00": _line_points(9, 1.0),
        "6.00": _line_points(9, 1.0),
    }
    parent_indices = {
        key: torch.arange(len(points)) + 100 * (level + 1)
        for level, (key, points) in enumerate(parent_points.items())
    }
    half_points = {
        "2.00": _line_points(9, 0.5),
        "4.00": _line_points(5, 1.0),
        "6.00": _line_points(5, 1.0),
    }

    result = build_fine_target_subclouds(
        parent_points,
        parent_indices,
        half_points,
        crop_diameter_factor=1.25,
        point_cap_factor=1.25,
    )

    crop_radius = 1.25 * 4.0 / 2.0
    nearest_center = torch.cdist(
        parent_points["6.00"], result["6.00"]["anchors"]
    ).min(dim=1).values
    assert torch.all(nearest_center <= crop_radius + 1e-6)
    assert len(result["6.00"]["anchors"]) > 1
    for key, level in result.items():
        assert level["points"].shape[:2] == level["masks"].shape
        assert level["indices"].shape == level["masks"].shape
        assert level["points"].shape[1] <= math.ceil(
            1.25 * len(half_points[key])
        )
        valid = level["masks"]
        assert torch.isin(level["indices"][valid], parent_indices[key]).all()
        assert (level["indices"][~valid] == -1).all()


def test_principal_axis_split_rejects_halves_too_small_for_pose() -> None:
    levels = {key: _line_points(5) for key in ("2.00", "4.00", "6.00")}

    try:
        split_chain_by_principal_axis(levels, min_points=3)
    except ValueError as error:
        assert "two halves" in str(error)
    else:
        raise AssertionError("expected a small-chain split failure")


def _fine_candidate(
    transform: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    score: float,
    index: int,
) -> FineCandidate:
    return FineCandidate(
        subcloud_index=index,
        transform=transform,
        source_correspondences=source,
        target_correspondences=target,
        correspondence_scores=torch.ones(len(source)),
        final_score=torch.tensor(score),
    )


def test_fuse_half_candidate_pairs_recovers_one_rigid_transform() -> None:
    angle = torch.tensor(math.pi / 6)
    rotation = torch.tensor(
        [
            [torch.cos(angle), -torch.sin(angle), 0.0],
            [torch.sin(angle), torch.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    expected = torch.eye(4)
    expected[:3, :3] = rotation
    expected[:3, 3] = torch.tensor([0.4, -0.2, 0.3])
    left_source = torch.tensor(
        [[-2.0, 0.0, 0.0], [-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]]
    )
    right_source = torch.tensor(
        [[1.0, 0.0, 0.0], [2.0, 1.0, 0.0], [2.0, 0.0, 1.0]]
    )
    left_true = _fine_candidate(
        expected, left_source, apply_transform(left_source, expected), 2.0, 4
    )
    right_true = _fine_candidate(
        expected, right_source, apply_transform(right_source, expected), 2.5, 7
    )
    wrong = torch.eye(4)
    left_wrong = _fine_candidate(
        wrong, left_source, left_source + 4.0, -2.0, 1
    )
    right_wrong = _fine_candidate(
        wrong, right_source, right_source - 4.0, -2.0, 2
    )

    fused = fuse_half_candidate_pairs(
        [left_wrong, left_true],
        [right_wrong, right_true],
        normalization_scale=10.0,
    )

    assert fused.left_subcloud_index == 4
    assert fused.right_subcloud_index == 7
    assert torch.allclose(fused.transform, expected, atol=1e-4)
    assert fused.inlier_ratio_3A == 1.0
    assert fused.residual_A < 1e-3


def test_fuse_half_candidate_pairs_rejects_insufficient_correspondences() -> None:
    point = torch.zeros(1, 3)
    candidate = _fine_candidate(torch.eye(4), point, point, 1.0, 0)

    try:
        fuse_half_candidate_pairs([candidate], [candidate], normalization_scale=1.0)
    except ValueError as error:
        assert "fuse" in str(error)
    else:
        raise AssertionError("expected correspondence fusion failure")


def test_hierarchical_refiner_returns_three_refined_whole_chain_poses() -> None:
    from cryo_registration.model import ProteinRegistrationModel

    base = torch.tensor(
        [
            [0.00, 0.00, 0.00],
            [0.20, 0.00, 0.00],
            [0.00, 0.25, 0.00],
            [0.00, 0.00, 0.30],
            [0.40, 0.10, 0.05],
            [0.45, 0.30, 0.10],
            [0.35, 0.15, 0.35],
            [0.50, 0.40, 0.40],
        ]
    )
    base = torch.cat([base, base + torch.tensor([0.03, 0.02, 0.01])])
    patches = torch.stack([base, base + 1.0, base - 1.0])
    masks = torch.ones(3, len(base), dtype=torch.bool)
    indices = torch.arange(3 * len(base)).reshape(3, len(base))
    shot = torch.eye(352)[: len(base)]
    structure = {
        "target": {
            key: patches.flatten(0, 1)
            for key in ("2.00", "4.00", "6.00")
        },
        "chains": {
            "A": {
                key: base.clone()
                for key in ("2.00", "4.00", "6.00")
            }
        },
        "subclouds": {
            "A": {
                key: {
                    "points": patches.clone(),
                    "masks": masks.clone(),
                    "indices": indices.clone(),
                }
                for key in ("2.00", "4.00", "6.00")
            }
        },
        "target_shot": {
            "features": shot.repeat(3, 1),
            "valid_mask": torch.ones(3 * len(base), dtype=torch.bool),
        },
        "chain_shot": {
            "A": {
                "features": shot,
                "valid_mask": torch.ones(len(base), dtype=torch.bool),
            }
        },
        "normalization": {"scale": 10.0},
    }
    coarse = ProteinRegistrationModel(
        feature_dim=16,
        num_heads=4,
        kernel_points=5,
        ops_topk=3,
        max_points_per_patch=16,
    ).eval()
    refiner = HierarchicalRegistrationRefiner(
        coarse,
        coarse_output_topk=3,
        fine_ops_topk=6,
        fine_output_topk=3,
        fine_min_half_points=8,
        pair_scoring_mode="topology",
        fine_point_matcher=FinePointMatcher(
            shot_dim=352,
            encoder_dim=16,
            feature_dim=16,
            num_heads=4,
            max_pair_elements=1024,
        ),
    ).eval()

    with torch.no_grad():
        output = refiner(structure, "A")

    assert output["candidate_transforms"].shape == (3, 4, 4)
    assert output["candidate_scores"].shape == (3,)
    assert output["coarse_subcloud_indices"].shape == (3,)
    assert len(output["refinement_status"]) == 3
    assert all(status == "refined" for status in output["refinement_status"]), output["refinement_status"]
    assert all(len(item["halves"]) == 2 for item in output["fine_diagnostics"])
    assert all(
        set(half["target_subclouds"]) == {"2.00", "4.00", "6.00"}
        for item in output["fine_diagnostics"]
        for half in item["halves"]
    )
    assert all(
        len(half["point_matcher_statuses"]) == len(half["candidate_indices"])
        for item in output["fine_diagnostics"]
        for half in item["halves"]
    )
    assert all(
        len(half["point_matcher_inputs"]) == len(half["candidate_indices"])
        for item in output["fine_diagnostics"]
        for half in item["halves"]
    )
    assert torch.isfinite(output["candidate_transforms"]).all()
    assert all(
        item["fusion"].components is not None
        for item in output["fine_diagnostics"]
    )
    assert torch.isfinite(output["best_transform"]).all()

    # Every coarse candidate falls back when an equal split cannot satisfy
    # the minimum half size. Candidate-level outputs must remain aligned.
    refiner.fine_min_half_points = 9
    with torch.no_grad():
        fallback_output = refiner(structure, "A")

    assert fallback_output["candidate_transforms"].shape == (3, 4, 4)
    assert fallback_output["candidate_coarse_scores"].shape == (3,)
    assert fallback_output["candidate_fusion_scores"].shape == (3,)
    assert fallback_output["candidate_local_scores"].shape == (3,)
    assert fallback_output["candidate_refined_mask"].shape == (3,)
    assert not fallback_output["candidate_refined_mask"].any()
    assert torch.count_nonzero(fallback_output["candidate_fusion_scores"]) == 0
    assert torch.count_nonzero(fallback_output["candidate_local_scores"]) == 0
    assert all(
        status.startswith("fallback:")
        for status in fallback_output["refinement_status"]
    )
