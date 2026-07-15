from __future__ import annotations

import copy

import pytest
import torch

from cryo_registration.config import ModelConfig, RegistrationConfig
from cryo_registration.fine_point_matching import (
    FinePointMatcher,
    FinePointPairTooLarge,
    bidirectional_matchability_loss,
    freeze_for_fine_point_training,
    make_gt_point_pairs,
    symmetric_multi_positive_descriptor_loss,
)
from cryo_registration.hierarchical import HierarchicalRegistrationRefiner
from cryo_registration.model import ProteinRegistrationModel
from cryo_registration.train_fine_mpn import build_hierarchical_refiner


def _tiny_coarse_model() -> ProteinRegistrationModel:
    return ProteinRegistrationModel(
        feature_dim=16,
        num_heads=4,
        kernel_points=5,
        ops_topk=3,
        max_points_per_patch=16,
    )


def test_phase_a_freezes_everything_except_fine_point_matcher() -> None:
    coarse = _tiny_coarse_model()
    matcher = FinePointMatcher(
        shot_dim=352,
        encoder_dim=16,
        feature_dim=16,
        num_heads=4,
    )
    refiner = HierarchicalRegistrationRefiner(
        coarse,
        coarse_output_topk=3,
        fine_ops_topk=3,
        fine_output_topk=3,
        fine_encoder=copy.deepcopy(coarse.encoder),
        fine_point_matcher=matcher,
    )

    freeze_for_fine_point_training(refiner)

    trainable = [
        name for name, parameter in refiner.named_parameters()
        if parameter.requires_grad
    ]
    assert trainable
    assert all(name.startswith("fine_point_matcher.") for name in trainable)
    assert not any(
        parameter.requires_grad for parameter in refiner.coarse_model.parameters()
    )
    assert not any(
        parameter.requires_grad for parameter in refiner.fine_encoder.parameters()
    )
    assert not any(
        parameter.requires_grad for parameter in refiner.fine_mpn.parameters()
    )


def _identity() -> torch.Tensor:
    return torch.eye(4)


def test_gt_point_pairs_use_physical_angstrom_scale() -> None:
    source_angstrom = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    target_angstrom = torch.tensor([[2.0, 0.0, 0.0], [4.1, 0.0, 0.0]])

    labels_scale10 = make_gt_point_pairs(
        source_angstrom / 10.0,
        target_angstrom / 10.0,
        _identity(),
        normalization_scale=10.0,
        radius_angstrom=3.0,
    )
    labels_scale20 = make_gt_point_pairs(
        source_angstrom / 20.0,
        target_angstrom / 20.0,
        _identity(),
        normalization_scale=20.0,
        radius_angstrom=3.0,
    )

    expected = torch.tensor([[True, False], [True, False]])
    assert labels_scale10.valid
    assert labels_scale10.positive_count == 2
    assert torch.equal(labels_scale10.positive_mask, expected)
    assert torch.equal(labels_scale20.positive_mask, expected)
    assert torch.equal(labels_scale10.source_matchable, torch.tensor([True, True]))
    assert torch.equal(labels_scale10.target_matchable, torch.tensor([True, False]))


def test_gt_point_pairs_do_not_fabricate_a_positive() -> None:
    labels = make_gt_point_pairs(
        torch.tensor([[0.0, 0.0, 0.0]]),
        torch.tensor([[1.0, 0.0, 0.0]]),
        _identity(),
        normalization_scale=10.0,
        radius_angstrom=3.0,
    )

    assert not labels.valid
    assert labels.positive_count == 0
    assert not labels.positive_mask.any()


def test_multi_positive_descriptor_loss_prefers_true_pairs_and_backpropagates() -> None:
    source = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0]],
        requires_grad=True,
    )
    target = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0]],
        requires_grad=True,
    )
    positives = torch.eye(2, dtype=torch.bool)

    good = symmetric_multi_positive_descriptor_loss(
        source,
        target,
        positives,
        temperature=0.1,
    )
    bad = symmetric_multi_positive_descriptor_loss(
        source,
        target.flip(0),
        positives,
        temperature=0.1,
    )
    assert good < bad

    good.backward()
    assert source.grad is not None
    assert target.grad is not None
    assert source.grad.norm() > 0
    assert target.grad.norm() > 0


def _point_matcher_inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(7)
    source_points = torch.randn(5, 3)
    target_points = torch.randn(4, 3)
    source_shot = torch.randn(5, 12)
    target_shot = torch.randn(4, 12)
    source_encoded = torch.randn(5, 8)
    target_encoded = torch.randn(4, 8)
    return (
        source_points,
        target_points,
        source_shot,
        target_shot,
        source_encoded,
        target_encoded,
    )


def test_fine_point_matcher_outputs_balanced_normalized_features() -> None:
    matcher = FinePointMatcher(
        shot_dim=12,
        encoder_dim=8,
        feature_dim=8,
        num_heads=2,
        equivariant_channels=4,
        query_chunk_size=2,
        max_pair_elements=100,
    ).eval()

    output = matcher(*_point_matcher_inputs())

    assert output.source_descriptors.shape == (5, 8)
    assert output.target_descriptors.shape == (4, 8)
    assert output.source_matchability_logits.shape == (5,)
    assert output.target_matchability_logits.shape == (4,)
    assert output.source_equivariant.shape == (5, 4, 3)
    assert output.target_equivariant.shape == (4, 4, 3)
    assert torch.allclose(
        output.source_descriptors.norm(dim=-1),
        torch.ones(5),
        atol=1e-5,
    )
    assert torch.allclose(
        matcher.fusion_weights,
        torch.tensor([0.5, 0.5]),
        atol=1e-6,
    )


def test_fine_point_matcher_rejects_oversized_dense_pair() -> None:
    matcher = FinePointMatcher(
        shot_dim=12,
        encoder_dim=8,
        feature_dim=8,
        num_heads=2,
        max_pair_elements=19,
    )
    with pytest.raises(FinePointPairTooLarge) as caught:
        matcher(*_point_matcher_inputs())
    assert caught.value.pair_elements == 20
    assert caught.value.limit == 19


def test_chunked_attention_matches_single_chunk() -> None:
    full = FinePointMatcher(
        shot_dim=12,
        encoder_dim=8,
        feature_dim=8,
        num_heads=2,
        equivariant_channels=4,
        query_chunk_size=32,
        max_pair_elements=100,
    ).eval()
    chunked = FinePointMatcher(
        shot_dim=12,
        encoder_dim=8,
        feature_dim=8,
        num_heads=2,
        equivariant_channels=4,
        query_chunk_size=2,
        max_pair_elements=100,
    ).eval()
    chunked.load_state_dict(full.state_dict())
    inputs = _point_matcher_inputs()

    full_output = full(*inputs)
    chunked_output = chunked(*inputs)

    assert torch.allclose(
        full_output.source_descriptors,
        chunked_output.source_descriptors,
        atol=1e-6,
    )
    assert torch.allclose(
        full_output.target_descriptors,
        chunked_output.target_descriptors,
        atol=1e-6,
    )


def test_fine_equivariant_vectors_follow_rotation() -> None:
    matcher = FinePointMatcher(
        shot_dim=12,
        encoder_dim=8,
        feature_dim=8,
        num_heads=2,
        equivariant_channels=4,
        max_pair_elements=100,
    ).eval()
    inputs = _point_matcher_inputs()
    angle = torch.tensor(0.7)
    rotation = torch.tensor(
        [
            [torch.cos(angle), -torch.sin(angle), 0.0],
            [torch.sin(angle), torch.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    baseline = matcher(*inputs)
    rotated_inputs = (
        inputs[0] @ rotation.T,
        inputs[1] @ rotation.T,
        *inputs[2:],
    )
    rotated = matcher(*rotated_inputs)

    expected = baseline.source_equivariant @ rotation.T
    assert torch.allclose(rotated.source_equivariant, expected, atol=1e-5)


def test_bidirectional_matchability_loss_prefers_correct_logits() -> None:
    source_labels = torch.tensor([True, False])
    target_labels = torch.tensor([False, True])
    good = bidirectional_matchability_loss(
        torch.tensor([5.0, -5.0]),
        torch.tensor([-5.0, 5.0]),
        source_labels,
        target_labels,
    )
    bad = bidirectional_matchability_loss(
        torch.tensor([-5.0, 5.0]),
        torch.tensor([5.0, -5.0]),
        source_labels,
        target_labels,
    )
    assert good < bad

def test_builder_attaches_point_matcher_only_when_enabled() -> None:
    disabled = RegistrationConfig(
        model=ModelConfig(
            feature_dim=16,
            num_heads=4,
            kernel_points=5,
            use_fine_point_matcher=False,
        )
    )
    enabled = RegistrationConfig(
        model=ModelConfig(
            feature_dim=16,
            num_heads=4,
            kernel_points=5,
            use_fine_point_matcher=True,
            fine_point_feature_dim=16,
            fine_point_attention_heads=4,
        )
    )

    assert build_hierarchical_refiner(
        disabled,
        _tiny_coarse_model(),
    ).fine_point_matcher is None
    refiner = build_hierarchical_refiner(
        enabled,
        _tiny_coarse_model(),
    )
    assert isinstance(refiner.fine_point_matcher, FinePointMatcher)
