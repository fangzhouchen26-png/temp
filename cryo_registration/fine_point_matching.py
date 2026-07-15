from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .model import EquivariantVectorHead, apply_transform


@dataclass(frozen=True)
class PointPairLabels:
    positive_mask: torch.Tensor
    source_matchable: torch.Tensor
    target_matchable: torch.Tensor
    positive_count: int
    valid: bool


@dataclass(frozen=True)
class FinePointMatchOutput:
    source_descriptors: torch.Tensor
    target_descriptors: torch.Tensor
    source_matchability_logits: torch.Tensor
    target_matchability_logits: torch.Tensor
    source_equivariant: torch.Tensor
    target_equivariant: torch.Tensor
    pair_elements: int


class FinePointPairTooLarge(RuntimeError):
    def __init__(self, pair_elements: int, limit: int) -> None:
        super().__init__(
            f"fine point pair has {pair_elements} elements, exceeding {limit}"
        )
        self.pair_elements = int(pair_elements)
        self.limit = int(limit)


def make_gt_point_pairs(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    ground_truth_transform: torch.Tensor,
    normalization_scale: float,
    radius_angstrom: float = 3.0,
) -> PointPairLabels:
    """Create all physical-space source/target positives within a radius."""
    if source_points.ndim != 2 or source_points.shape[-1] != 3:
        raise ValueError("source_points must have shape (N, 3)")
    if target_points.ndim != 2 or target_points.shape[-1] != 3:
        raise ValueError("target_points must have shape (M, 3)")
    if normalization_scale <= 0 or radius_angstrom <= 0:
        raise ValueError("normalization_scale and radius_angstrom must be positive")

    if len(source_points) == 0 or len(target_points) == 0:
        positive = torch.zeros(
            len(source_points),
            len(target_points),
            dtype=torch.bool,
            device=source_points.device,
        )
    else:
        aligned_source = apply_transform(source_points, ground_truth_transform)
        distance_angstrom = (
            torch.cdist(aligned_source.float(), target_points.float())
            * float(normalization_scale)
        )
        positive = distance_angstrom <= float(radius_angstrom)
    count = int(positive.sum().item())
    return PointPairLabels(
        positive_mask=positive,
        source_matchable=positive.any(dim=1),
        target_matchable=positive.any(dim=0),
        positive_count=count,
        valid=count > 0,
    )


def _multi_positive_directional_loss(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
) -> torch.Tensor | None:
    valid = positive_mask.any(dim=1)
    if not bool(valid.any()):
        return None
    positive_logits = logits.masked_fill(~positive_mask, -torch.inf)
    numerator = torch.logsumexp(positive_logits[valid], dim=1)
    denominator = torch.logsumexp(logits[valid], dim=1)
    return (denominator - numerator).mean()


def symmetric_multi_positive_descriptor_loss(
    source_descriptors: torch.Tensor,
    target_descriptors: torch.Tensor,
    positive_mask: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Symmetric contrastive loss that accepts multiple geometric positives."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if source_descriptors.ndim != 2 or target_descriptors.ndim != 2:
        raise ValueError("descriptors must have shape (N, C) and (M, C)")
    if source_descriptors.shape[1] != target_descriptors.shape[1]:
        raise ValueError("descriptor dimensions must match")
    if positive_mask.shape != (
        len(source_descriptors),
        len(target_descriptors),
    ):
        raise ValueError("positive_mask shape must match the descriptor counts")
    if not bool(positive_mask.any()):
        raise ValueError("descriptor loss requires at least one positive pair")

    logits = (
        F.normalize(source_descriptors, dim=-1)
        @ F.normalize(target_descriptors, dim=-1).T
    ) / float(temperature)
    terms = [
        _multi_positive_directional_loss(logits, positive_mask),
        _multi_positive_directional_loss(logits.T, positive_mask.T),
    ]
    valid_terms = [term for term in terms if term is not None]
    return torch.stack(valid_terms).mean()


def bidirectional_matchability_loss(
    source_logits: torch.Tensor,
    target_logits: torch.Tensor,
    source_labels: torch.Tensor,
    target_labels: torch.Tensor,
) -> torch.Tensor:
    if source_logits.shape != source_labels.shape:
        raise ValueError("source matchability shapes must match")
    if target_logits.shape != target_labels.shape:
        raise ValueError("target matchability shapes must match")
    source_loss = F.binary_cross_entropy_with_logits(
        source_logits,
        source_labels.to(source_logits.dtype),
    )
    target_loss = F.binary_cross_entropy_with_logits(
        target_logits,
        target_labels.to(target_logits.dtype),
    )
    return 0.5 * (source_loss + target_loss)


def fine_equivariant_alignment_loss(
    source_vectors: torch.Tensor,
    target_vectors: torch.Tensor,
    positive_mask: torch.Tensor,
    ground_truth_transform: torch.Tensor,
) -> torch.Tensor:
    """Align Fine vector features with the known ground-truth rotation."""
    if source_vectors.ndim != 3 or source_vectors.shape[-1] != 3:
        raise ValueError("source_vectors must have shape (N, C, 3)")
    if target_vectors.ndim != 3 or target_vectors.shape[-1] != 3:
        raise ValueError("target_vectors must have shape (M, C, 3)")
    if source_vectors.shape[1:] != target_vectors.shape[1:]:
        raise ValueError("source and target vector channel shapes must match")
    if positive_mask.shape != (len(source_vectors), len(target_vectors)):
        raise ValueError("positive_mask shape must match vector point counts")
    source_indices, target_indices = torch.nonzero(
        positive_mask,
        as_tuple=True,
    )
    if source_indices.numel() == 0:
        raise ValueError("equivariant loss requires at least one positive pair")
    rotation = ground_truth_transform[:3, :3]
    rotated = source_vectors @ rotation.T
    source_positive = F.normalize(rotated[source_indices], dim=-1)
    target_positive = F.normalize(target_vectors[target_indices], dim=-1)
    cosine = (source_positive * target_positive).sum(dim=-1)
    return (1.0 - cosine).mean()


class _ChunkedBidirectionalAttention(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_heads: int,
        query_chunk_size: int,
        ff_multiplier: int = 2,
        gate_init: float = 0.05,
    ) -> None:
        super().__init__()
        self.query_chunk_size = int(query_chunk_size)
        self.source_to_target = nn.MultiheadAttention(
            feature_dim,
            num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.target_to_source = nn.MultiheadAttention(
            feature_dim,
            num_heads,
            dropout=0.0,
            batch_first=True,
        )
        hidden_dim = feature_dim * ff_multiplier
        self.source_norm1 = nn.LayerNorm(feature_dim)
        self.source_norm2 = nn.LayerNorm(feature_dim)
        self.target_norm1 = nn.LayerNorm(feature_dim)
        self.target_norm2 = nn.LayerNorm(feature_dim)
        self.source_ff = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
        )
        self.target_ff = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
        )
        self.source_gate = nn.Parameter(torch.tensor(float(gate_init)))
        self.target_gate = nn.Parameter(torch.tensor(float(gate_init)))

    def _attend(
        self,
        attention: nn.MultiheadAttention,
        queries: torch.Tensor,
        keys: torch.Tensor,
    ) -> torch.Tensor:
        chunks = []
        for start in range(0, len(queries), self.query_chunk_size):
            query = queries[start : start + self.query_chunk_size].unsqueeze(0)
            delta, _ = attention(
                query,
                keys.unsqueeze(0),
                keys.unsqueeze(0),
                need_weights=False,
            )
            chunks.append(delta.squeeze(0))
        return torch.cat(chunks, dim=0)

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source_delta = self._attend(self.source_to_target, source, target)
        source = source + self.source_gate * self.source_norm1(source_delta)
        source = source + self.source_gate * self.source_ff(
            self.source_norm2(source)
        )

        target_delta = self._attend(self.target_to_source, target, source)
        target = target + self.target_gate * self.target_norm1(target_delta)
        target = target + self.target_gate * self.target_ff(
            self.target_norm2(target)
        )
        return source, target


class FinePointMatcher(nn.Module):
    """Candidate-local trainable module for downstream 4A point matching."""

    def __init__(
        self,
        shot_dim: int = 352,
        encoder_dim: int = 64,
        feature_dim: int = 64,
        num_heads: int = 4,
        equivariant_channels: int = 8,
        query_chunk_size: int = 512,
        max_pair_elements: int = 4_000_000,
    ) -> None:
        super().__init__()
        values = (
            shot_dim,
            encoder_dim,
            feature_dim,
            num_heads,
            equivariant_channels,
            query_chunk_size,
            max_pair_elements,
        )
        if min(values) <= 0:
            raise ValueError("FinePointMatcher dimensions and limits must be positive")
        if feature_dim % num_heads:
            raise ValueError("feature_dim must be divisible by num_heads")
        self.max_pair_elements = int(max_pair_elements)
        self.shot_projection = nn.Sequential(
            nn.Linear(shot_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
        )
        self.encoder_projection = nn.Sequential(
            nn.Linear(encoder_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
        )
        self.fusion_logits = nn.Parameter(torch.zeros(2))
        self.cross_attention = _ChunkedBidirectionalAttention(
            feature_dim,
            num_heads,
            query_chunk_size,
        )
        self.descriptor_norm = nn.LayerNorm(feature_dim)
        self.source_matchability = nn.Linear(feature_dim, 1)
        self.target_matchability = nn.Linear(feature_dim, 1)
        self.equivariant_head = EquivariantVectorHead(
            scalar_dim=feature_dim,
            channels=equivariant_channels,
        )

    @property
    def fusion_weights(self) -> torch.Tensor:
        return torch.softmax(self.fusion_logits, dim=0)

    def _fuse(
        self,
        shot_features: torch.Tensor,
        encoded_features: torch.Tensor,
    ) -> torch.Tensor:
        shot = F.normalize(self.shot_projection(shot_features), dim=-1)
        encoded = F.normalize(self.encoder_projection(encoded_features), dim=-1)
        weights = self.fusion_weights
        return F.normalize(weights[0] * shot + weights[1] * encoded, dim=-1)

    def forward(
        self,
        source_points: torch.Tensor,
        target_points: torch.Tensor,
        source_shot: torch.Tensor,
        target_shot: torch.Tensor,
        source_encoded: torch.Tensor,
        target_encoded: torch.Tensor,
    ) -> FinePointMatchOutput:
        if source_points.ndim != 2 or source_points.shape[-1] != 3:
            raise ValueError("source_points must have shape (N, 3)")
        if target_points.ndim != 2 or target_points.shape[-1] != 3:
            raise ValueError("target_points must have shape (M, 3)")
        if len(source_points) == 0 or len(target_points) == 0:
            raise ValueError("FinePointMatcher requires non-empty point clouds")
        if source_shot.shape[0] != len(source_points):
            raise ValueError("source SHOT count must match source points")
        if target_shot.shape[0] != len(target_points):
            raise ValueError("target SHOT count must match target points")
        if source_encoded.shape[0] != len(source_points):
            raise ValueError("source encoder count must match source points")
        if target_encoded.shape[0] != len(target_points):
            raise ValueError("target encoder count must match target points")

        pair_elements = len(source_points) * len(target_points)
        if pair_elements > self.max_pair_elements:
            raise FinePointPairTooLarge(pair_elements, self.max_pair_elements)

        source = self._fuse(source_shot, source_encoded)
        target = self._fuse(target_shot, target_encoded)
        source, target = self.cross_attention(source, target)
        source_descriptors = F.normalize(
            self.descriptor_norm(source),
            dim=-1,
        )
        target_descriptors = F.normalize(
            self.descriptor_norm(target),
            dim=-1,
        )
        return FinePointMatchOutput(
            source_descriptors=source_descriptors,
            target_descriptors=target_descriptors,
            source_matchability_logits=self.source_matchability(
                source_descriptors
            ).squeeze(-1),
            target_matchability_logits=self.target_matchability(
                target_descriptors
            ).squeeze(-1),
            source_equivariant=self.equivariant_head(
                source_points,
                source_descriptors,
            ),
            target_equivariant=self.equivariant_head(
                target_points,
                target_descriptors,
            ),
            pair_elements=pair_elements,
        )


def freeze_for_fine_point_training(refiner: nn.Module) -> None:
    """Freeze the complete hierarchy except its candidate-local point matcher."""
    matcher = getattr(refiner, "fine_point_matcher", None)
    if matcher is None:
        raise ValueError("refiner must have a fine_point_matcher")
    for parameter in refiner.parameters():
        parameter.requires_grad_(False)
    for parameter in matcher.parameters():
        parameter.requires_grad_(True)
    refiner.eval()
    matcher.train()

