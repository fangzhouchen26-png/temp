from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml


@dataclass(frozen=True)
class DataConfig:
    input_root: str = "/fangzhouc/point-data-homologous"
    processed_root: str = "processed"
    split_seed: int = 0


@dataclass(frozen=True)
class ModelConfig:
    shot_dim: int = 352
    feature_dim: int = 64
    num_heads: int = 4
    kernel_points: int = 15
    ops_topk: int = 6
    mutual_topk: int = 3
    max_points_per_patch: int | None = None
    max_dense_points_per_patch: int | None = None
    use_compatibility_graph: bool = False
    compatibility_distance_tolerance_angstrom: float = 12.0
    compatibility_max_nodes: int = 512
    compatibility_min_clique_size: int = 3
    use_multiscale_pose_refinement: bool = True
    use_fusion_mlp: bool = False
    fusion_mlp_hidden_dim: int = 256
    coarse_output_topk: int = 3
    fine_ops_topk: int = 6
    fine_output_topk: int = 6
    fine_crop_diameter_factor: float = 1.25
    fine_point_cap_factor: float = 1.25
    fine_max_points_per_patch: int | None = 3000
    fine_min_valid_points: int = 3
    fine_min_half_points: int = 8
    pair_scoring_mode: str = "legacy"
    use_fine_point_matcher: bool = False
    fine_point_feature_dim: int = 64
    fine_point_attention_heads: int = 4
    fine_point_attention_query_chunk: int = 512
    fine_point_pair_radius_angstrom: float = 3.0
    max_fine_point_pair_elements: int = 4_000_000
    fine_point_descriptor_temperature: float = 0.1
    use_equivariant_hypothesis_pose: bool = False
    use_learned_equivariant_features: bool = False
    equivariant_feature_dim: int = 8
    equivariant_max_hypotheses: int = 32
    equivariant_acceptance_radius_angstrom: float = 3.0


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 100
    correspondence_start_epoch: int = 1
    mpn_start_epoch: int = 2
    sup_awl: bool = False
    equivariant_loss_weight: float = 0.0
    equivariant_loss_start_epoch: int = 1
    checkpoint_interval_structures: int = 250
    max_coarse_workload_elements: int | None = 4_000_000
    log_interval_structures: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    early_stopping_patience: int = 15
    grad_clip_norm: float = 1.0
    mixed_precision: bool = True
    seed: int = 0
    device: str = "auto"


@dataclass(frozen=True)
class OutputConfig:
    run_dir: str = "outputs/registration"


@dataclass(frozen=True)
class RegistrationConfig:
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    output: OutputConfig = OutputConfig()


T = TypeVar("T")


def load_config(path: str | Path | None = None) -> RegistrationConfig:
    raw: dict[str, Any] = {}
    if path is not None:
        loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        raw = loaded or {}
    allowed = {field.name for field in fields(RegistrationConfig)}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown configuration sections: {sorted(unknown)}")
    return RegistrationConfig(
        data=_build(DataConfig, raw.get("data", {})),
        model=_build(ModelConfig, raw.get("model", {})),
        training=_build(TrainingConfig, raw.get("training", {})),
        output=_build(OutputConfig, raw.get("output", {})),
    )


def _build(cls: type[T], values: dict[str, Any]) -> T:
    allowed = {field.name for field in fields(cls)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"unknown {cls.__name__} options: {sorted(unknown)}")
    return cls(**values)
