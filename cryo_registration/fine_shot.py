from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import tempfile

import numpy as np
import torch

from .shot import compute_shot


ShotComputer = Callable[..., tuple[np.ndarray, np.ndarray]]


def _original_angstrom(points: torch.Tensor, structure: dict) -> np.ndarray:
    normalization = structure["normalization"]
    scale = float(normalization["scale"])
    center = normalization["center"]
    if isinstance(center, torch.Tensor):
        center = center.detach().cpu().numpy()
    return (
        points.detach().cpu().numpy().astype(np.float64) * scale
        + np.asarray(center, dtype=np.float64)
    )


def _shot_level(
    features: np.ndarray,
    valid: np.ndarray,
    radius_angstrom: float,
) -> dict[str, torch.Tensor | float]:
    return {
        "features": torch.as_tensor(features, dtype=torch.float32),
        "valid_mask": torch.as_tensor(valid, dtype=torch.bool),
        "radius_angstrom": float(radius_angstrom),
    }


def _validate_level(level: dict, expected: int, label: str) -> dict:
    if len(level["features"]) != expected or len(level["valid_mask"]) != expected:
        raise ValueError(f"{label} count must match 4 A point count")
    return level


def select_fine_shot_levels(
    structure: dict,
    chain_id: str,
) -> tuple[dict, dict]:
    source = structure.get("chain_shot_4a", {}).get(chain_id)
    if source is None:
        legacy = structure["chain_shot"][chain_id]
        source = legacy.get("4.00") if isinstance(legacy, dict) else None
        if source is None and isinstance(legacy, dict) and "features" in legacy:
            if len(legacy["features"]) == len(structure["chains"][chain_id]["4.00"]):
                source = legacy
        if source is None:
            raise ValueError(
                "4 A chain SHOT is missing; 6 A SHOT cannot index 4 A points"
            )

    target = structure.get("target_shot_4a")
    if target is None:
        legacy = structure["target_shot"]
        target = legacy.get("4.00") if isinstance(legacy, dict) else None
        if target is None and isinstance(legacy, dict) and "features" in legacy:
            if len(legacy["features"]) == len(structure["target"]["4.00"]):
                target = legacy
        if target is None:
            raise ValueError(
                "4 A target SHOT is missing; 6 A SHOT cannot index 4 A points"
            )

    return (
        _validate_level(
            source,
            len(structure["chains"][chain_id]["4.00"]),
            "4 A chain SHOT",
        ),
        _validate_level(
            target,
            len(structure["target"]["4.00"]),
            "4 A target SHOT",
        ),
    )


def attach_fine_shot_4a(
    structure: dict,
    structure_id: str,
    cache_root: str | Path,
    radius_angstrom: float = 15.0,
    shot_computer: ShotComputer = compute_shot,
) -> dict:
    if "chain_shot_4a" in structure and "target_shot_4a" in structure:
        for chain_id in structure["chains"]:
            select_fine_shot_levels(structure, chain_id)
        return structure

    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root / f"{structure_id}.pt"
    if cache_path.is_file():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        structure["chain_shot_4a"] = cached["chain_shot_4a"]
        structure["target_shot_4a"] = cached["target_shot_4a"]
        for chain_id in structure["chains"]:
            select_fine_shot_levels(structure, chain_id)
        return structure

    with tempfile.TemporaryDirectory(
        prefix=f"{structure_id}-",
        dir=cache_root,
    ) as temp_root:
        target_features, target_valid = shot_computer(
            _original_angstrom(structure["target"]["2.00"], structure),
            structure["target_normals"].detach().cpu().numpy(),
            _original_angstrom(structure["target"]["4.00"], structure),
            radius_angstrom=radius_angstrom,
            temp_dir=Path(temp_root) / "target",
        )
        chain_levels = {}
        for chain_id in sorted(structure["chains"]):
            features, valid = shot_computer(
                _original_angstrom(
                    structure["chains"][chain_id]["2.00"],
                    structure,
                ),
                structure["chain_normals"][chain_id].detach().cpu().numpy(),
                _original_angstrom(
                    structure["chains"][chain_id]["4.00"],
                    structure,
                ),
                radius_angstrom=radius_angstrom,
                temp_dir=Path(temp_root) / f"chain_{chain_id}",
            )
            chain_levels[chain_id] = _shot_level(
                features,
                valid,
                radius_angstrom,
            )

    target_level = _shot_level(
        target_features,
        target_valid,
        radius_angstrom,
    )
    payload = {
        "chain_shot_4a": chain_levels,
        "target_shot_4a": target_level,
    }
    temporary = cache_path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(cache_path)
    structure.update(payload)
    for chain_id in structure["chains"]:
        select_fine_shot_levels(structure, chain_id)
    return structure
