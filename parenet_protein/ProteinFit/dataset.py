"""Protein density point-cloud pairs for PARE-Net.

Each sample registers a simulated single-chain density point cloud (source)
against either an oracle target crop or a sliding-window crop extracted from the
complete protein/complex density point cloud (reference). Coordinates remain in
Angstroms.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from pareconv.utils.data import (
    build_dataloader_stack_mode,
    registration_collate_fn_stack_mode,
)


@dataclass(frozen=True)
class ProteinPair:
    case_id: str
    chain_id: str
    ref_path: Path
    src_path: Path
    ref_count: int
    src_count: int


@dataclass(frozen=True)
class ProteinWindow:
    pair_index: int
    candidate_id: int
    center: Tuple[float, float, float]
    radius: float


def _nonempty_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def count_density_points(path: Path) -> int:
    """Return the number of point records without parsing all floating values."""
    try:
        line_count = len(_nonempty_lines(path))
    except (OSError, UnicodeError):
        return 0
    if line_count < 7:
        return 0
    payload = line_count - 5
    return payload // 2 if payload % 2 == 0 else 0


def load_density_txt(path: os.PathLike) -> Dict[str, np.ndarray]:
    """Parse the custom ``*_2.00.txt`` density point-cloud format.

    File layout:
      1. voxel spacing
      2. grid dimensions
      3. map center
      4. map origin
      5. statistics
      6+. alternating index and attribute rows

    Point coordinates are reconstructed as ``origin + spacing * (ix, iy, iz)``.
    """
    path = Path(path)
    lines = _nonempty_lines(path)
    if len(lines) < 7:
        raise ValueError(f"{path}: empty or incomplete density point file")

    try:
        spacing = float(lines[0])
        grid_shape = np.asarray([int(v) for v in lines[1].split()], dtype=np.int32)
        center = np.asarray([float(v) for v in lines[2].split()], dtype=np.float64)
        origin = np.asarray([float(v) for v in lines[3].split()], dtype=np.float64)
        statistics = np.asarray([float(v) for v in lines[4].split()], dtype=np.float64)
    except ValueError as exc:
        raise ValueError(f"{path}: malformed five-line header") from exc

    if grid_shape.shape != (3,) or center.shape != (3,) or origin.shape != (3,):
        raise ValueError(f"{path}: expected three-dimensional grid metadata")
    if spacing <= 0:
        raise ValueError(f"{path}: voxel spacing must be positive")

    payload = lines[5:]
    if len(payload) % 2:
        raise ValueError(f"{path}: point records must occupy two lines each")

    count = len(payload) // 2
    voxel_indices = np.empty((count, 3), dtype=np.int32)
    flat_indices = np.empty((count,), dtype=np.int64)
    normals = np.empty((count, 3), dtype=np.float32)
    density = np.empty((count,), dtype=np.float32)

    for point_index in range(count):
        index_fields = payload[2 * point_index].split()
        attr_fields = payload[2 * point_index + 1].split()
        if len(index_fields) != 4 or len(attr_fields) != 4:
            raise ValueError(f"{path}: malformed point record {point_index}")
        try:
            flat_indices[point_index] = int(index_fields[0])
            voxel_indices[point_index] = [int(v) for v in index_fields[1:4]]
            normals[point_index] = [float(v) for v in attr_fields[0:3]]
            density[point_index] = float(attr_fields[3])
        except ValueError as exc:
            raise ValueError(f"{path}: non-numeric point record {point_index}") from exc

    points = origin[None, :] + spacing * voxel_indices.astype(np.float64)
    return {
        "points": points.astype(np.float32),
        "normals": normals,
        "density": density,
        "voxel_indices": voxel_indices,
        "flat_indices": flat_indices,
        "voxel_size": np.asarray(spacing, dtype=np.float32),
        "grid_shape": grid_shape,
        "center": center.astype(np.float32),
        "origin": origin.astype(np.float32),
        "statistics": statistics.astype(np.float32),
    }


def _chain_id_from_path(path: Path) -> str:
    match = re.search(r"_chain([^_]+)_src_", path.name)
    return match.group(1) if match else path.stem


def discover_pairs(
    dataset_root: os.PathLike,
    point_suffix: str = "_2.00.txt",
    min_source_points: int = 128,
    min_target_points: int = 256,
) -> Tuple[List[ProteinPair], List[str]]:
    """Discover valid chain-to-complex pairs and return skipped-file messages."""
    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    pairs: List[ProteinPair] = []
    skipped: List[str] = []
    for case_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        case_id = case_dir.name
        ref_candidates = sorted(case_dir.glob(f"*_tgt{point_suffix}"))
        if not ref_candidates:
            skipped.append(f"{case_id}: no target file matching *_tgt{point_suffix}")
            continue
        ref_path = ref_candidates[0]
        ref_count = count_density_points(ref_path)
        if ref_count < min_target_points:
            skipped.append(
                f"{case_id}: target has {ref_count} points (<{min_target_points})"
            )
            continue

        src_candidates = sorted(case_dir.glob(f"*_src{point_suffix}"))
        if not src_candidates:
            skipped.append(f"{case_id}: no source file matching *_src{point_suffix}")
            continue

        for src_path in src_candidates:
            src_count = count_density_points(src_path)
            chain_id = _chain_id_from_path(src_path)
            if src_count < min_source_points:
                skipped.append(
                    f"{case_id}/chain{chain_id}: source has {src_count} points "
                    f"(<{min_source_points})"
                )
                continue
            pairs.append(
                ProteinPair(
                    case_id=case_id,
                    chain_id=chain_id,
                    ref_path=ref_path,
                    src_path=src_path,
                    ref_count=ref_count,
                    src_count=src_count,
                )
            )
    return pairs, skipped


def build_case_splits(
    case_ids: Sequence[str],
    seed: int = 7351,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Dict[str, List[str]]:
    """Create deterministic, case-level splits with no target-map leakage."""
    unique_cases = np.asarray(sorted(set(case_ids)), dtype=object)
    if unique_cases.size == 0:
        return {"train": [], "val": [], "test": []}

    rng = np.random.default_rng(seed)
    unique_cases = unique_cases[rng.permutation(unique_cases.size)]
    n_cases = unique_cases.size

    if n_cases < 3:
        return {
            "train": unique_cases.tolist(),
            "val": unique_cases[-1:].tolist(),
            "test": unique_cases[-1:].tolist(),
        }

    n_train = max(1, int(round(n_cases * train_ratio)))
    n_val = max(1, int(round(n_cases * val_ratio)))
    if n_train + n_val >= n_cases:
        n_train = max(1, n_cases - 2)
        n_val = 1
    n_test = n_cases - n_train - n_val
    if n_test < 1:
        n_test = 1
        n_train = max(1, n_train - 1)

    return {
        "train": unique_cases[:n_train].tolist(),
        "val": unique_cases[n_train : n_train + n_val].tolist(),
        "test": unique_cases[n_train + n_val :].tolist(),
    }


def load_or_build_splits(
    pairs: Sequence[ProteinPair],
    split_file: Optional[os.PathLike],
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> Dict[str, List[str]]:
    if split_file:
        path = Path(split_file).expanduser()
        if path.is_file():
            with path.open("r", encoding="utf-8") as handle:
                splits = json.load(handle)
            required = {"train", "val", "test"}
            if set(splits) != required:
                raise ValueError(f"{path}: expected exactly {sorted(required)}")
            return {key: [str(value) for value in values] for key, values in splits.items()}
    return build_case_splits(
        [pair.case_id for pair in pairs],
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )


def _sample_uniform_rotation(rng: np.random.Generator) -> np.ndarray:
    # Shoemake's method: a uniform unit quaternion, represented as x, y, z, w.
    u1, u2, u3 = rng.random(3)
    quat = np.asarray(
        [
            math.sqrt(1.0 - u1) * math.sin(2.0 * math.pi * u2),
            math.sqrt(1.0 - u1) * math.cos(2.0 * math.pi * u2),
            math.sqrt(u1) * math.sin(2.0 * math.pi * u3),
            math.sqrt(u1) * math.cos(2.0 * math.pi * u3),
        ],
        dtype=np.float64,
    )
    return Rotation.from_quat(quat).as_matrix().astype(np.float32)


def _make_source_augmentation(
    points: np.ndarray,
    rng: np.random.Generator,
    translation_magnitude: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Transform source points and return (augmented_points, inverse_transform)."""
    rotation = _sample_uniform_rotation(rng)
    delta = rng.uniform(-translation_magnitude, translation_magnitude, size=3).astype(
        np.float32
    )
    centroid = points.mean(axis=0).astype(np.float32)

    # Rotate around the source centroid, then translate by delta.
    translation = centroid + delta - centroid @ rotation.T
    augmented = points @ rotation.T + translation

    forward = np.eye(4, dtype=np.float32)
    forward[:3, :3] = rotation
    forward[:3, 3] = translation
    inverse = np.linalg.inv(forward).astype(np.float32)
    return augmented.astype(np.float32), inverse


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def compute_chain_radius(
    points: np.ndarray,
    quantile: float = 0.99,
) -> Tuple[np.ndarray, float]:
    """Return the chain centroid and a robust bounding-sphere radius in Angstroms."""
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError(f"Expected a non-empty Nx3 point cloud, got {points.shape}")
    if not (0.0 < quantile <= 1.0):
        raise ValueError("radius quantile must be in (0, 1]")

    center = points.mean(axis=0).astype(np.float32)
    distances = np.linalg.norm(points - center[None, :], axis=1)
    radius = float(np.quantile(distances, quantile))
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("Computed an invalid chain radius")
    return center, radius


def crop_spherical_window(
    points: np.ndarray,
    center: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Extract all points inside a closed spherical window."""
    if radius <= 0.0:
        raise ValueError("crop radius must be positive")
    delta = points - np.asarray(center, dtype=np.float32)[None, :]
    mask = np.einsum("ij,ij->i", delta, delta) <= radius * radius
    return points[mask]


def chain_window_coverage(
    chain_points: np.ndarray,
    center: np.ndarray,
    radius: float,
) -> float:
    """Fraction of original chain points covered by a spherical window."""
    if chain_points.size == 0:
        return 0.0
    delta = chain_points - np.asarray(center, dtype=np.float32)[None, :]
    inside = np.einsum("ij,ij->i", delta, delta) <= radius * radius
    return float(np.mean(inside))


def _voxelized_window_centers(points: np.ndarray, stride: float) -> np.ndarray:
    """Generate target-supported sliding centers using a regular 3-D grid."""
    if points.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float32)
    if stride <= 0.0:
        raise ValueError("window stride must be positive")

    origin = points.min(axis=0)
    voxel_indices = np.floor((points - origin[None, :]) / stride).astype(np.int64)
    _, inverse = np.unique(voxel_indices, axis=0, return_inverse=True)
    num_voxels = int(inverse.max()) + 1

    centers = np.zeros((num_voxels, 3), dtype=np.float64)
    counts = np.zeros((num_voxels,), dtype=np.int64)
    np.add.at(centers, inverse, points)
    np.add.at(counts, inverse, 1)
    centers /= counts[:, None]
    return centers.astype(np.float32)


def _farthest_point_subset(points: np.ndarray, max_points: Optional[int]) -> np.ndarray:
    """Deterministically retain spatially distributed candidate centers."""
    if max_points is None or points.shape[0] <= max_points:
        return points
    if max_points < 1:
        raise ValueError("max_candidates must be positive or None")

    centroid = points.mean(axis=0)
    first = int(np.argmax(np.linalg.norm(points - centroid[None, :], axis=1)))
    selected = np.empty((max_points,), dtype=np.int64)
    selected[0] = first
    min_sq_distances = np.full((points.shape[0],), np.inf, dtype=np.float64)

    for i in range(1, max_points):
        delta = points - points[selected[i - 1]][None, :]
        sq_distances = np.einsum("ij,ij->i", delta, delta)
        min_sq_distances = np.minimum(min_sq_distances, sq_distances)
        selected[i] = int(np.argmax(min_sq_distances))

    return points[selected]


def generate_sliding_window_centers(
    target_points: np.ndarray,
    stride: float,
    crop_radius: float,
    min_points: int,
    max_candidates: Optional[int],
) -> np.ndarray:
    """Generate valid target-supported centers for spherical sliding windows."""
    centers = _voxelized_window_centers(target_points, stride)
    if centers.shape[0] == 0:
        return centers

    tree = cKDTree(target_points)
    counts = tree.query_ball_point(centers, crop_radius, return_length=True, workers=1)
    centers = centers[np.asarray(counts) >= min_points]
    return _farthest_point_subset(centers, max_candidates)


def _random_keep(
    points: np.ndarray,
    keep_ratio: float,
    rng: np.random.Generator,
    min_points: int,
) -> np.ndarray:
    if keep_ratio >= 1.0 or points.shape[0] <= min_points:
        return points
    keep = max(min_points, int(round(points.shape[0] * keep_ratio)))
    keep = min(keep, points.shape[0])
    indices = rng.choice(points.shape[0], size=keep, replace=False)
    return points[indices]


def _limit_points(
    points: np.ndarray,
    point_limit: Optional[int],
    rng: np.random.Generator,
) -> np.ndarray:
    if point_limit is None or points.shape[0] <= point_limit:
        return points
    indices = rng.choice(points.shape[0], size=point_limit, replace=False)
    return points[indices]


def source_overlap(
    ref_points: np.ndarray,
    src_points: np.ndarray,
    transform: np.ndarray,
    radius: float,
) -> float:
    if ref_points.size == 0 or src_points.size == 0:
        return 0.0
    aligned = apply_transform(src_points, transform)
    distances, _ = cKDTree(ref_points).query(aligned, k=1, workers=1)
    return float(np.mean(distances <= radius))


class ProteinPairDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_root: os.PathLike,
        subset: str,
        point_suffix: str = "_2.00.txt",
        split_file: Optional[os.PathLike] = None,
        split_seed: int = 7351,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        point_limit: Optional[int] = 30000,
        min_source_points: int = 128,
        min_target_points: int = 256,
        use_augmentation: bool = False,
        deterministic_augmentation: bool = False,
        augmentation_seed: int = 7351,
        augmentation_noise: float = 0.25,
        augmentation_translation: float = 30.0,
        source_keep_ratio: float = 0.90,
        matching_radius: float = 4.0,
        crop_mode: str = "none",
        crop_diameter_scale: float = 1.25,
        crop_radius_quantile: float = 0.99,
        crop_stride_ratio: float = 0.25,
        crop_min_stride: float = 2.0,
        crop_min_points: int = 128,
        crop_max_candidates: Optional[int] = None,
        crop_center_jitter_ratio: float = 0.10,
        crop_min_chain_coverage: float = 0.85,
        crop_min_oracle_overlap: float = 0.50,
    ):
        super().__init__()
        if subset not in {"train", "val", "test"}:
            raise ValueError(f"Unknown subset: {subset}")
        if not (0.0 < source_keep_ratio <= 1.0):
            raise ValueError("source_keep_ratio must be in (0, 1]")
        if crop_mode not in {"none", "oracle", "sliding"}:
            raise ValueError("crop_mode must be one of: none, oracle, sliding")
        if crop_diameter_scale <= 0.0:
            raise ValueError("crop_diameter_scale must be positive")
        if crop_stride_ratio <= 0.0 or crop_min_stride <= 0.0:
            raise ValueError("crop stride parameters must be positive")
        if crop_min_points < 1:
            raise ValueError("crop_min_points must be positive")
        if not (0.0 <= crop_center_jitter_ratio <= 1.0):
            raise ValueError("crop_center_jitter_ratio must be in [0, 1]")
        if not (0.0 < crop_min_chain_coverage <= 1.0):
            raise ValueError("crop_min_chain_coverage must be in (0, 1]")
        if not (0.0 < crop_min_oracle_overlap <= 1.0):
            raise ValueError("crop_min_oracle_overlap must be in (0, 1]")

        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.subset = subset
        self.point_limit = point_limit
        self.min_source_points = min_source_points
        self.use_augmentation = use_augmentation
        self.deterministic_augmentation = deterministic_augmentation
        self.augmentation_seed = augmentation_seed
        self.augmentation_noise = augmentation_noise
        self.augmentation_translation = augmentation_translation
        self.source_keep_ratio = source_keep_ratio
        self.matching_radius = matching_radius
        self.crop_mode = crop_mode
        self.crop_diameter_scale = crop_diameter_scale
        self.crop_radius_quantile = crop_radius_quantile
        self.crop_stride_ratio = crop_stride_ratio
        self.crop_min_stride = crop_min_stride
        self.crop_min_points = crop_min_points
        self.crop_max_candidates = crop_max_candidates
        self.crop_center_jitter_ratio = crop_center_jitter_ratio
        self.crop_min_chain_coverage = crop_min_chain_coverage
        self.crop_min_oracle_overlap = crop_min_oracle_overlap

        all_pairs, self.skipped = discover_pairs(
            self.dataset_root,
            point_suffix=point_suffix,
            min_source_points=min_source_points,
            min_target_points=min_target_points,
        )
        splits = load_or_build_splits(
            all_pairs,
            split_file=split_file,
            seed=split_seed,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )
        selected_cases = set(splits[subset])
        self.case_splits = splits
        self.pairs = [pair for pair in all_pairs if pair.case_id in selected_cases]
        if not self.pairs:
            raise RuntimeError(
                f"No valid {subset} pairs found under {self.dataset_root}. "
                "Run inspect_dataset.py and lower min-point thresholds only if justified."
            )

        if self.crop_mode == "oracle":
            self._filter_oracle_pairs()

        self.windows: List[ProteinWindow] = []
        if self.crop_mode == "sliding":
            self._build_sliding_windows()

    def _pair_crop_geometry(
        self,
        src_points: np.ndarray,
    ) -> Tuple[np.ndarray, float, float, float]:
        chain_center, chain_radius = compute_chain_radius(
            src_points,
            quantile=self.crop_radius_quantile,
        )
        chain_diameter = 2.0 * chain_radius
        crop_diameter = self.crop_diameter_scale * chain_diameter
        crop_radius = 0.5 * crop_diameter
        return chain_center, chain_radius, chain_diameter, crop_radius

    def _filter_oracle_pairs(self) -> None:
        """Keep only oracle crops with sufficient physical source-to-target overlap."""
        valid_pairs: List[ProteinPair] = []
        identity = np.eye(4, dtype=np.float32)
        for pair in self.pairs:
            target_points = load_density_txt(pair.ref_path)["points"]
            source_points = load_density_txt(pair.src_path)["points"]
            chain_center, _, _, crop_radius = self._pair_crop_geometry(source_points)
            crop_points = crop_spherical_window(
                target_points,
                chain_center,
                crop_radius,
            )
            overlap = source_overlap(
                crop_points,
                source_points,
                identity,
                radius=self.matching_radius,
            )
            if (
                crop_points.shape[0] >= self.crop_min_points
                and overlap >= self.crop_min_oracle_overlap
            ):
                valid_pairs.append(pair)
            else:
                self.skipped.append(
                    f"{pair.case_id}/chain{pair.chain_id}: oracle crop rejected; "
                    f"points={crop_points.shape[0]}, overlap={overlap:.4f}"
                )

        self.pairs = valid_pairs
        if not self.pairs:
            raise RuntimeError(
                f"No valid {self.subset} oracle pairs remain after overlap filtering. "
                "Check coordinate frames, density thresholding, or crop geometry."
            )

    def _build_sliding_windows(self) -> None:
        for pair_index, pair in enumerate(self.pairs):
            target_points = load_density_txt(pair.ref_path)["points"]
            source_points = load_density_txt(pair.src_path)["points"]
            _, _, chain_diameter, crop_radius = self._pair_crop_geometry(source_points)
            stride = max(self.crop_min_stride, self.crop_stride_ratio * chain_diameter)
            centers = generate_sliding_window_centers(
                target_points,
                stride=stride,
                crop_radius=crop_radius,
                min_points=self.crop_min_points,
                max_candidates=self.crop_max_candidates,
            )
            if centers.shape[0] == 0:
                raise RuntimeError(
                    f"{pair.case_id}/chain{pair.chain_id}: no valid spherical windows; "
                    "reduce crop_min_points or crop_stride_ratio"
                )
            for candidate_id, center in enumerate(centers):
                self.windows.append(
                    ProteinWindow(
                        pair_index=pair_index,
                        candidate_id=candidate_id,
                        center=tuple(float(v) for v in center),
                        radius=float(crop_radius),
                    )
                )

    def __len__(self) -> int:
        return len(self.windows) if self.crop_mode == "sliding" else len(self.pairs)

    def _rng(self, index: int) -> np.random.Generator:
        if self.deterministic_augmentation:
            return np.random.default_rng(self.augmentation_seed + index)
        return np.random.default_rng()

    def _resolve_sample(
        self,
        index: int,
    ) -> Tuple[ProteinPair, int, int, Optional[np.ndarray], Optional[float]]:
        if self.crop_mode != "sliding":
            return self.pairs[index], index, 0, None, None
        window = self.windows[index]
        pair = self.pairs[window.pair_index]
        return (
            pair,
            window.pair_index,
            window.candidate_id,
            np.asarray(window.center, dtype=np.float32),
            window.radius,
        )

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        pair, pair_index, candidate_id, window_center, stored_crop_radius = self._resolve_sample(index)
        # All sliding windows for one chain must receive the same deterministic
        # source sampling, augmentation and source noise.  Candidate-specific target
        # sampling/noise uses a separate RNG so it cannot shift the source RNG state.
        source_rng = self._rng(pair_index)
        target_rng = self._rng(index + 10000000)

        ref_all = load_density_txt(pair.ref_path)["points"]
        src_original = load_density_txt(pair.src_path)["points"]
        chain_center, chain_radius, chain_diameter, crop_radius = self._pair_crop_geometry(
            src_original
        )

        if self.crop_mode == "oracle":
            window_center = chain_center.copy()
            if self.use_augmentation and self.crop_center_jitter_ratio > 0.0:
                max_jitter = self.crop_center_jitter_ratio * chain_radius
                candidate_center = window_center + target_rng.uniform(
                    -max_jitter,
                    max_jitter,
                    size=3,
                ).astype(np.float32)
                coverage = chain_window_coverage(
                    src_original,
                    candidate_center,
                    crop_radius,
                )
                candidate_crop = crop_spherical_window(
                    ref_all,
                    candidate_center,
                    crop_radius,
                )
                candidate_overlap = source_overlap(
                    candidate_crop,
                    src_original,
                    np.eye(4, dtype=np.float32),
                    radius=self.matching_radius,
                )
                if (
                    coverage >= self.crop_min_chain_coverage
                    and candidate_crop.shape[0] >= self.crop_min_points
                    and candidate_overlap >= self.crop_min_oracle_overlap
                ):
                    window_center = candidate_center
        elif self.crop_mode == "sliding":
            crop_radius = float(stored_crop_radius)
        else:
            window_center = ref_all.mean(axis=0).astype(np.float32)
            crop_radius = float("inf")

        ref_points = (
            crop_spherical_window(ref_all, window_center, crop_radius)
            if self.crop_mode != "none"
            else ref_all
        )
        if ref_points.shape[0] < self.crop_min_points:
            raise RuntimeError(
                f"{pair.case_id}/chain{pair.chain_id}/candidate{candidate_id}: "
                f"crop contains {ref_points.shape[0]} points (<{self.crop_min_points})"
            )

        src_points = src_original
        # Point limiting happens after spherical cropping, so the physical window is
        # preserved even when a dense crop exceeds the model point budget.
        ref_points = _limit_points(ref_points, self.point_limit, target_rng)
        src_points = _limit_points(src_points, self.point_limit, source_rng)

        transform = np.eye(4, dtype=np.float32)
        if self.use_augmentation:
            src_points = _random_keep(
                src_points,
                keep_ratio=self.source_keep_ratio,
                rng=source_rng,
                min_points=self.min_source_points,
            )
            src_points, transform = _make_source_augmentation(
                src_points,
                rng=source_rng,
                translation_magnitude=self.augmentation_translation,
            )
            if self.augmentation_noise > 0:
                ref_points = ref_points + target_rng.normal(
                    0.0, self.augmentation_noise, size=ref_points.shape
                ).astype(np.float32)
                src_points = src_points + source_rng.normal(
                    0.0, self.augmentation_noise, size=src_points.shape
                ).astype(np.float32)

        overlap = source_overlap(
            ref_points,
            src_points,
            transform,
            radius=self.matching_radius,
        )

        return {
            "scene_name": pair.case_id,
            "case_id": pair.case_id,
            "chain_id": pair.chain_id,
            "candidate_id": np.asarray(candidate_id, dtype=np.int64),
            "window_center": np.asarray(window_center, dtype=np.float32),
            "crop_radius": np.asarray(crop_radius, dtype=np.float32),
            "crop_diameter": np.asarray(2.0 * crop_radius, dtype=np.float32),
            "chain_radius": np.asarray(chain_radius, dtype=np.float32),
            "chain_diameter": np.asarray(chain_diameter, dtype=np.float32),
            "ref_frame": f"target_candidate{candidate_id:04d}",
            "src_frame": f"chain{pair.chain_id}",
            "overlap": np.asarray(overlap, dtype=np.float32),
            "ref_points": ref_points.astype(np.float32),
            "src_points": src_points.astype(np.float32),
            "ref_feats": np.ones((ref_points.shape[0], 1), dtype=np.float32),
            "src_feats": np.ones((src_points.shape[0], 1), dtype=np.float32),
            "transform": transform.astype(np.float32),
        }



def _dataset_kwargs(cfg) -> Dict[str, object]:
    return {
        "dataset_root": cfg.data.dataset_root,
        "point_suffix": cfg.data.point_suffix,
        "split_file": cfg.data.split_file,
        "split_seed": cfg.data.split_seed,
        "train_ratio": cfg.data.train_ratio,
        "val_ratio": cfg.data.val_ratio,
        "min_source_points": cfg.data.min_source_points,
        "min_target_points": cfg.data.min_target_points,
        "matching_radius": cfg.train.matching_radius,
        "crop_diameter_scale": cfg.crop.diameter_scale,
        "crop_radius_quantile": cfg.crop.radius_quantile,
        "crop_stride_ratio": cfg.crop.stride_ratio,
        "crop_min_stride": cfg.crop.min_stride,
        "crop_min_points": cfg.crop.min_points,
        "crop_max_candidates": cfg.crop.max_candidates,
        "crop_center_jitter_ratio": cfg.crop.train_center_jitter_ratio,
        "crop_min_chain_coverage": cfg.crop.min_chain_coverage,
        "crop_min_oracle_overlap": cfg.crop.min_oracle_overlap,
    }


def train_valid_data_loader(cfg, distributed):
    common = _dataset_kwargs(cfg)
    train_dataset = ProteinPairDataset(
        subset="train",
        point_limit=cfg.train.point_limit,
        use_augmentation=True,
        deterministic_augmentation=False,
        augmentation_seed=cfg.seed,
        augmentation_noise=cfg.train.augmentation_noise,
        augmentation_translation=cfg.train.augmentation_translation,
        source_keep_ratio=cfg.train.source_keep_ratio,
        crop_mode=cfg.crop.train_mode if cfg.crop.enabled else "none",
        **common,
    )
    valid_dataset = ProteinPairDataset(
        subset="val",
        point_limit=cfg.test.point_limit,
        use_augmentation=True,
        deterministic_augmentation=True,
        augmentation_seed=cfg.seed + 100000,
        augmentation_noise=cfg.test.augmentation_noise,
        augmentation_translation=cfg.test.augmentation_translation,
        source_keep_ratio=1.0,
        crop_mode=cfg.crop.val_mode if cfg.crop.enabled else "none",
        **common,
    )

    train_loader = build_dataloader_stack_mode(
        train_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.num_neighbors,
        cfg.backbone.subsample_ratio,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        distributed=distributed,
        precompute_data=True,
    )
    valid_loader = build_dataloader_stack_mode(
        valid_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.num_neighbors,
        cfg.backbone.subsample_ratio,
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False,
        distributed=distributed,
        precompute_data=True,
    )
    return train_loader, valid_loader, cfg.backbone.num_neighbors


def test_data_loader(cfg, benchmark):
    subset = benchmark if benchmark in {"train", "val", "test"} else "test"
    dataset = ProteinPairDataset(
        subset=subset,
        point_limit=cfg.test.point_limit,
        use_augmentation=True,
        deterministic_augmentation=True,
        augmentation_seed=cfg.seed + 200000,
        augmentation_noise=cfg.test.augmentation_noise,
        augmentation_translation=cfg.test.augmentation_translation,
        source_keep_ratio=1.0,
        crop_mode=cfg.crop.test_mode if cfg.crop.enabled else "none",
        **_dataset_kwargs(cfg),
    )
    loader = build_dataloader_stack_mode(
        dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.num_neighbors,
        cfg.backbone.subsample_ratio,
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False,
        precompute_data=True,
    )
    return loader, cfg.backbone.num_neighbors
