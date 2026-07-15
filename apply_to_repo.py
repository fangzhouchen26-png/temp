from __future__ import annotations

import argparse
from pathlib import Path
import shutil


NEW_POSE_FUNCTION = r'''@torch.no_grad()
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
    """Create deterministic multi-correspondence equivariant pose hypotheses.

    Each seed builds a small distance-compatible correspondence group. Vector
    covariances are accumulated across that group before SVD, so rank-one
    vector sets at individual points can jointly constrain all rotation axes.
    """
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
        0,
        len(source_indices) - 1,
        count,
        device=source_points.device,
    ).long().unique()
    source_correspondences = source_points[source_indices]
    target_correspondences = target_points[target_indices]
    group_size = min(8, len(source_indices))
    best_transform = None
    best_support = 0
    best_residual = torch.tensor(float("inf"), device=source_points.device)

    for hypothesis_index in hypothesis_indices.tolist():
        source_radii = torch.linalg.norm(
            source_correspondences - source_correspondences[hypothesis_index],
            dim=1,
        )
        target_radii = torch.linalg.norm(
            target_correspondences - target_correspondences[hypothesis_index],
            dim=1,
        )
        compatibility_error = (source_radii - target_radii).abs()
        group = compatibility_error.topk(group_size, largest=False).indices

        try:
            rotation, singular_values = _equivariant_svd_rotation(
                source_equivariant[source_indices[group]].reshape(-1, 3),
                target_equivariant[target_indices[group]].reshape(-1, 3),
            )
        except RuntimeError:
            continue
        if not torch.isfinite(singular_values).all():
            continue
        if singular_values[0] <= 1e-8:
            continue
        if (
            singular_values[1] / singular_values[0] < 0.05
            or singular_values[2] / singular_values[0] < 0.01
        ):
            continue

        translations = (
            target_correspondences[group]
            - source_correspondences[group] @ rotation.T
        )
        translation = translations.median(dim=0).values
        transform = torch.eye(
            4,
            dtype=source_points.dtype,
            device=source_points.device,
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
'''


def replace_function(text: str) -> str:
    start_marker = "@torch.no_grad()\ndef equivariant_pose_hypotheses("
    end_marker = "\n\ndef compute_ops_scores("
    start = text.find(start_marker)
    end = text.find(end_marker, start)
    if start < 0 or end < 0:
        raise RuntimeError("could not locate equivariant_pose_hypotheses in model.py")
    return text[:start] + NEW_POSE_FUNCTION + text[end:]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo", type=Path, help="path to the temp repository")
    args = parser.parse_args()
    repo = args.repo.resolve()
    package = Path(__file__).resolve().parent

    matcher_target = repo / "cryo_registration" / "fine_point_matching.py"
    model_target = repo / "cryo_registration" / "model.py"
    test_target = repo / "tests" / "test_equivariant_channel_diversity.py"
    for path in (matcher_target, model_target):
        if not path.is_file():
            raise FileNotFoundError(path)

    shutil.copy2(
        package / "cryo_registration" / "fine_point_matching.py",
        matcher_target,
    )
    model_text = model_target.read_text(encoding="utf-8")
    model_target.write_text(replace_function(model_text), encoding="utf-8")
    test_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        package / "tests" / "test_equivariant_channel_diversity.py",
        test_target,
    )
    print("Applied equivariant channel-collapse and multi-point pose fixes.")
    print("Run: pytest -q tests/test_fine_point_matching.py "
          "tests/test_fine_point_matching_training.py "
          "tests/test_equivariant_channel_diversity.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
