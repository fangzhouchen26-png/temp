from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from cryo_registration.fine_shot import (
    attach_fine_shot_4a,
    select_fine_shot_levels,
)


def _structure() -> dict:
    return {
        "normalization": {
            "center": torch.tensor([10.0, 20.0, 30.0]),
            "scale": 10.0,
        },
        "target": {
            "2.00": torch.tensor(
                [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.4, 0.0, 0.0]]
            ),
            "4.00": torch.tensor(
                [[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]]
            ),
            "6.00": torch.tensor([[0.0, 0.0, 0.0]]),
        },
        "target_normals": torch.tensor(
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        ),
        "chains": {
            "A": {
                "2.00": torch.tensor(
                    [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.4, 0.0, 0.0]]
                ),
                "4.00": torch.tensor(
                    [[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]]
                ),
                "6.00": torch.tensor([[0.0, 0.0, 0.0]]),
            }
        },
        "chain_normals": {
            "A": torch.tensor(
                [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
            )
        },
        "chain_shot": {
            "A": {
                "features": torch.ones(1, 352),
                "valid_mask": torch.ones(1, dtype=torch.bool),
            }
        },
        "target_shot": {
            "features": torch.ones(1, 352),
            "valid_mask": torch.ones(1, dtype=torch.bool),
        },
    }


def test_fine_shot_uses_2a_support_4a_keypoints_and_original_angstrom(tmp_path) -> None:
    structure = _structure()
    calls = []

    def fake_shot(support, normals, keypoints, **kwargs):
        calls.append((support.copy(), normals.copy(), keypoints.copy(), kwargs))
        features = np.ones((len(keypoints), 352), dtype=np.float32)
        return features, np.ones(len(keypoints), dtype=bool)

    attach_fine_shot_4a(
        structure,
        "synthetic",
        tmp_path,
        radius_angstrom=15.0,
        shot_computer=fake_shot,
    )

    source, target = select_fine_shot_levels(structure, "A")
    assert source["features"].shape == (2, 352)
    assert target["features"].shape == (2, 352)
    assert len(calls) == 2
    expected_target_keypoints = (
        _structure()["target"]["4.00"].numpy() * 10.0
        + np.array([10.0, 20.0, 30.0])
    )
    assert np.allclose(calls[0][2], expected_target_keypoints)
    assert calls[0][0].shape[0] == len(structure["target"]["2.00"])
    assert calls[0][3]["radius_angstrom"] == 15.0
    assert (tmp_path / "synthetic.pt").is_file()


def test_fine_shot_reuses_sidecar_cache(tmp_path) -> None:
    first = _structure()

    def fake_shot(support, normals, keypoints, **kwargs):
        return (
            np.ones((len(keypoints), 352), dtype=np.float32),
            np.ones(len(keypoints), dtype=bool),
        )

    attach_fine_shot_4a(first, "synthetic", tmp_path, shot_computer=fake_shot)
    second = _structure()

    def fail_if_called(*args, **kwargs):
        raise AssertionError("SHOT should be loaded from cache")

    attach_fine_shot_4a(second, "synthetic", tmp_path, shot_computer=fail_if_called)
    assert torch.equal(
        second["chain_shot_4a"]["A"]["features"],
        first["chain_shot_4a"]["A"]["features"],
    )


def test_fine_shot_rejects_misaligned_6a_fallback() -> None:
    structure = _structure()
    with pytest.raises(ValueError, match="4 A chain SHOT"):
        select_fine_shot_levels(structure, "A")

    structure["chain_shot_4a"] = {
        "A": {
            "features": torch.ones(2, 352),
            "valid_mask": torch.ones(2, dtype=torch.bool),
        }
    }
    structure["target_shot_4a"] = {
        "features": torch.ones(2, 352),
        "valid_mask": torch.ones(2, dtype=torch.bool),
    }
    source, target = select_fine_shot_levels(structure, "A")
    assert len(source["features"]) == 2
    assert len(target["features"]) == 2
