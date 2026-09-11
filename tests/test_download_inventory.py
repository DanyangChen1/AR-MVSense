from pathlib import Path

import pytest

from scripts.download_datasets import UBFC_SUBJECTS, _ubfc_target


def test_ubfc_target_accepts_subject_relative_path(tmp_path: Path) -> None:
    assert _ubfc_target(tmp_path, "subject1/vid.avi") == tmp_path / "subject1" / "vid.avi"


def test_ubfc_target_rejects_path_traversal(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        _ubfc_target(tmp_path, "../vid.avi")


def test_ubfc_subject_inventory_is_the_official_42_subject_subset() -> None:
    assert len(UBFC_SUBJECTS) == 42
    assert len(set(UBFC_SUBJECTS)) == 42
    assert set(UBFC_SUBJECTS).isdisjoint({2, 6, 7, 19, 21, 28, 29})
    assert 27 in UBFC_SUBJECTS
