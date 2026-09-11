from pathlib import Path

import numpy as np

from scripts.check_data_ready import check_ubfc


def test_incomplete_ubfc_is_not_ready(tmp_path: Path) -> None:
    subject = tmp_path / "subject1"
    subject.mkdir()
    values = " ".join(["1"] * 129)
    (subject / "ground_truth.txt").write_text("\n".join([values] * 3), encoding="utf-8")
    report = check_ubfc(tmp_path, deep=True)
    assert report["ready"] is False
    assert report["labels"] == 1
    assert report["videos"] == 0
