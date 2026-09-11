from __future__ import annotations

from scripts.audit_reproduction_completion import check


def test_completion_check_is_fail_closed() -> None:
    item = check("evidence", False, "missing.json", "not available")
    assert item["achieved"] is False
    assert item["evidence"] == "missing.json"
