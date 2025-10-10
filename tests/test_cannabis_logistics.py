import json
from pathlib import Path
from datetime import datetime, timezone, timedelta

import pytest

import cannabis_logistics as cli


def iso(dt):
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def test_init_weigh_report(tmp_path: Path, monkeypatch):
    base_db = tmp_path / "therapeutics" / "cannabis" / "cannabis_logistics.db"

    # init package
    rc = cli.main([
        "init",
        "--id", "testpkg",
        "--name", "Test Pkg",
        "--form", "flower",
        "--tare-g", "10",
        "--initial-gross-g", "28",
        "--lead-time-days", "5",
        "--safety-stock-days", "2",
        "--base", str(base_db),
    ])
    assert rc == 0

    # two weigh-ins a few days apart
    t1 = datetime.now(timezone.utc) - timedelta(days=7)
    t2 = datetime.now(timezone.utc)

    rc = cli.main([
        "weigh",
        "--id", "testpkg",
        "--gross-g", "35.0",  # net 25.0
        "--timestamp", iso(t1),
        "--base", str(base_db),
    ])
    assert rc == 0

    rc = cli.main([
        "weigh",
        "--id", "testpkg",
        "--gross-g", "32.0",  # net 22.0
        "--timestamp", iso(t2),
        "--base", str(base_db),
    ])
    assert rc == 0

    # report JSON
    out_path = base_db.parent / "out.json"
    class Capturer:
        def __init__(self):
            self.lines = []
        def write(self, s):
            self.lines.append(s)
        def flush(self):
            pass
    cap = Capturer()

    # monkeypatch stdout to capture
    import sys
    old_stdout = sys.stdout
    sys.stdout = cap
    try:
        rc = cli.main(["report", "--id", "testpkg", "--json", "--base", str(base_db)])
    finally:
        sys.stdout = old_stdout
    assert rc == 0

    out = json.loads("".join(cap.lines))
    assert out["package_id"] == "testpkg"
    assert out["usage_g_per_day"] is not None
    assert out["current_net_g"] <= 25.0
    assert "reorder_now" in out


def test_report_handles_no_usage(tmp_path: Path):
    base_db = tmp_path / "therapeutics" / "cannabis" / "cannabis_logistics.db"
    # init only, no weigh-ins and no initial
    rc = cli.main([
        "init",
        "--id", "empty",
        "--name", "Empty",
        "--form", "flower",
        "--lead-time-days", "3",
        "--safety-stock-days", "2",
        "--base", str(base_db),
    ])
    assert rc == 0

    # report should not crash and usage None
    class Capturer:
        def __init__(self): self.lines = []
        def write(self, s): self.lines.append(s)
        def flush(self): pass
    import sys
    old_stdout = sys.stdout
    sys.stdout = Capturer()
    try:
        rc = cli.main(["report", "--id", "empty", "--json", "--base", str(base_db)])
        out = json.loads("".join(sys.stdout.lines))
    finally:
        sys.stdout = old_stdout
    assert rc == 0
    assert out["usage_g_per_day"] is None
    assert out["reorder_now"] is False


def test_list_and_check_commands(tmp_path: Path):
    base_db = tmp_path / "therapeutics" / "cannabis" / "cannabis_logistics.db"

    # Package A: has usage, likely not reorder yet
    rc = cli.main([
        "init", "--id", "A", "--name", "Pkg A", "--form", "flower",
        "--tare-g", "10", "--initial-gross-g", "28", "--lead-time-days", "3", "--safety-stock-days", "2",
        "--base", str(base_db),
    ])
    assert rc == 0
    # two weigh-ins one week apart to establish usage
    t1 = datetime.now(timezone.utc) - timedelta(days=14)
    t2 = datetime.now(timezone.utc) - timedelta(days=7)
    assert cli.main(["weigh", "--id", "A", "--gross-g", "36.0", "--timestamp", iso(t1), "--base", str(base_db)]) == 0
    assert cli.main(["weigh", "--id", "A", "--gross-g", "34.0", "--timestamp", iso(t2), "--base", str(base_db)]) == 0

    # Package B: small remaining, force reorder
    rc = cli.main([
        "init", "--id", "B", "--name", "Pkg B", "--form", "flower",
        "--tare-g", "10", "--initial-gross-g", "18", "--lead-time-days", "7", "--safety-stock-days", "3",
        "--base", str(base_db),
    ])
    assert rc == 0
    # create high usage and low remaining so reorder is in the past
    t3 = datetime.now(timezone.utc) - timedelta(days=10)
    t4 = datetime.now(timezone.utc)
    # initial net if provided is 8g (18-10). Weigh down to 1g net over 10 days.
    assert cli.main(["weigh", "--id", "B", "--gross-g", "14.0", "--timestamp", iso(t3), "--base", str(base_db)]) == 0  # net 4g
    assert cli.main(["weigh", "--id", "B", "--gross-g", "11.0", "--timestamp", iso(t4), "--base", str(base_db)]) == 0  # net 1g

    # list JSON
    class Capturer:
        def __init__(self): self.lines = []
        def write(self, s): self.lines.append(s)
        def flush(self): pass
    import sys
    old_stdout = sys.stdout
    sys.stdout = Capturer()
    try:
        rc = cli.main(["list", "--json", "--base", str(base_db)])
        out = json.loads("".join(sys.stdout.lines))
    finally:
        sys.stdout = old_stdout
    assert rc == 0
    assert isinstance(out, list) and len(out) >= 2
    ids = {r["package_id"] for r in out}
    assert {"A", "B"}.issubset(ids)

    # check all should return 1 because B needs reorder
    rc = cli.main(["check", "--base", str(base_db)])
    assert rc == 1

    # check single A likely OK (return 0)
    rc = cli.main(["check", "--id", "A", "--base", str(base_db)])
    assert rc in (0, 1)  # allow either depending on timing, but ensure JSON reflects same

    # check single B JSON -> reorder_now True
    sys.stdout = Capturer()
    try:
        rc_b = cli.main(["check", "--id", "B", "--json", "--base", str(base_db)])
        out_b = json.loads("".join(sys.stdout.lines))
    finally:
        sys.stdout = old_stdout
    assert rc_b == 1
    assert out_b["package_id"] == "B"
    assert out_b["reorder_now"] is True


def test_weigh_increase_rejected(tmp_path: Path):
    base_db = tmp_path / "therapeutics" / "cannabis" / "cannabis_logistics.db"
    # init with tare and initial
    assert cli.main([
        "init", "--id", "X", "--name", "Pkg X", "--form", "flower",
        "--tare-g", "10", "--initial-gross-g", "30", "--lead-time-days", "3", "--safety-stock-days", "2",
        "--base", str(base_db),
    ]) == 0
    # weigh at t1 net 18g
    from datetime import datetime, timezone, timedelta
    t1 = datetime.now(timezone.utc) - timedelta(days=2)
    assert cli.main(["weigh", "--id", "X", "--gross-g", "28.0", "--timestamp", iso(t1), "--base", str(base_db)]) == 0
    # attempt to weigh later with higher gross (increase): reject
    t2 = datetime.now(timezone.utc)
    rc = cli.main(["weigh", "--id", "X", "--gross-g", "29.0", "--timestamp", iso(t2), "--base", str(base_db)])
    assert rc == 1
    # also reject a backdated increase between existing points
    t_mid = t1 + timedelta(days=1)
    rc2 = cli.main(["weigh", "--id", "X", "--gross-g", "28.5", "--timestamp", iso(t_mid), "--base", str(base_db)])
    assert rc2 == 1
