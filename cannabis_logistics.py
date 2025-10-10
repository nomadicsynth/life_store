#!/usr/bin/env python3
"""
Medical cannabis logistics CLI: periodic weigh-ins -> average usage -> reorder forec# ---------------- Data model ----------------choices
- File-first storage under therapeutics/cannabis/Package_<ID>/
- JSON-only metadata and entries; no DB
- Subcommands: init, weigh, report
- Exposes main(argv) for easy testing (repo convention)

Exit codes
- 0: success
- 1: validation/usage errors
- 2: unexpected error
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import math
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from statistics import median
from typing import List, Optional, Tuple, Dict, Any


# ---------------- Database schema ----------------
def create_tables(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS packages (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            form TEXT NOT NULL,
            initial_net_g REAL,
            initial_gross_g REAL,
            lead_time_days INTEGER NOT NULL,
            safety_stock_days INTEGER NOT NULL,
            thc_percent REAL,
            cbd_percent REAL,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS weighins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            package_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            gross_g REAL NOT NULL,
            note TEXT,
            FOREIGN KEY (package_id) REFERENCES packages(id)
        )
    """)
    conn.commit()


# ---------------- Time helpers ----------------
def now_utc() -> datetime:
	return datetime.now(timezone.utc)


def parse_iso8601_z(s: str) -> datetime:
	"""Accept '...Z' or full offset; return timezone-aware UTC datetime."""
	try:
		if s.endswith("Z"):
			return datetime.fromisoformat(s.replace("Z", "+00:00"))
		# If no offset info, assume UTC
		dt = datetime.fromisoformat(s)
		if dt.tzinfo is None:
			return dt.replace(tzinfo=timezone.utc)
		return dt.astimezone(timezone.utc)
	except Exception as e:
		raise ValueError(f"Invalid ISO-8601 timestamp: {s}") from e


def to_iso_z(dt: datetime) -> str:
	return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------- Data model ----------------
@dataclass
class PackageMeta:
	id: str
	name: str
	form: str  # flower | oil | edible | tincture | capsule | concentrate
	initial_net_g: Optional[float]
	initial_gross_g: Optional[float]
	lead_time_days: int
	safety_stock_days: int
	thc_percent: Optional[float] = None
	cbd_percent: Optional[float] = None
	created_at: str = ""


@dataclass
class WeighIn:
	timestamp: str  # ISO-8601 Z
	gross_g: float  # gross weight of the jar
	note: Optional[str] = None


def get_tare(meta: PackageMeta) -> Optional[float]:
	if meta.initial_gross_g is not None and meta.initial_net_g is not None:
		return meta.initial_gross_g - meta.initial_net_g
	return None


# ---------------- IO helpers ----------------
def read_json(path: Path) -> Dict[str, Any]:
	with path.open("r", encoding="utf-8") as f:
		return json.load(f)


def write_json(path: Path, data: Dict[str, Any]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as f:
		json.dump(data, f, indent=2, ensure_ascii=False)
		f.write("\n")


# ---------------- Core paths ----------------
def db_path_from_env_or_arg(base: Optional[str]) -> Path:
    # Allow env override; default to repo-local data/therapeutics/cannabis/cannabis_logistics.db
    env_base = os.environ.get("THERAPEUTICS_CANNABIS_DB")
    b = base or env_base or "data/therapeutics/cannabis/cannabis_logistics.db"
    return Path(b)


def get_db_connection(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    create_tables(conn)
    return conn


def load_package(db_path: Path, pkg_id: str) -> PackageMeta:
    conn = get_db_connection(db_path)
    row = conn.execute("SELECT * FROM packages WHERE id = ?", (pkg_id,)).fetchone()
    if not row:
        raise FileNotFoundError(f"Package not found: {pkg_id}")
    meta = PackageMeta(
        id=row[0],
        name=row[1],
        form=row[2],
        initial_net_g=row[3],
        initial_gross_g=row[4],
        lead_time_days=row[5],
        safety_stock_days=row[6],
        thc_percent=row[7],
        cbd_percent=row[8],
        created_at=row[9]
    )
    conn.close()
    return meta


def iter_packages(db_path: Path) -> List[PackageMeta]:
    conn = get_db_connection(db_path)
    rows = conn.execute("SELECT * FROM packages ORDER BY created_at, id").fetchall()
    packages = []
    for row in rows:
        meta = PackageMeta(
            id=row[0],
            name=row[1],
            form=row[2],
            initial_net_g=row[3],
            initial_gross_g=row[4],
            lead_time_days=row[5],
            safety_stock_days=row[6],
            thc_percent=row[7],
            cbd_percent=row[8],
            created_at=row[9]
        )
        packages.append(meta)
    conn.close()
    return packages


def list_weighins(db_path: Path, pkg_id: str) -> List[WeighIn]:
    conn = get_db_connection(db_path)
    rows = conn.execute("SELECT timestamp, gross_g, note FROM weighins WHERE package_id = ? ORDER BY timestamp", (pkg_id,)).fetchall()
    weighins = [WeighIn(timestamp=row[0], gross_g=row[1], note=row[2]) for row in rows]
    conn.close()
    return weighins


def previous_net_at_or_before(meta: PackageMeta, db_path: Path, ts: datetime) -> Optional[float]:
	"""Return the net grams from the latest data point at or before ts.

	Rule of thumb:
	- Prefer the latest weigh-in at or before ts.
	- If no weigh-in exists, fall back to the initial snapshot if its timestamp <= ts.
	- Otherwise, None.
	"""
	latest_wi: Optional[Tuple[datetime, float]] = None
	for w in list_weighins(db_path, meta.id):
		try:
			tw = parse_iso8601_z(w.timestamp)
			if tw <= ts:
				nw = net_from_gross(w.gross_g, get_tare(meta))
				if latest_wi is None or tw > latest_wi[0]:
					latest_wi = (tw, nw)
		except Exception:
			continue
	if latest_wi is not None:
		return latest_wi[1]

	# No weigh-ins; consider initial snapshot
	if meta.initial_gross_g is not None and meta.created_at:
		try:
			t0 = parse_iso8601_z(meta.created_at)
			if t0 <= ts:
				return meta.initial_net_g
		except Exception:
			pass
	return None


def previous_gross_at_or_before(meta: PackageMeta, db_path: Path, ts: datetime) -> Optional[float]:
	"""Return the gross grams from the latest data point at or before ts.

	Prefer latest weigh-in; if none, fall back to initial_gross_g if created_at <= ts.
	"""
	latest_wi: Optional[Tuple[datetime, float]] = None
	for w in list_weighins(db_path, meta.id):
		try:
			tw = parse_iso8601_z(w.timestamp)
			if tw <= ts:
				if latest_wi is None or tw > latest_wi[0]:
					latest_wi = (tw, w.gross_g)
		except Exception:
			continue
	if latest_wi is not None:
		return latest_wi[1]
	if meta.initial_gross_g is not None and meta.created_at:
		try:
			t0 = parse_iso8601_z(meta.created_at)
			if t0 <= ts:
				return float(meta.initial_gross_g)
		except Exception:
			pass
	return None


# ---------------- Usage/forecast logic ----------------
def net_from_gross(gross_g: float, tare_g: Optional[float]) -> float:
	if tare_g is None:
		return gross_g
	return max(gross_g - tare_g, 0.0)


def usage_rate_g_per_day(meta: PackageMeta, weighins: List[WeighIn]) -> Optional[float]:
	"""Estimate usage rate in grams/day using median of pairwise consumption rates.

	Prefer using weigh-ins only (robust to backdated entries). If there are
	fewer than two weigh-ins, fall back to combining a single weigh-in with the
	synthetic initial (if available). Returns None if rate cannot be estimated.
	"""
	wi_points: List[Tuple[datetime, float]] = []
	for w in weighins:
		try:
			t = parse_iso8601_z(w.timestamp)
			n = net_from_gross(w.gross_g, get_tare(meta))
			wi_points.append((t, n))
		except Exception:
			continue

	wi_points.sort(key=lambda x: x[0])

	points: List[Tuple[datetime, float]]
	if len(wi_points) >= 2:
		points = wi_points
	elif len(wi_points) == 1 and meta.initial_gross_g is not None and meta.created_at:
		# Use the single weigh-in and the initial snapshot
		try:
			t0 = parse_iso8601_z(meta.created_at)
			n0 = meta.initial_net_g
			points = sorted([wi_points[0], (t0, n0)], key=lambda x: x[0])
		except Exception:
			return None
	else:
		return None

	rates: List[float] = []
	for (t_prev, n_prev), (t_cur, n_cur) in zip(points, points[1:]):
		dt_days = (t_cur - t_prev).total_seconds() / 86400.0
		if dt_days <= 0.0:
			continue
		consumed = max(n_prev - n_cur, 0.0)
		if consumed <= 0.0:
			continue
		rates.append(consumed / dt_days)

	if not rates:
		return None
	try:
		return max(median(rates), 0.0)
	except Exception:
		# Fallback to mean if median fails for some reason
		return max(sum(rates) / len(rates), 0.0)


def latest_state(meta: PackageMeta, weighins: List[WeighIn]) -> Tuple[datetime, float]:
	"""Return (timestamp, current_net_g). If no weigh-ins, fall back to initial."""
	if weighins:
		w = max(weighins, key=lambda wi: parse_iso8601_z(wi.timestamp))
		t = parse_iso8601_z(w.timestamp)
		n = net_from_gross(w.gross_g, get_tare(meta))
		return t, n
	# Fallback
	t0 = parse_iso8601_z(meta.created_at) if meta.created_at else now_utc()
	n0 = meta.initial_net_g or 0.0
	return t0, n0


def forecast(meta: PackageMeta, weighins: List[WeighIn], as_of: Optional[datetime] = None) -> Dict[str, Any]:
	as_of = as_of or now_utc()
	rate = usage_rate_g_per_day(meta, weighins)  # may be None
	last_ts, current_net = latest_state(meta, weighins)

	result: Dict[str, Any] = {
		"package_id": meta.id,
		"last_weigh_in_at": to_iso_z(last_ts),
		"as_of": to_iso_z(as_of),
		"current_net_g": round(current_net, 3),
		"usage_g_per_day": round(rate, 4) if rate is not None else None,
		"lead_time_days": meta.lead_time_days,
		"safety_stock_days": meta.safety_stock_days,
	}

	if rate is None or rate <= 0:
		result.update(
			{
				"estimated_depletion_date": None,
				"reorder_date": None,
				"reorder_in_days": None,
				"reorder_now": False,
			}
		)
		return result

	days_remaining = current_net / rate if rate > 0 else float("inf")
	depletion_dt = last_ts + timedelta(days=days_remaining)
	# Reorder date is depletion minus (lead + safety). If that is before as_of, reorder now.
	buffer_days = meta.lead_time_days + meta.safety_stock_days
	reorder_dt = depletion_dt - timedelta(days=buffer_days)
	reorder_in = math.floor((reorder_dt - as_of).total_seconds() / 86400.0)
	reorder_now = reorder_in <= 0

	result.update(
		{
			"estimated_depletion_date": to_iso_z(depletion_dt),
			"reorder_date": to_iso_z(reorder_dt),
			"reorder_in_days": round(reorder_in, 2),
			"reorder_now": bool(reorder_now),
		}
	)
	return result


# ---------------- CLI impl ----------------
def cmd_init(args: argparse.Namespace) -> int:
	db_path = db_path_from_env_or_arg(args.base)
	if not args.id:
		name_slug = args.name.lower().replace(' ', '_').replace('-', '_')
		date_str = now_utc().strftime("%Y%m%d")
		args.id = f"{name_slug}_{date_str}"
	conn = get_db_connection(db_path)
	existing = conn.execute("SELECT id FROM packages WHERE id = ?", (args.id,)).fetchone()
	if existing and not args.force:
		print(f"Package already exists: {args.id}", file=sys.stderr)
		conn.close()
		return 1

	if args.created_at:
		created_at = to_iso_z(parse_iso8601_z(args.created_at))
	else:
		created_at = to_iso_z(now_utc())
	conn.execute("INSERT OR REPLACE INTO packages (id, name, form, initial_net_g, initial_gross_g, lead_time_days, safety_stock_days, thc_percent, cbd_percent, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
		(args.id, args.name, args.form, args.initial_net_g, args.initial_gross_g, args.lead_time_days, args.safety_stock_days, args.thc_percent, args.cbd_percent, created_at))
	conn.commit()
	conn.close()
	print(f"Initialized package {args.id}")
	return 0


def cmd_weigh(args: argparse.Namespace) -> int:
	db_path = db_path_from_env_or_arg(args.base)
	try:
		meta = load_package(db_path, args.id)
	except FileNotFoundError as e:
		print(str(e), file=sys.stderr)
		return 1

	ts = parse_iso8601_z(args.timestamp) if args.timestamp else now_utc()
	# Reject increases vs previous gross at or before this timestamp
	prev_gross = previous_gross_at_or_before(meta, db_path, ts)
	if prev_gross is not None and args.gross_g > prev_gross:
		print(
			f"Error: gross increased ({args.gross_g:.3f}g) vs previous {prev_gross:.3f}g at or before {to_iso_z(ts)}; rejecting entry.",
			file=sys.stderr,
		)
		return 1

	conn = get_db_connection(db_path)
	conn.execute("INSERT INTO weighins (package_id, timestamp, gross_g, note) VALUES (?, ?, ?, ?)", (args.id, to_iso_z(ts), args.gross_g, args.note))
	conn.commit()
	conn.close()
	new_net = net_from_gross(args.gross_g, get_tare(meta))
	print(f"Recorded weigh-in for {args.id}: gross={args.gross_g:.3f}g net={new_net:.3f}g @ {to_iso_z(ts)}")
	return 0


def cmd_report(args: argparse.Namespace) -> int:
	db_path = db_path_from_env_or_arg(args.base)
	try:
		meta = load_package(db_path, args.id)
	except FileNotFoundError as e:
		print(str(e), file=sys.stderr)
		return 1

	weighins = list_weighins(db_path, args.id)
	as_of = parse_iso8601_z(args.as_of) if args.as_of else None
	data = forecast(meta, weighins, as_of=as_of)

	if args.json:
		print(json.dumps(data, indent=2))
		return 0

	# Human-friendly output
	print(f"Package: {data['package_id']}")
	print(f"As of:   {data['as_of']}")
	print(f"Last weigh-in: {data['last_weigh_in_at']}")
	print(f"Current net: {data['current_net_g']} g")
	print(f"Usage: {data['usage_g_per_day']} g/day")
	print(f"Depletion: {data['estimated_depletion_date']}")
	print(f"Reorder date: {data['reorder_date']}  (in {data['reorder_in_days']} days)")
	print("Action: REORDER NOW" if data["reorder_now"] else "Action: OK")
	return 0


def cmd_list(args: argparse.Namespace) -> int:
	db_path = db_path_from_env_or_arg(args.base)
	rows: List[Dict[str, Any]] = []
	for meta in iter_packages(db_path):
		data = forecast(meta, list_weighins(db_path, meta.id))
		# enrich
		data["name"] = meta.name
		data["form"] = meta.form
		rows.append(data)

	if args.json:
		print(json.dumps(rows, indent=2))
		return 0

	if not rows:
		print("No packages found.")
		return 0

	for r in rows:
		print(f"- {r['package_id']}: {r['name']} [{r['form']}] | net {r['current_net_g']} g | usage {r['usage_g_per_day']} g/day | reorder {r['reorder_now']}")
	return 0


def cmd_check(args: argparse.Namespace) -> int:
	db_path = db_path_from_env_or_arg(args.base)
	results: List[Dict[str, Any]] = []

	if args.id:
		try:
			meta = load_package(db_path, args.id)
		except FileNotFoundError as e:
			print(str(e), file=sys.stderr)
			return 1
		results.append(forecast(meta, list_weighins(db_path, args.id)))
	else:
		for meta in iter_packages(db_path):
			results.append(forecast(meta, list_weighins(db_path, meta.id)))

	any_reorder = any(r.get("reorder_now") for r in results)

	if args.json:
		if args.id:
			print(json.dumps(results[0], indent=2))
		else:
			print(json.dumps(results, indent=2))
	else:
		if not results:
			print("No packages found.")
		else:
			for r in results:
				status = "REORDER" if r.get("reorder_now") else "OK"
				print(f"{r['package_id']}: {status} (reorder_date {r['reorder_date']})")

	# Exit code: 1 if any need reorder, else 0
	return 1 if any_reorder else 0

	# Human-friendly output
	# TODO: unreachable - fix.
	print(f"Package: {data['package_id']}")
	print(f"As of:   {data['as_of']}")
	print(f"Last weigh-in: {data['last_weigh_in_at']}")
	print(f"Current net: {data['current_net_g']} g")
	print(f"Usage: {data['usage_g_per_day']} g/day")
	print(f"Depletion: {data['estimated_depletion_date']}")
	print(f"Reorder date: {data['reorder_date']}  (in {data['reorder_in_days']} days)")
	print("Action: REORDER NOW" if data["reorder_now"] else "Action: OK")
	return 0


def build_parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(description="Medical cannabis logistics CLI")
	sub = p.add_subparsers(dest="cmd", required=True)

	# init
	pi = sub.add_parser("init", help="Initialize a package")
	pi.add_argument("--id", required=False, help="Package ID (free-form); auto-generated from name and date if omitted")
	pi.add_argument("--name", required=True, help="Display name")
	pi.add_argument("--form", required=True, help="Form: flower|oil|edible|tincture|capsule|concentrate")
	pi.add_argument("--initial-net-g", type=float, default=None, dest="initial_net_g", help="Initial net mass in grams")
	pi.add_argument("--initial-gross-g", type=float, default=None, dest="initial_gross_g", help="Initial gross mass (g)")
	pi.add_argument("--lead-time-days", type=int, required=True, default=0, dest="lead_time_days", help="Supplier lead time in days")
	pi.add_argument("--safety-stock-days", type=int, default=0, dest="safety_stock_days", help="Safety stock buffer in days")
	pi.add_argument("--thc-percent", type=float, default=None, dest="thc_percent")
	pi.add_argument("--cbd-percent", type=float, default=None, dest="cbd_percent")
	pi.add_argument("--base", default=None, help="Base DB path (default data/therapeutics/cannabis/cannabis_logistics.db)")
	pi.add_argument("--force", action="store_true", help="Overwrite existing package metadata")
	pi.add_argument("--created-at", default=None, help="ISO-8601 timestamp for package creation; default now UTC")
	pi.set_defaults(func=cmd_init)

	# weigh
	pw = sub.add_parser("weigh", help="Record a weigh-in for a package")
	pw.add_argument("--id", required=True, help="Package ID")
	pw.add_argument("--gross-g", required=True, type=float, dest="gross_g", help="Gross package mass (g)")
	pw.add_argument("--timestamp", default=None, help="ISO-8601 timestamp; default now UTC")
	pw.add_argument("--note", default=None, help="Optional note")
	pw.add_argument("--base", default=None, help="Base DB path")
	pw.set_defaults(func=cmd_weigh)

	# report
	pr = sub.add_parser("report", help="Show usage and reorder forecast")
	pr.add_argument("--id", required=True, help="Package ID")
	pr.add_argument("--as-of", default=None, help="Report 'as of' timestamp (ISO-8601)")
	pr.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
	pr.add_argument("--base", default=None, help="Base DB path")
	pr.set_defaults(func=cmd_report)

	# list
	pl = sub.add_parser("list", help="List packages with current status")
	pl.add_argument("--json", action="store_true", help="Emit machine-readable JSON array")
	pl.add_argument("--base", default=None, help="Base DB path")
	pl.set_defaults(func=cmd_list)

	# check
	pc = sub.add_parser("check", help="Check reorder status; non-zero exit if reorder needed")
	pc.add_argument("--id", help="Package ID (if omitted, checks all)")
	pc.add_argument("--json", action="store_true", help="Emit JSON (object for single, array for all)")
	pc.add_argument("--base", default=None, help="Base DB path")
	pc.set_defaults(func=cmd_check)

	return p


def main(argv: Optional[List[str]] = None) -> int:
	argv = argv if argv is not None else sys.argv[1:]
	parser = build_parser()
	try:
		args = parser.parse_args(argv)
		return args.func(args)
	except SystemExit as e:
		# argparse uses SystemExit on parsing errors
		return 1 if e.code != 0 else 0
	except Exception as e:
		print(f"Unexpected error: {e}", file=sys.stderr)
		return 2


if __name__ == "__main__":
	sys.exit(main())

