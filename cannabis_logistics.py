#!/usr/bin/env python3
"""
Medical cannabis logistics CLI: periodic weigh-ins -> average usage -> reorder forecast
- Subcommands: init, weigh, report, list, check
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

from dotenv import load_dotenv

load_dotenv()

# ---------------- Database schema ----------------
SCHEMA_VERSION = 2  # Incremented when schema changes

def get_schema_version(conn: sqlite3.Connection) -> int:
    """Get current schema version from database."""
    try:
        cursor = conn.execute("SELECT version FROM schema_version ORDER BY version DESC LIMIT 1")
        row = cursor.fetchone()
        return row[0] if row else 0
    except sqlite3.OperationalError:
        # Table doesn't exist yet
        return 0

def set_schema_version(conn: sqlite3.Connection, version: int) -> None:
    """Set schema version in database."""
    conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY, updated_at TEXT NOT NULL)")
    conn.execute("INSERT OR REPLACE INTO schema_version (version, updated_at) VALUES (?, ?)", 
                 (version, to_iso_z(now_utc())))
    conn.commit()

def backup_database(db_path: Path) -> Path:
    """Create a backup of the database using SQLite's backup API. Returns path to backup."""
    timestamp = now_utc().strftime("%Y%m%d_%H%M%S")
    backup_path = db_path.parent / f"{db_path.stem}_backup_{timestamp}{db_path.suffix}"
    
    # Use SQLite's backup API for a consistent snapshot
    source_conn = sqlite3.connect(str(db_path))
    backup_conn = sqlite3.connect(str(backup_path))
    
    with backup_conn:
        source_conn.backup(backup_conn)
    
    backup_conn.close()
    source_conn.close()
    
    return backup_path

def migrate_database(conn: sqlite3.Connection, current_version: int) -> None:
    """Run database migrations from current_version to SCHEMA_VERSION."""
    if current_version >= SCHEMA_VERSION:
        return  # No migration needed
    
    if current_version < 1:
        # Version 0 -> 1: Add finished column
        # Check if column exists first to avoid overwriting data
        cursor = conn.execute("PRAGMA table_info(packages)")
        columns = [row[1] for row in cursor.fetchall()]
        if "finished" not in columns:
            conn.execute("ALTER TABLE packages ADD COLUMN finished INTEGER DEFAULT 0")
    
    if current_version < 2:
        # Version 1 -> 2: Migrate from lead_time_days to new transit/processing columns
        cursor = conn.execute("PRAGMA table_info(packages)")
        columns = [row[1] for row in cursor.fetchall()]
        
        if "transit_days" not in columns:
            conn.execute("ALTER TABLE packages ADD COLUMN transit_days INTEGER DEFAULT 2")
        if "dispensary_processing_days" not in columns:
            conn.execute("ALTER TABLE packages ADD COLUMN dispensary_processing_days INTEGER DEFAULT 1")
        if "post_office_processing_days" not in columns:
            conn.execute("ALTER TABLE packages ADD COLUMN post_office_processing_days INTEGER DEFAULT 1")
        if "skip_weekends" not in columns:
            conn.execute("ALTER TABLE packages ADD COLUMN skip_weekends INTEGER DEFAULT 1")
        
        # Migrate old lead_time_days to new columns if needed
        if "lead_time_days" in columns and "transit_days" in columns:
            # Migrate data: if transit_days is NULL or 0, copy from lead_time_days
            conn.execute("""
                UPDATE packages 
                SET transit_days = COALESCE(transit_days, lead_time_days, 2),
                    dispensary_processing_days = COALESCE(dispensary_processing_days, 1),
                    post_office_processing_days = COALESCE(post_office_processing_days, 1)
                WHERE transit_days IS NULL OR transit_days = 0
            """)
    
    # Update to current schema version
    set_schema_version(conn, SCHEMA_VERSION)
    conn.commit()

def create_tables(conn: sqlite3.Connection) -> None:
    current_version = get_schema_version(conn)
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS packages (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            form TEXT NOT NULL,
            initial_net_g REAL,
            initial_gross_g REAL,
            transit_days INTEGER NOT NULL,
            dispensary_processing_days INTEGER NOT NULL,
            post_office_processing_days INTEGER NOT NULL,
            safety_stock_days INTEGER NOT NULL,
            skip_weekends INTEGER DEFAULT 1,
            thc_percent REAL,
            cbd_percent REAL,
            created_at TEXT NOT NULL,
            finished INTEGER DEFAULT 0
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
	initial_net_g: float
	initial_gross_g: float
	transit_days: int
	dispensary_processing_days: int
	post_office_processing_days: int
	safety_stock_days: int
	skip_weekends: bool = True
	thc_percent: Optional[float] = None
	cbd_percent: Optional[float] = None
	created_at: str = ""
	finished: bool = False


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
    env_base = os.environ.get("LIFESTORE_CANNABIS_DB")
    b = base or env_base or "data/therapeutics/cannabis/cannabis_logistics.db"
    return Path(b)


def get_db_connection(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_exists = db_path.exists()
    conn = sqlite3.connect(str(db_path))
    
    # Check if migration is needed
    current_version = get_schema_version(conn)
    needs_migration = current_version < SCHEMA_VERSION
    
    # Create tables first (safe to run on existing DB)
    create_tables(conn)
    
    # Run migrations if needed
    if needs_migration and db_exists:
        conn.close()
        # Backup before migrating
        backup_path = backup_database(db_path)
        print(f"Created backup: {backup_path}", file=sys.stderr)
        # Reopen and migrate
        conn = sqlite3.connect(str(db_path))
        migrate_database(conn, current_version)
    elif needs_migration:
        # New database, just set version
        migrate_database(conn, current_version)
    
    return conn


def load_package(db_path: Path, pkg_id: str) -> PackageMeta:
    conn = get_db_connection(db_path)
    conn.row_factory = sqlite3.Row  # Enable named column access
    row = conn.execute("SELECT * FROM packages WHERE id = ?", (pkg_id,)).fetchone()
    if not row:
        raise FileNotFoundError(f"Package not found: {pkg_id}")
    meta = PackageMeta(
        id=row["id"],
        name=row["name"],
        form=row["form"],
        initial_net_g=row["initial_net_g"],
        initial_gross_g=row["initial_gross_g"],
        transit_days=row["transit_days"] if row["transit_days"] is not None else 2,
        dispensary_processing_days=row["dispensary_processing_days"] if row["dispensary_processing_days"] is not None else 1,
        post_office_processing_days=row["post_office_processing_days"] if row["post_office_processing_days"] is not None else 1,
        safety_stock_days=row["safety_stock_days"] if row["safety_stock_days"] is not None else 2,
        skip_weekends=bool(row["skip_weekends"]) if row["skip_weekends"] is not None else True,
        thc_percent=row["thc_percent"],
        cbd_percent=row["cbd_percent"],
        created_at=row["created_at"] if row["created_at"] else "",
        finished=bool(row["finished"]) if row["finished"] is not None else False
    )
    conn.close()
    return meta


def iter_packages(db_path: Path) -> List[PackageMeta]:
    conn = get_db_connection(db_path)
    conn.row_factory = sqlite3.Row  # Enable named column access
    rows = conn.execute("SELECT * FROM packages ORDER BY created_at, id").fetchall()
    packages = []
    for row in rows:
        meta = PackageMeta(
            id=row["id"],
            name=row["name"],
            form=row["form"],
            initial_net_g=row["initial_net_g"],
            initial_gross_g=row["initial_gross_g"],
            transit_days=row["transit_days"] if row["transit_days"] is not None else 2,
            dispensary_processing_days=row["dispensary_processing_days"] if row["dispensary_processing_days"] is not None else 1,
            post_office_processing_days=row["post_office_processing_days"] if row["post_office_processing_days"] is not None else 1,
            safety_stock_days=row["safety_stock_days"] if row["safety_stock_days"] is not None else 2,
            skip_weekends=bool(row["skip_weekends"]) if row["skip_weekends"] is not None else True,
            thc_percent=row["thc_percent"],
            cbd_percent=row["cbd_percent"],
            created_at=row["created_at"] if row["created_at"] else "",
            finished=bool(row["finished"]) if row["finished"] is not None else False
        )
        packages.append(meta)
    conn.close()
    return packages


def list_weighins(db_path: Path, pkg_id: str) -> List[WeighIn]:
    conn = get_db_connection(db_path)
    conn.row_factory = sqlite3.Row  # Enable named column access
    rows = conn.execute("SELECT timestamp, gross_g, note FROM weighins WHERE package_id = ? ORDER BY timestamp", (pkg_id,)).fetchall()
    weighins = [WeighIn(timestamp=row["timestamp"], gross_g=row["gross_g"], note=row["note"]) for row in rows]
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
	if meta.created_at:
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
	if meta.created_at:
		try:
			t0 = parse_iso8601_z(meta.created_at)
			if t0 <= ts:
				return meta.initial_gross_g
		except Exception:
			pass
	return None


# ---------------- Usage/forecast logic ----------------
def is_business_day(dt: datetime) -> bool:
	"""Check if a datetime falls on a business day (Mon-Fri)."""
	return dt.weekday() < 5  # 0=Mon, 4=Fri, 5=Sat, 6=Sun


def previous_business_day(dt: datetime) -> datetime:
	"""Walk backward from dt to find the most recent business day (at or before dt)."""
	while not is_business_day(dt):
		dt = dt - timedelta(days=1)
	return dt


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
	elif len(wi_points) == 1 and meta.created_at:
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
		"transit_days": meta.transit_days,
		"dispensary_processing_days": meta.dispensary_processing_days,
		"post_office_processing_days": meta.post_office_processing_days,
		"safety_stock_days": meta.safety_stock_days,
		"skip_weekends": meta.skip_weekends,
		"finished": meta.finished,
	}

	if meta.finished:
		result.update(
			{
				"estimated_depletion_date": None,
				"required_pickup_date": None,
				"required_post_office_arrival_date": None,
				"courier_pickup_date": None,
				"order_by_date": None,
				"order_in_days": None,
				"reorder_now": False,
			}
		)
		return result

	if rate is None or rate <= 0:
		result.update(
			{
				"estimated_depletion_date": None,
				"required_pickup_date": None,
				"required_post_office_arrival_date": None,
				"courier_pickup_date": None,
				"order_by_date": None,
				"order_in_days": None,
				"reorder_now": False,
			}
		)
		return result

	# Step 1: Calculate depletion date (when we run out based on usage)
	days_remaining = current_net / rate if rate > 0 else float("inf")
	depletion_dt = last_ts + timedelta(days=days_remaining)

	# Step 2: Required pickup date = last business day before depletion (minus safety stock)
	# Subtract safety stock first, then adjust to business day
	target_pickup_dt = depletion_dt - timedelta(days=meta.safety_stock_days)
	if meta.skip_weekends:
		required_pickup_dt = previous_business_day(target_pickup_dt)
	else:
		required_pickup_dt = target_pickup_dt

	# Step 3: Required post office arrival = pickup date minus post office processing buffer
	# If pickup is Monday and we need 1 day processing, package must arrive by Friday (not Sunday)
	required_po_arrival_dt = required_pickup_dt - timedelta(days=meta.post_office_processing_days)
	if meta.skip_weekends:
		required_po_arrival_dt = previous_business_day(required_po_arrival_dt)

	# Step 4: Courier pickup from dispensary = post office arrival minus transit time
	courier_pickup_dt = required_po_arrival_dt - timedelta(days=meta.transit_days)
	if meta.skip_weekends:
		courier_pickup_dt = previous_business_day(courier_pickup_dt)

	# Step 5: Order-by date = courier pickup minus dispensary processing buffer
	order_by_dt = courier_pickup_dt - timedelta(days=meta.dispensary_processing_days)
	if meta.skip_weekends:
		order_by_dt = previous_business_day(order_by_dt)

	order_in = math.floor((order_by_dt - as_of).total_seconds() / 86400.0)
	reorder_now = order_in <= 0

	result.update(
		{
			"estimated_depletion_date": to_iso_z(depletion_dt),
			"required_pickup_date": to_iso_z(required_pickup_dt),
			"required_post_office_arrival_date": to_iso_z(required_po_arrival_dt),
			"courier_pickup_date": to_iso_z(courier_pickup_dt),
			"order_by_date": to_iso_z(order_by_dt),
			"order_in_days": round(order_in, 2),
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
	conn.execute("""
		INSERT OR REPLACE INTO packages 
		(id, name, form, initial_net_g, initial_gross_g, transit_days, dispensary_processing_days, 
		 post_office_processing_days, safety_stock_days, skip_weekends, thc_percent, cbd_percent, created_at, finished) 
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
	""",
		(args.id, args.name, args.form, args.initial_net_g, args.initial_gross_g, 
		 args.transit_days, args.dispensary_processing_days, args.post_office_processing_days,
		 args.safety_stock_days, int(args.skip_weekends), args.thc_percent, args.cbd_percent, created_at, 0))
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
	if args.finished:
		conn.execute("UPDATE packages SET finished = 1 WHERE id = ?", (args.id,))
	conn.commit()
	conn.close()
	new_net = net_from_gross(args.gross_g, get_tare(meta))
	print(f"Recorded weigh-in for {args.id}: gross={args.gross_g:.3f}g net={new_net:.3f}g @ {to_iso_z(ts)}")
	if args.finished:
		print(f"Marked package {args.id} as finished.")
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
	print(f"Required pickup: {data['required_pickup_date']}")
	print(f"Post office arrival: {data['required_post_office_arrival_date']}")
	print(f"Courier pickup: {data['courier_pickup_date']}")
	print(f"Order by: {data['order_by_date']}  (in {data['order_in_days']} days)")
	if data.get("finished"):
		action = "FINISHED"
	elif data["reorder_now"]:
		action = "REORDER NOW"
	else:
		action = "OK"
	print(f"Action: {action}")
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
		status = "FINISHED" if r.get("finished") else ("REORDER" if r.get("reorder_now") else "OK")
		print(f"- {r['package_id']}: {r['name']} [{r['form']}] | net {r['current_net_g']} g | usage {r['usage_g_per_day']} g/day | {status}")
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

	any_reorder = any(r.get("reorder_now") for r in results if not r.get("finished"))

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
				if r.get("finished"):
					status = "FINISHED"
				else:
					status = "REORDER" if r.get("reorder_now") else "OK"
				print(f"{r['package_id']}: {status} (order by {r['order_by_date']})")

	# Exit code: 1 if any need reorder, else 0
	return 1 if any_reorder else 0


def cmd_finish(args: argparse.Namespace) -> int:
	db_path = db_path_from_env_or_arg(args.base)
	try:
		meta = load_package(db_path, args.id)
	except FileNotFoundError as e:
		print(str(e), file=sys.stderr)
		return 1

	if meta.finished:
		print(f"Package {args.id} is already finished.", file=sys.stderr)
		return 1

	conn = get_db_connection(db_path)
	conn.execute("UPDATE packages SET finished = 1 WHERE id = ?", (args.id,))
	conn.commit()
	conn.close()
	print(f"Marked package {args.id} as finished.")
	return 0


def build_parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(description="Medical cannabis logistics CLI")
	sub = p.add_subparsers(dest="cmd", required=True)

	default_transit_days = int(os.environ.get('CANNABIS_TRANSIT_DAYS', 2))
	default_dispensary_processing_days = int(os.environ.get('CANNABIS_DISPENSARY_PROCESSING_DAYS', 1))
	default_post_office_processing_days = int(os.environ.get('CANNABIS_POST_OFFICE_PROCESSING_DAYS', 1))
	default_safety_stock = int(os.environ.get('CANNABIS_SAFETY_STOCK_DAYS', 2))
	default_skip_weekends = os.environ.get('CANNABIS_SKIP_WEEKENDS', 'True').lower() in ('true', '1', 'yes')

	# init
	pi = sub.add_parser("init", help="Initialize a package")
	pi.add_argument("--id", required=False, help="Package ID (free-form); auto-generated from name and date if omitted")
	pi.add_argument("--name", required=True, help="Display name")
	pi.add_argument("--form", required=True, help="Form: flower|oil|edible|tincture|capsule|concentrate")
	pi.add_argument("--initial-net-g", type=float, required=True, dest="initial_net_g", help="Initial net mass in grams (required)")
	pi.add_argument("--initial-gross-g", type=float, required=True, dest="initial_gross_g", help="Initial gross mass (g) (required)")
	pi.add_argument("--transit-days", type=int, default=default_transit_days, dest="transit_days", help="Transit time from dispensary to post office in days (default from CANNABIS_TRANSIT_DAYS env var)")
	pi.add_argument("--dispensary-processing-days", type=int, default=default_dispensary_processing_days, dest="dispensary_processing_days", help="Dispensary processing buffer in days (default from CANNABIS_DISPENSARY_PROCESSING_DAYS env var)")
	pi.add_argument("--post-office-processing-days", type=int, default=default_post_office_processing_days, dest="post_office_processing_days", help="Post office processing buffer in days (default from CANNABIS_POST_OFFICE_PROCESSING_DAYS env var)")
	pi.add_argument("--safety-stock-days", type=int, default=default_safety_stock, dest="safety_stock_days", help="Safety stock buffer in days (default from CANNABIS_SAFETY_STOCK_DAYS env var)")
	pi.add_argument("--skip-weekends", type=lambda x: x.lower() in ('true', '1', 'yes'), default=default_skip_weekends, dest="skip_weekends", help="Skip weekends when computing order dates (default from CANNABIS_SKIP_WEEKENDS env var)")
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
	pw.add_argument("--finished", action="store_true", help="Mark the package as finished after this weigh-in")
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

	# finish
	pf = sub.add_parser("finish", help="Mark a package as finished")
	pf.add_argument("--id", required=True, help="Package ID")
	pf.add_argument("--base", default=None, help="Base DB path")
	pf.set_defaults(func=cmd_finish)

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
