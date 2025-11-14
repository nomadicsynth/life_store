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
SCHEMA_VERSION = 5  # Incremented when schema changes

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
    
    if current_version < 3:
        # Version 2 -> 3: Add package_cost, reordered, reordered_date, and finished_date columns
        cursor = conn.execute("PRAGMA table_info(packages)")
        columns = [row[1] for row in cursor.fetchall()]
        if "package_cost" not in columns:
            conn.execute("ALTER TABLE packages ADD COLUMN package_cost REAL")
        if "reordered" not in columns:
            conn.execute("ALTER TABLE packages ADD COLUMN reordered INTEGER DEFAULT 0")
        if "reordered_date" not in columns:
            conn.execute("ALTER TABLE packages ADD COLUMN reordered_date TEXT")
        if "finished_date" not in columns:
            conn.execute("ALTER TABLE packages ADD COLUMN finished_date TEXT")

    if current_version < 4:
		# Version 3 -> 4: Fix: "Unexpected error: NOT NULL constraint failed: packages.lead_time_days" when adding new packages
        cursor = conn.execute("PRAGMA table_info(packages)")
        columns = [row[1] for row in cursor.fetchall()]
        if "lead_time_days" in columns:
            conn.execute("ALTER TABLE packages DROP COLUMN lead_time_days")
    
    if current_version < 5:
        # Version 4 -> 5: Add weight_discrepancy_g column to track discrepancy when package is finished
        cursor = conn.execute("PRAGMA table_info(packages)")
        columns = [row[1] for row in cursor.fetchall()]
        if "weight_discrepancy_g" not in columns:
            conn.execute("ALTER TABLE packages ADD COLUMN weight_discrepancy_g REAL")
            
            # Clear any discrepancy values for unfinished packages (safety check)
			# TODO: not sure if this is needed. think it through and remove if redundant.
            conn.execute("UPDATE packages SET weight_discrepancy_g = NULL WHERE finished = 0")
            
            # Calculate discrepancy for existing finished packages
            # For each finished package, get the latest weigh-in and calculate net weight
            # Discrepancy = computed_net - actual (where actual is 0 when finished)
            finished_packages = conn.execute("""
                SELECT id, initial_net_g, initial_gross_g 
                FROM packages 
                WHERE finished = 1
            """).fetchall()
            
            for pkg_id, initial_net_g, initial_gross_g in finished_packages:
                # Get the latest weigh-in for this package
                latest_weighin = conn.execute("""
                    SELECT gross_g 
                    FROM weighins 
                    WHERE package_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT 1
                """, (pkg_id,)).fetchone()
                
                if latest_weighin:
                    # Calculate computed net weight from latest weigh-in
                    gross_g = latest_weighin[0]
                    tare = initial_gross_g - initial_net_g if (initial_gross_g is not None and initial_net_g is not None) else None
                    computed_net = gross_g - tare if tare is not None else gross_g
                    # Discrepancy = computed_net
                    discrepancy = computed_net
                else:
                    # No weigh-ins, discrepancy = 0.0 to avoid biasing reports with incorrect values
                    discrepancy = 0.0
                
                # Update the package with the calculated discrepancy
                conn.execute("""
                    UPDATE packages 
                    SET weight_discrepancy_g = ? 
                    WHERE id = ?
                """, (discrepancy, pkg_id))
    
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
            package_cost REAL,
            created_at TEXT NOT NULL,
            finished INTEGER DEFAULT 0,
            finished_date TEXT,
            reordered INTEGER DEFAULT 0,
            reordered_date TEXT,
            weight_discrepancy_g REAL
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
	package_cost: Optional[float] = None
	created_at: str = ""
	finished: bool = False
	finished_date: Optional[str] = None
	reordered: bool = False
	reordered_date: Optional[str] = None
	weight_discrepancy_g: Optional[float] = None


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
        package_cost=row["package_cost"] if "package_cost" in row.keys() else None,
        created_at=row["created_at"] if row["created_at"] else "",
        finished=bool(row["finished"]) if "finished" in row.keys() and row["finished"] is not None else False,
        finished_date=row["finished_date"] if "finished_date" in row.keys() and row["finished_date"] else None,
        reordered=bool(row["reordered"]) if "reordered" in row.keys() and row["reordered"] is not None else False,
        reordered_date=row["reordered_date"] if "reordered_date" in row.keys() and row["reordered_date"] else None,
        weight_discrepancy_g=row["weight_discrepancy_g"] if "weight_discrepancy_g" in row.keys() and row["weight_discrepancy_g"] is not None else None
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
            package_cost=row["package_cost"] if "package_cost" in row.keys() else None,
            created_at=row["created_at"] if row["created_at"] else "",
            finished=bool(row["finished"]) if "finished" in row.keys() and row["finished"] is not None else False,
            finished_date=row["finished_date"] if "finished_date" in row.keys() and row["finished_date"] else None,
            reordered=bool(row["reordered"]) if "reordered" in row.keys() and row["reordered"] is not None else False,
            reordered_date=row["reordered_date"] if "reordered_date" in row.keys() and row["reordered_date"] else None,
            weight_discrepancy_g=row["weight_discrepancy_g"] if "weight_discrepancy_g" in row.keys() and row["weight_discrepancy_g"] is not None else None
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
	"""Calculate net weight from gross, clipping to 0 to avoid negative values for forecasting."""
	if tare_g is None:
		return gross_g
	return max(gross_g - tare_g, 0.0)

def net_from_gross_unclipped(gross_g: float, tare_g: Optional[float]) -> float:
	"""Calculate net weight from gross without clipping. Use for discrepancy calculations."""
	if tare_g is None:
		return gross_g
	return gross_g - tare_g


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


def latest_state(meta: PackageMeta, weighins: List[WeighIn]) -> Tuple[datetime, float, float]:
	"""Return (timestamp, current_net_g). If no weigh-ins, fall back to initial."""
	if weighins:
		w = max(weighins, key=lambda wi: parse_iso8601_z(wi.timestamp))
		t = parse_iso8601_z(w.timestamp)
		last_gross = w.gross_g
		tare = get_tare(meta)
		n = net_from_gross(last_gross, tare)
		return t, last_gross, n
	else:
		if not meta.created_at:
			raise ValueError(f"Package {meta.id} has no created_at timestamp")
		t0 = parse_iso8601_z(meta.created_at)
		last_gross = meta.initial_gross_g or 0.0
		n0 = meta.initial_net_g or 0.0
		return t0, last_gross, n0


def forecast(meta: PackageMeta, weighins: List[WeighIn], as_of: Optional[datetime] = None) -> Dict[str, Any]:
	as_of = as_of or now_utc()
	rate = usage_rate_g_per_day(meta, weighins)  # may be None
	last_ts, last_gross, current_net = latest_state(meta, weighins)

	result: Dict[str, Any] = {
		"package_id": meta.id,
		"last_weigh_in_at": to_iso_z(last_ts),
		"last_weigh_in_gross_g": last_gross,
		"as_of": to_iso_z(as_of),
		"current_net_g": round(current_net, 3),
		"usage_g_per_day": round(rate, 4) if rate is not None else None,
		"transit_days": meta.transit_days,
		"dispensary_processing_days": meta.dispensary_processing_days,
		"post_office_processing_days": meta.post_office_processing_days,
		"safety_stock_days": meta.safety_stock_days,
		"skip_weekends": meta.skip_weekends,
		"finished": meta.finished,
		"finished_date": meta.finished_date,
		"reordered": meta.reordered,
		"reordered_date": meta.reordered_date,
		"package_cost": meta.package_cost,
		"cost_per_gram": round(meta.package_cost / meta.initial_net_g, 2) if meta.package_cost is not None and meta.initial_net_g and meta.initial_net_g > 0 else None,
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
	if meta.reordered:
		reorder_now = False
	else:
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


# ---------------- Budgeting module helper ----------------
def get_all_forecasts(db_path: Optional[Path] = None, as_of: Optional[datetime] = None) -> List[Dict[str, Any]]:
	"""Get forecasts for all packages. Intended for use by budgeting module.
	
	Args:
		db_path: Path to database (defaults to standard path)
		as_of: Forecast 'as of' timestamp (defaults to now)
	
	Returns:
		List of forecast dictionaries, one per package
	"""
	if db_path is None:
		db_path = db_path_from_env_or_arg(None)
	
	forecasts = []
	for meta in iter_packages(db_path):
		weighins = list_weighins(db_path, meta.id)
		data = forecast(meta, weighins, as_of=as_of)
		# Enrich with name and form for convenience
		data["name"] = meta.name
		data["form"] = meta.form
		forecasts.append(data)
	
	return forecasts


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
		 post_office_processing_days, safety_stock_days, skip_weekends, thc_percent, cbd_percent, package_cost, created_at, finished, finished_date, reordered, reordered_date, weight_discrepancy_g) 
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
	""",
		(args.id, args.name, args.form, args.initial_net_g, args.initial_gross_g, 
		 args.transit_days, args.dispensary_processing_days, args.post_office_processing_days,
		 args.safety_stock_days, int(args.skip_weekends), args.thc_percent, args.cbd_percent, args.package_cost, created_at, 0, None, 0, None, None))
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
		finished_date = to_iso_z(ts)
		# Calculate discrepancy using unclipped net weight to capture actual discrepancy
		# even when gross < tare (which would give negative net)
		computed_net_unclipped = net_from_gross_unclipped(args.gross_g, get_tare(meta))
		discrepancy = computed_net_unclipped
		conn.execute("UPDATE packages SET finished = 1, finished_date = ?, weight_discrepancy_g = ? WHERE id = ?", (finished_date, discrepancy, args.id))
	conn.commit()
	conn.close()
	# Use clipped net for display (for consistency with forecasting)
	computed_net = net_from_gross(args.gross_g, get_tare(meta))
	print(f"Recorded weigh-in for {args.id}: gross={args.gross_g:.3f}g net={computed_net:.3f}g @ {to_iso_z(ts)}")
	if args.finished:
		print(f"Marked package {args.id} as finished.")
		print(f"Weight discrepancy: {discrepancy:.3f}g")
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
	print(f"As of: {data['as_of']}")
	print(f"Initial gross: {meta.initial_gross_g:.2f}g at {meta.created_at}")
	print(f"Initial net: {meta.initial_net_g:.2f}g")
	print(f"Last weigh-in: gross {data['last_weigh_in_gross_g']:.2f}g at {data['last_weigh_in_at']}")
	print(f"Current net: {data['current_net_g']} g")
	print(f"Usage: {data['usage_g_per_day']} g/day")
	print(f"Depletion: {data['estimated_depletion_date']}")
	print(f"Required pickup: {data['required_pickup_date']}")
	print(f"Post office arrival: {data['required_post_office_arrival_date']}")
	print(f"Courier pickup: {data['courier_pickup_date']}")
	print(f"Order by: {data['order_by_date']} (in {data['order_in_days']} days)")

	# Set the status
	status = "Ok"
	if data.get("finished"):
		status = "Finished"
		if data.get("finished_date"):
			status_date = data.get("finished_date")
			status += f" on {status_date}"
	print(f"Status: {status}")
	
	# Show weight discrepancy if package is finished
	if meta.finished and meta.weight_discrepancy_g is not None:
		print(f"Weight discrepancy: {meta.weight_discrepancy_g:.2f}g")

	# Set the action
	action = "No action required"
	if data.get("reorder_now"):
		action = "REORDER NOW"
	elif data.get("reordered"):
		action += " | Reordered"
		if data.get("reordered_date"):
			action_date = data.get("reordered_date")
			action += f" on {action_date}."
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

	# Calculate discrepancy using unclipped net weight to capture actual discrepancy
	# even when gross < tare (which would give negative net)
	weighins = list_weighins(db_path, args.id)
	if weighins:
		# Get the latest weigh-in directly to calculate unclipped net
		w = max(weighins, key=lambda wi: parse_iso8601_z(wi.timestamp))
		computed_net_unclipped = net_from_gross_unclipped(w.gross_g, get_tare(meta))
		discrepancy = computed_net_unclipped
	else:
		# No weigh-ins, use initial net weight
		discrepancy = meta.initial_net_g if meta.initial_net_g is not None else 0.0

	conn = get_db_connection(db_path)
	finished_date = to_iso_z(now_utc())
	conn.execute("UPDATE packages SET finished = 1, finished_date = ?, weight_discrepancy_g = ? WHERE id = ?", (finished_date, discrepancy, args.id))
	conn.commit()
	conn.close()
	print(f"Marked package {args.id} as finished.")
	print(f"Weight discrepancy: {discrepancy:.2f}g")
	return 0


def cmd_reorder(args: argparse.Namespace) -> int:
	db_path = db_path_from_env_or_arg(args.base)
	try:
		meta = load_package(db_path, args.id)
	except FileNotFoundError as e:
		print(str(e), file=sys.stderr)
		return 1

	if meta.reordered:
		print(f"Package {args.id} is already marked as reordered.", file=sys.stderr)
		return 1

	conn = get_db_connection(db_path)
	# check if reorder-date is a valid ISO-8601 timestamp
	if args.reorder_date and not parse_iso8601_z(args.reorder_date):
		print(f"Invalid reorder-date: {args.reorder_date}", file=sys.stderr)
		return 1
	reorder_date = args.reorder_date if args.reorder_date else to_iso_z(now_utc())
	conn.execute("UPDATE packages SET reordered = 1, reordered_date = ? WHERE id = ?", (reorder_date, args.id))
	conn.commit()
	conn.close()
	print(f"Marked package {args.id} as reordered on {reorder_date}.")
	return 0

def cmd_edit(args: argparse.Namespace) -> int:
	db_path = db_path_from_env_or_arg(args.base)
	try:
		meta = load_package(db_path, args.id)
	except FileNotFoundError as e:
		print(str(e), file=sys.stderr)
		return 1

	conn = get_db_connection(db_path)
	
	# Build dynamic UPDATE statement based on provided fields
	updates = []
	values = []
	
	if args.name is not None:
		updates.append("name = ?")
		values.append(args.name)
	if args.form is not None:
		updates.append("form = ?")
		values.append(args.form)
	if args.initial_net_g is not None:
		updates.append("initial_net_g = ?")
		values.append(args.initial_net_g)
	if args.initial_gross_g is not None:
		updates.append("initial_gross_g = ?")
		values.append(args.initial_gross_g)
	if args.transit_days is not None:
		updates.append("transit_days = ?")
		values.append(args.transit_days)
	if args.dispensary_processing_days is not None:
		updates.append("dispensary_processing_days = ?")
		values.append(args.dispensary_processing_days)
	if args.post_office_processing_days is not None:
		updates.append("post_office_processing_days = ?")
		values.append(args.post_office_processing_days)
	if args.safety_stock_days is not None:
		updates.append("safety_stock_days = ?")
		values.append(args.safety_stock_days)
	if args.skip_weekends is not None:
		updates.append("skip_weekends = ?")
		values.append(int(args.skip_weekends))
	if args.thc_percent is not None:
		updates.append("thc_percent = ?")
		values.append(args.thc_percent)
	if args.cbd_percent is not None:
		updates.append("cbd_percent = ?")
		values.append(args.cbd_percent)
	if args.package_cost is not None:
		updates.append("package_cost = ?")
		values.append(args.package_cost)
	if args.finished is not None:
		updates.append("finished = ?")
		values.append(int(args.finished))
		# If marking as not finished, clear discrepancy (it's only valid when finished)
		if not args.finished:
			updates.append("weight_discrepancy_g = ?")
			values.append(None)
	if args.reordered is not None:
		updates.append("reordered = ?")
		values.append(int(args.reordered))
	
	if not updates:
		print("No fields to update. Provide at least one field to edit.", file=sys.stderr)
		conn.close()
		return 1
	
	values.append(args.id)
	update_sql = f"UPDATE packages SET {', '.join(updates)} WHERE id = ?"
	conn.execute(update_sql, values)
	conn.commit()
	conn.close()
	
	updated_fields = [u.split(' =')[0] for u in updates]
	print(f"Updated package {args.id}: {', '.join(updated_fields)}")
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
	pi.add_argument("--package-cost", type=float, default=None, dest="package_cost", help="Cost of the package (for budgeting)")
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

	# edit
	pe = sub.add_parser("edit", help="Edit package details")
	pe.add_argument("--id", required=True, help="Package ID")
	pe.add_argument("--name", default=None, help="Display name")
	pe.add_argument("--form", default=None, help="Form: flower|oil|edible|tincture|capsule|concentrate")
	pe.add_argument("--initial-net-g", type=float, default=None, dest="initial_net_g", help="Initial net mass in grams")
	pe.add_argument("--initial-gross-g", type=float, default=None, dest="initial_gross_g", help="Initial gross mass (g)")
	pe.add_argument("--transit-days", type=int, default=None, dest="transit_days", help="Transit time from dispensary to post office in days")
	pe.add_argument("--dispensary-processing-days", type=int, default=None, dest="dispensary_processing_days", help="Dispensary processing buffer in days")
	pe.add_argument("--post-office-processing-days", type=int, default=None, dest="post_office_processing_days", help="Post office processing buffer in days")
	pe.add_argument("--safety-stock-days", type=int, default=None, dest="safety_stock_days", help="Safety stock buffer in days")
	pe.add_argument("--skip-weekends", type=lambda x: x.lower() in ('true', '1', 'yes'), default=None, dest="skip_weekends", help="Skip weekends when computing order dates (true/false)")
	pe.add_argument("--thc-percent", type=float, default=None, dest="thc_percent", help="THC percentage")
	pe.add_argument("--cbd-percent", type=float, default=None, dest="cbd_percent", help="CBD percentage")
	pe.add_argument("--package-cost", type=float, default=None, dest="package_cost", help="Cost of the package (for budgeting)")
	pe.add_argument("--finished", type=lambda x: x.lower() in ('true', '1', 'yes'), default=None, dest="finished", help="Mark package as finished (true/false)")
	pe.add_argument("--reordered", type=lambda x: x.lower() in ('true', '1', 'yes'), default=None, dest="reordered", help="Mark package as reordered (true/false)")
	pe.add_argument("--base", default=None, help="Base DB path")
	pe.set_defaults(func=cmd_edit)

	# reorder
	prd = sub.add_parser("reorder", help="Mark a package as reordered")
	prd.add_argument("--id", required=True, help="Package ID")
	prd.add_argument("--reorder-date", default=None, help="ISO-8601 timestamp for reorder date. Defaults to now UTC.")
	prd.add_argument("--base", default=None, help="Base DB path")
	prd.set_defaults(func=cmd_reorder)

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
