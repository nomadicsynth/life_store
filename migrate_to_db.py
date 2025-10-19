#!/usr/bin/env python3
"""
Migration script to import existing filesystem inventory data into SQLite database.

This script scans the data/inventory directory and populates the database with:
- Boxes (from Box_* directories)
- Items (from Item_* subdirectories)
- Photos (from photo_*.jpg files)
- Metadata from item.json files if present
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from inventory_db import InventoryDB

# Load environment variables
load_dotenv()
LIFESTORE_DATA_PATH = os.environ.get("LIFESTORE_DATA_PATH", "data")
DEFAULT_INVENTORY_ROOT = os.path.join(LIFESTORE_DATA_PATH, "inventory")
DEFAULT_DB_PATH = os.path.join(LIFESTORE_DATA_PATH, "inventory", "inventory.db")


def parse_timestamp_from_filename(filename: str) -> str:
    """
    Extract timestamp from photo filename like photo_20250913_121045_1.jpg.
    
    Returns ISO 8601 format or current time if parsing fails.
    """
    try:
        # Extract YYYYMMDD_HHMMSS part
        parts = filename.split('_')
        if len(parts) >= 3 and parts[0] == 'photo':
            date_str = parts[1]  # YYYYMMDD
            time_str = parts[2]  # HHMMSS
            dt = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except (ValueError, IndexError):
        pass
    
    # Fallback to current time
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def migrate_inventory(inventory_root: str, db_path: str, dry_run: bool = False):
    """
    Migrate filesystem inventory to database.
    
    Args:
        inventory_root: Path to inventory root (e.g., data/inventory)
        db_path: Path to SQLite database
        dry_run: If True, only print what would be done
    """
    if not os.path.isdir(inventory_root):
        print(f"❌ Inventory root not found: {inventory_root}")
        return 1
    
    db = InventoryDB(db_path)
    stats = {'boxes': 0, 'items': 0, 'photos': 0, 'metadata': 0}
    
    # Scan for boxes
    for box_name in sorted(os.listdir(inventory_root)):
        if not box_name.startswith("Box_"):
            continue
        
        box_path = os.path.join(inventory_root, box_name)
        if not os.path.isdir(box_path):
            continue
        
        box_id = box_name[4:]  # Remove "Box_" prefix
        
        if dry_run:
            print(f"[DRY RUN] Would create box: {box_id}")
        else:
            if db.create_box(box_id):
                print(f"✅ Created box: {box_id}")
                stats['boxes'] += 1
            else:
                print(f"ℹ️  Box already exists: {box_id}")
        
        # Scan for items in this box
        for item_name in sorted(os.listdir(box_path)):
            if not item_name.startswith("Item_"):
                continue
            
            item_path = os.path.join(box_path, item_name)
            if not os.path.isdir(item_path):
                continue
            
            item_id = item_name[5:]  # Remove "Item_" prefix
            
            # Check for item.json
            item_json_path = os.path.join(item_path, "item.json")
            title = ""
            description = ""
            tags = []
            
            if os.path.isfile(item_json_path):
                try:
                    with open(item_json_path, 'r') as f:
                        metadata = json.load(f)
                        title = metadata.get('title', '')
                        description = metadata.get('description', '')
                        tags = metadata.get('tags', [])
                        stats['metadata'] += 1
                except Exception as e:
                    print(f"⚠️  Error reading {item_json_path}: {e}")
            
            if dry_run:
                print(f"  [DRY RUN] Would create item: {item_id} in box {box_id}")
            else:
                if db.create_item(item_id, box_id, title=title, description=description):
                    print(f"  ✅ Created item: {item_id}")
                    stats['items'] += 1
                    
                    # Add tags
                    for tag in tags:
                        db.add_tag(item_id, tag)
                else:
                    print(f"  ℹ️  Item already exists: {item_id}")
            
            # Scan for photos
            for filename in sorted(os.listdir(item_path)):
                if not filename.startswith("photo_") or not filename.endswith(".jpg"):
                    continue
                
                photo_path = os.path.join(item_path, filename)
                if not os.path.isfile(photo_path):
                    continue
                
                # Relative path from inventory root
                relative_path = os.path.join(box_name, item_name, filename)
                
                if dry_run:
                    print(f"    [DRY RUN] Would add photo: {filename}")
                else:
                    db.add_photo(item_id, relative_path)
                    stats['photos'] += 1
    
    # Print summary
    print("\n" + "="*60)
    if dry_run:
        print("DRY RUN SUMMARY (no changes made)")
    else:
        print("MIGRATION SUMMARY")
    print("="*60)
    print(f"Boxes:    {stats['boxes']}")
    print(f"Items:    {stats['items']}")
    print(f"Photos:   {stats['photos']}")
    print(f"Metadata: {stats['metadata']} item.json files found")
    print("="*60)
    
    return 0


def main(argv=None):
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Migrate filesystem inventory to SQLite database"
    )
    parser.add_argument(
        "--inventory",
        default=DEFAULT_INVENTORY_ROOT,
        help=f"Path to inventory root directory (default: {DEFAULT_INVENTORY_ROOT})"
    )
    parser.add_argument(
        "--db",
        default=DEFAULT_DB_PATH,
        help=f"Path to SQLite database (default: {DEFAULT_DB_PATH})"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    
    args = parser.parse_args(argv)
    
    return migrate_inventory(args.inventory, args.db, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
