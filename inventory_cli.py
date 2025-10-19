#!/usr/bin/env python3
"""
Command-line interface for querying inventory database.

Usage examples:
    python inventory_cli.py list-boxes
    python inventory_cli.py list-items --box B001
    python inventory_cli.py show-item <item_id>
    python inventory_cli.py update-box B001 --location "Garage shelf 2"
"""
import argparse
import json
import os
import sys
from typing import Optional
from dotenv import load_dotenv

from inventory_db import InventoryDB

# Load environment variables
load_dotenv()
LIFESTORE_DATA_PATH = os.environ.get("LIFESTORE_DATA_PATH", "data")
DEFAULT_DB_PATH = os.path.join(LIFESTORE_DATA_PATH, "inventory", "inventory.db")


def list_boxes(db: InventoryDB):
    """List all boxes."""
    boxes = db.list_boxes()
    if not boxes:
        print("No boxes found.")
        return
    
    print(f"Found {len(boxes)} box(es):\n")
    for box_id in boxes:
        box = db.get_box(box_id)
        print(f"📦 {box_id}")
        if box.get('location'):
            print(f"   Location: {box['location']}")
        if box.get('description'):
            print(f"   Description: {box['description']}")
        
        items = db.list_items(box_id)
        print(f"   Items: {len(items)}")
        print()


def list_items(db: InventoryDB, box_id: Optional[str] = None):
    """List items, optionally filtered by box."""
    items = db.list_items(box_id)
    if not items:
        print("No items found.")
        return
    
    print(f"Found {len(items)} item(s):\n")
    for item in items:
        print(f"📄 {item['item_id']} (Box: {item['box_id']})")
        if item.get('title'):
            print(f"   Title: {item['title']}")
        if item.get('description'):
            print(f"   Description: {item['description']}")
        
        photos = db.get_photos(item['item_id'])
        tags = db.get_tags(item['item_id'])
        print(f"   Photos: {len(photos)}")
        if tags:
            print(f"   Tags: {', '.join(tags)}")
        print()


def show_item(db: InventoryDB, item_id: str):
    """Show detailed information about an item."""
    item = db.get_item(item_id)
    if not item:
        print(f"❌ Item not found: {item_id}")
        return 1
    
    print(f"📄 Item: {item['item_id']}")
    print(f"   Box: {item['box_id']}")
    print(f"   Title: {item.get('title', '(no title)')}")
    print(f"   Description: {item.get('description', '(no description)')}")
    print(f"   Created: {item['created_at']}")
    print(f"   Updated: {item['updated_at']}")
    
    if item.get('tags'):
        print(f"   Tags: {', '.join(item['tags'])}")
    
    print(f"\n   Photos ({len(item['photos'])}):")
    for photo in item['photos']:
        print(f"   - {photo['file_path']}")
        if photo.get('caption'):
            print(f"     Caption: {photo['caption']}")
    
    return 0


def update_box(db: InventoryDB, box_id: str, location: Optional[str] = None,
               description: Optional[str] = None):
    """Update box metadata."""
    if not db.get_box(box_id):
        print(f"❌ Box not found: {box_id}")
        return 1
    
    if db.update_box(box_id, location=location, description=description):
        print(f"✅ Updated box: {box_id}")
        return 0
    else:
        print(f"⚠️  No changes made to box: {box_id}")
        return 0


def update_item(db: InventoryDB, item_id: str, title: Optional[str] = None,
                description: Optional[str] = None, box_id: Optional[str] = None):
    """Update item metadata."""
    if not db.get_item(item_id):
        print(f"❌ Item not found: {item_id}")
        return 1
    
    if db.update_item(item_id, title=title, description=description, box_id=box_id):
        print(f"✅ Updated item: {item_id}")
        return 0
    else:
        print(f"⚠️  No changes made to item: {item_id}")
        return 0


def search_tags(db: InventoryDB, tag: str):
    """Search items by tag."""
    items = db.search_by_tag(tag)
    if not items:
        print(f"No items found with tag: {tag}")
        return
    
    print(f"Found {len(items)} item(s) with tag '{tag}':\n")
    for item in items:
        print(f"📄 {item['item_id']} (Box: {item['box_id']})")
        if item.get('title'):
            print(f"   Title: {item['title']}")
        print()


def export_json(db: InventoryDB, output_file: str):
    """Export entire database to JSON."""
    data = {
        'boxes': [],
        'items': []
    }
    
    for box_id in db.list_boxes():
        box = db.get_box(box_id)
        data['boxes'].append(box)
    
    for item in db.list_items():
        item_data = db.get_item(item['item_id'])
        data['items'].append(item_data)
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Exported to: {output_file}")


def main(argv=None):
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Inventory database CLI")
    parser.add_argument(
        "--db",
        default=DEFAULT_DB_PATH,
        help=f"Path to SQLite database (default: {DEFAULT_DB_PATH})"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # list-boxes
    subparsers.add_parser('list-boxes', help='List all boxes')
    
    # list-items
    list_items_parser = subparsers.add_parser('list-items', help='List items')
    list_items_parser.add_argument('--box', help='Filter by box ID')
    
    # show-item
    show_item_parser = subparsers.add_parser('show-item', help='Show item details')
    show_item_parser.add_argument('item_id', help='Item ID')
    
    # update-box
    update_box_parser = subparsers.add_parser('update-box', help='Update box metadata')
    update_box_parser.add_argument('box_id', help='Box ID')
    update_box_parser.add_argument('--location', help='Physical location')
    update_box_parser.add_argument('--description', help='Box description')
    
    # update-item
    update_item_parser = subparsers.add_parser('update-item', help='Update item metadata')
    update_item_parser.add_argument('item_id', help='Item ID')
    update_item_parser.add_argument('--title', help='Item title')
    update_item_parser.add_argument('--description', help='Item description')
    update_item_parser.add_argument('--box', help='Move to different box')
    
    # search-tag
    search_tag_parser = subparsers.add_parser('search-tag', help='Search items by tag')
    search_tag_parser.add_argument('tag', help='Tag to search for')
    
    # export
    export_parser = subparsers.add_parser('export', help='Export database to JSON')
    export_parser.add_argument('output', help='Output file path')
    
    args = parser.parse_args(argv)
    
    if not args.command:
        parser.print_help()
        return 1
    
    db = InventoryDB(args.db)
    
    if args.command == 'list-boxes':
        list_boxes(db)
        return 0
    
    elif args.command == 'list-items':
        list_items(db, args.box)
        return 0
    
    elif args.command == 'show-item':
        return show_item(db, args.item_id)
    
    elif args.command == 'update-box':
        return update_box(db, args.box_id, args.location, args.description)
    
    elif args.command == 'update-item':
        return update_item(db, args.item_id, args.title, args.description, args.box)
    
    elif args.command == 'search-tag':
        search_tags(db, args.tag)
        return 0
    
    elif args.command == 'export':
        export_json(db, args.output)
        return 0
    
    return 1


if __name__ == "__main__":
    sys.exit(main())
