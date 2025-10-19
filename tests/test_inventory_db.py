"""
Tests for inventory_db module.
"""
import os
import tempfile
from pathlib import Path

import pytest

from inventory_db import InventoryDB


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        db = InventoryDB(db_path)
        yield db


def test_create_and_get_box(temp_db):
    """Test creating and retrieving a box."""
    assert temp_db.create_box("B001", location="Garage", description="Storage box")
    
    box = temp_db.get_box("B001")
    assert box is not None
    assert box['box_id'] == "B001"
    assert box['location'] == "Garage"
    assert box['description'] == "Storage box"
    assert 'created_at' in box
    assert 'updated_at' in box


def test_create_duplicate_box(temp_db):
    """Test that creating a duplicate box returns False."""
    assert temp_db.create_box("B001")
    assert not temp_db.create_box("B001")


def test_list_boxes(temp_db):
    """Test listing boxes."""
    temp_db.create_box("B001")
    temp_db.create_box("B002")
    temp_db.create_box("A001")
    
    boxes = temp_db.list_boxes()
    assert boxes == ["A001", "B001", "B002"]


def test_update_box(temp_db):
    """Test updating box metadata."""
    temp_db.create_box("B001", location="Garage")
    
    assert temp_db.update_box("B001", location="Attic")
    
    box = temp_db.get_box("B001")
    assert box['location'] == "Attic"


def test_update_nonexistent_box(temp_db):
    """Test updating a box that doesn't exist."""
    assert not temp_db.update_box("B999", location="Nowhere")


def test_create_and_get_item(temp_db):
    """Test creating and retrieving an item."""
    temp_db.create_box("B001")
    
    assert temp_db.create_item("item001", "B001", title="Sweater", description="Blue wool")
    
    item = temp_db.get_item("item001")
    assert item is not None
    assert item['item_id'] == "item001"
    assert item['box_id'] == "B001"
    assert item['title'] == "Sweater"
    assert item['description'] == "Blue wool"
    assert item['photos'] == []
    assert item['tags'] == []


def test_create_duplicate_item(temp_db):
    """Test that creating a duplicate item returns False."""
    temp_db.create_box("B001")
    assert temp_db.create_item("item001", "B001")
    assert not temp_db.create_item("item001", "B001")


def test_list_items(temp_db):
    """Test listing items."""
    temp_db.create_box("B001")
    temp_db.create_box("B002")
    
    temp_db.create_item("item001", "B001")
    temp_db.create_item("item002", "B001")
    temp_db.create_item("item003", "B002")
    
    all_items = temp_db.list_items()
    assert len(all_items) == 3
    
    box1_items = temp_db.list_items("B001")
    assert len(box1_items) == 2
    assert all(item['box_id'] == "B001" for item in box1_items)


def test_update_item(temp_db):
    """Test updating item metadata."""
    temp_db.create_box("B001")
    temp_db.create_item("item001", "B001", title="Old Title")
    
    assert temp_db.update_item("item001", title="New Title", description="New desc")
    
    item = temp_db.get_item("item001")
    assert item['title'] == "New Title"
    assert item['description'] == "New desc"


def test_move_item_to_different_box(temp_db):
    """Test moving an item to a different box."""
    temp_db.create_box("B001")
    temp_db.create_box("B002")
    temp_db.create_item("item001", "B001")
    
    assert temp_db.update_item("item001", box_id="B002")
    
    item = temp_db.get_item("item001")
    assert item['box_id'] == "B002"


def test_add_and_get_photos(temp_db):
    """Test adding photos to an item."""
    temp_db.create_box("B001")
    temp_db.create_item("item001", "B001")
    
    photo_id1 = temp_db.add_photo("item001", "Box_B001/Item_item001/photo_1.jpg", 
                                   caption="Front view", hash_sha256="abc123")
    photo_id2 = temp_db.add_photo("item001", "Box_B001/Item_item001/photo_2.jpg")
    
    assert photo_id1 > 0
    assert photo_id2 > 0
    
    photos = temp_db.get_photos("item001")
    assert len(photos) == 2
    assert photos[0]['file_path'] == "Box_B001/Item_item001/photo_1.jpg"
    assert photos[0]['caption'] == "Front view"
    assert photos[0]['hash_sha256'] == "abc123"


def test_add_and_get_tags(temp_db):
    """Test adding tags to an item."""
    temp_db.create_box("B001")
    temp_db.create_item("item001", "B001")
    
    tag_id1 = temp_db.add_tag("item001", "clothing")
    tag_id2 = temp_db.add_tag("item001", "winter")
    
    assert tag_id1 > 0
    assert tag_id2 > 0
    
    tags = temp_db.get_tags("item001")
    assert tags == ["clothing", "winter"]


def test_remove_tag(temp_db):
    """Test removing a tag from an item."""
    temp_db.create_box("B001")
    temp_db.create_item("item001", "B001")
    temp_db.add_tag("item001", "clothing")
    temp_db.add_tag("item001", "winter")
    
    assert temp_db.remove_tag("item001", "clothing")
    
    tags = temp_db.get_tags("item001")
    assert tags == ["winter"]


def test_search_by_tag(temp_db):
    """Test searching items by tag."""
    temp_db.create_box("B001")
    temp_db.create_item("item001", "B001")
    temp_db.create_item("item002", "B001")
    temp_db.create_item("item003", "B001")
    
    temp_db.add_tag("item001", "clothing")
    temp_db.add_tag("item002", "clothing")
    temp_db.add_tag("item003", "electronics")
    
    results = temp_db.search_by_tag("clothing")
    assert len(results) == 2
    assert all(item['item_id'] in ["item001", "item002"] for item in results)


def test_get_item_includes_photos_and_tags(temp_db):
    """Test that get_item includes photos and tags."""
    temp_db.create_box("B001")
    temp_db.create_item("item001", "B001")
    temp_db.add_photo("item001", "photo1.jpg")
    temp_db.add_photo("item001", "photo2.jpg")
    temp_db.add_tag("item001", "tag1")
    temp_db.add_tag("item001", "tag2")
    
    item = temp_db.get_item("item001")
    assert len(item['photos']) == 2
    assert len(item['tags']) == 2


def test_cascade_delete_items_when_box_deleted(temp_db):
    """Test that deleting a box cascades to items (via foreign key)."""
    # Note: This test demonstrates the FK constraint but we don't have
    # a delete_box method yet. If needed, we can add one.
    temp_db.create_box("B001")
    temp_db.create_item("item001", "B001")
    
    # Manually delete box to test cascade
    with temp_db._get_connection() as conn:
        conn.execute("DELETE FROM boxes WHERE box_id = ?", ("B001",))
    
    item = temp_db.get_item("item001")
    assert item is None


def test_empty_database(temp_db):
    """Test operations on empty database."""
    assert temp_db.list_boxes() == []
    assert temp_db.list_items() == []
    assert temp_db.get_box("nonexistent") is None
    assert temp_db.get_item("nonexistent") is None
