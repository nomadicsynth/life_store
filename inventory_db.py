"""
SQLite database layer for inventory metadata.

Stores box and item metadata while photos remain on filesystem.
"""
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Optional, List, Dict, Any
import os
import json


class InventoryDB:
    """Manages inventory metadata in SQLite."""
    
    def __init__(self, db_path: str = "data/inventory.db"):
        """Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
        self._init_db()
    
    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def _init_db(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Enable foreign key constraints
            cursor.execute("PRAGMA foreign_keys = ON")
            
            # Boxes table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS boxes (
                    box_id TEXT PRIMARY KEY,
                    location TEXT,
                    description TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            # Items table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS items (
                    item_id TEXT PRIMARY KEY,
                    box_id TEXT NOT NULL,
                    title TEXT,
                    description TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (box_id) REFERENCES boxes(box_id) ON DELETE CASCADE
                )
            """)
            
            # Photos table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS photos (
                    photo_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    item_id TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    caption TEXT,
                    hash_sha256 TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (item_id) REFERENCES items(item_id) ON DELETE CASCADE
                )
            """)
            
            # Tags table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tags (
                    tag_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    item_id TEXT NOT NULL,
                    tag TEXT NOT NULL,
                    FOREIGN KEY (item_id) REFERENCES items(item_id) ON DELETE CASCADE
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_items_box_id ON items(box_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_photos_item_id ON photos(item_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_tags_item_id ON tags(item_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_tags_tag ON tags(tag)")
    
    def _now_iso(self) -> str:
        """Return current UTC timestamp in ISO format."""
        return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    
    # Box operations
    
    def create_box(self, box_id: str, location: str = "", description: str = "") -> bool:
        """Create a new box.
        
        Args:
            box_id: Unique box identifier
            location: Physical location of the box
            description: Box description
            
        Returns:
            True if created, False if already exists
        """
        now = self._now_iso()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("""
                    INSERT INTO boxes (box_id, location, description, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (box_id, location, description, now, now))
                return True
            except sqlite3.IntegrityError:
                return False
    
    def get_box(self, box_id: str) -> Optional[Dict[str, Any]]:
        """Get box metadata.
        
        Args:
            box_id: Box identifier
            
        Returns:
            Box data dict or None if not found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM boxes WHERE box_id = ?", (box_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def list_boxes(self) -> List[str]:
        """List all box IDs.
        
        Returns:
            Sorted list of box IDs
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT box_id FROM boxes ORDER BY box_id")
            return [row[0] for row in cursor.fetchall()]
    
    def update_box(self, box_id: str, location: Optional[str] = None, 
                   description: Optional[str] = None) -> bool:
        """Update box metadata.
        
        Args:
            box_id: Box identifier
            location: New location (None = no change)
            description: New description (None = no change)
            
        Returns:
            True if updated, False if box not found
        """
        updates = []
        params = []
        
        if location is not None:
            updates.append("location = ?")
            params.append(location)
        
        if description is not None:
            updates.append("description = ?")
            params.append(description)
        
        if not updates:
            return True
        
        updates.append("updated_at = ?")
        params.append(self._now_iso())
        params.append(box_id)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                UPDATE boxes SET {', '.join(updates)}
                WHERE box_id = ?
            """, params)
            return cursor.rowcount > 0
    
    # Item operations
    
    def create_item(self, item_id: str, box_id: str, title: str = "", 
                   description: str = "") -> bool:
        """Create a new item.
        
        Args:
            item_id: Unique item identifier
            box_id: Parent box ID
            title: Item title
            description: Item description
            
        Returns:
            True if created, False if already exists
        """
        now = self._now_iso()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("""
                    INSERT INTO items (item_id, box_id, title, description, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (item_id, box_id, title, description, now, now))
                return True
            except sqlite3.IntegrityError:
                return False
    
    def get_item(self, item_id: str) -> Optional[Dict[str, Any]]:
        """Get item metadata including photos and tags.
        
        Args:
            item_id: Item identifier
            
        Returns:
            Item data dict with 'photos' and 'tags' lists, or None
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Get item
            cursor.execute("SELECT * FROM items WHERE item_id = ?", (item_id,))
            row = cursor.fetchone()
            if not row:
                return None
            
            item = dict(row)
            
            # Get photos
            cursor.execute("""
                SELECT photo_id, file_path, caption, hash_sha256, created_at
                FROM photos WHERE item_id = ? ORDER BY created_at
            """, (item_id,))
            item['photos'] = [dict(r) for r in cursor.fetchall()]
            
            # Get tags
            cursor.execute("""
                SELECT tag FROM tags WHERE item_id = ? ORDER BY tag
            """, (item_id,))
            item['tags'] = [r[0] for r in cursor.fetchall()]
            
            return item
    
    def list_items(self, box_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List items, optionally filtered by box.
        
        Args:
            box_id: Filter by box ID (None = all items)
            
        Returns:
            List of item dicts
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if box_id:
                cursor.execute("""
                    SELECT * FROM items WHERE box_id = ? ORDER BY created_at DESC
                """, (box_id,))
            else:
                cursor.execute("SELECT * FROM items ORDER BY created_at DESC")
            return [dict(row) for row in cursor.fetchall()]
    
    def update_item(self, item_id: str, title: Optional[str] = None,
                   description: Optional[str] = None, box_id: Optional[str] = None) -> bool:
        """Update item metadata.
        
        Args:
            item_id: Item identifier
            title: New title (None = no change)
            description: New description (None = no change)
            box_id: Move to different box (None = no change)
            
        Returns:
            True if updated, False if item not found
        """
        updates = []
        params = []
        
        if title is not None:
            updates.append("title = ?")
            params.append(title)
        
        if description is not None:
            updates.append("description = ?")
            params.append(description)
        
        if box_id is not None:
            updates.append("box_id = ?")
            params.append(box_id)
        
        if not updates:
            return True
        
        updates.append("updated_at = ?")
        params.append(self._now_iso())
        params.append(item_id)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                UPDATE items SET {', '.join(updates)}
                WHERE item_id = ?
            """, params)
            return cursor.rowcount > 0
    
    # Photo operations
    
    def add_photo(self, item_id: str, file_path: str, caption: str = "",
                 hash_sha256: str = "") -> int:
        """Add a photo to an item.
        
        Args:
            item_id: Parent item ID
            file_path: Relative path to photo file
            caption: Photo caption
            hash_sha256: SHA-256 hash of file
            
        Returns:
            Photo ID
        """
        now = self._now_iso()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO photos (item_id, file_path, caption, hash_sha256, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (item_id, file_path, caption, hash_sha256, now))
            return cursor.lastrowid
    
    def get_photos(self, item_id: str) -> List[Dict[str, Any]]:
        """Get all photos for an item.
        
        Args:
            item_id: Item identifier
            
        Returns:
            List of photo dicts
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM photos WHERE item_id = ? ORDER BY created_at
            """, (item_id,))
            return [dict(row) for row in cursor.fetchall()]
    
    # Tag operations
    
    def add_tag(self, item_id: str, tag: str) -> int:
        """Add a tag to an item.
        
        Args:
            item_id: Parent item ID
            tag: Tag string
            
        Returns:
            Tag ID
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO tags (item_id, tag) VALUES (?, ?)
            """, (item_id, tag))
            return cursor.lastrowid
    
    def remove_tag(self, item_id: str, tag: str) -> bool:
        """Remove a tag from an item.
        
        Args:
            item_id: Item identifier
            tag: Tag to remove
            
        Returns:
            True if tag was removed
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM tags WHERE item_id = ? AND tag = ?
            """, (item_id, tag))
            return cursor.rowcount > 0
    
    def get_tags(self, item_id: str) -> List[str]:
        """Get all tags for an item.
        
        Args:
            item_id: Item identifier
            
        Returns:
            List of tag strings
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT tag FROM tags WHERE item_id = ? ORDER BY tag
            """, (item_id,))
            return [row[0] for row in cursor.fetchall()]
    
    def search_by_tag(self, tag: str) -> List[Dict[str, Any]]:
        """Search items by tag.
        
        Args:
            tag: Tag to search for
            
        Returns:
            List of item dicts
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT DISTINCT i.* FROM items i
                JOIN tags t ON i.item_id = t.item_id
                WHERE t.tag = ?
                ORDER BY i.created_at DESC
            """, (tag,))
            return [dict(row) for row in cursor.fetchall()]
