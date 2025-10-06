#!/usr/bin/env python3
"""
Task Manager - A simple CLI for managing tasks with prioritization and dependencies.
Integrates with PriorityEngine for intelligent, learned prioritization.
"""

import argparse
import asyncio
import json
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from priority_engine import PriorityEngine
from chat_log_analyzer import ChatLogAnalyzer


def init_database(db_path: str) -> None:
    """Initialize SQLite database with required tables."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    
    # Tasks table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT,
            status TEXT DEFAULT 'active',
            properties TEXT,  -- JSON blob for flexible properties
            due_date TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Dependencies table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS dependencies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id INTEGER NOT NULL,
            depends_on_task_id INTEGER NOT NULL,
            FOREIGN KEY (task_id) REFERENCES tasks (id) ON DELETE CASCADE,
            FOREIGN KEY (depends_on_task_id) REFERENCES tasks (id) ON DELETE CASCADE,
            UNIQUE(task_id, depends_on_task_id)
        )
    """)
    
    # Weighting profiles table (legacy - kept for backwards compatibility)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS weighting_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            weights TEXT NOT NULL,  -- JSON blob of property -> weight mappings
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # User actions table - for PDV learning
    conn.execute("""
        CREATE TABLE IF NOT EXISTS user_actions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id INTEGER NOT NULL,
            action TEXT NOT NULL,  -- 'complete', 'snooze', 'promote', 'demote'
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (task_id) REFERENCES tasks (id) ON DELETE CASCADE
        )
    """)
    
    conn.commit()
    conn.close()


def add_task(db_path: str, title: str, description: str = "", properties: Dict = None, due_date: str = None) -> int:
    """Add a new task to the database."""
    conn = sqlite3.connect(db_path)
    
    properties_json = json.dumps(properties or {})
    cursor = conn.execute(
        "INSERT INTO tasks (title, description, properties, due_date) VALUES (?, ?, ?, ?)",
        (title, description, properties_json, due_date)
    )
    task_id = cursor.lastrowid
    
    conn.commit()
    conn.close()
    
    return task_id


def add_dependency(db_path: str, task_id: int, depends_on_task_id: int) -> None:
    """Add a dependency relationship between tasks."""
    conn = sqlite3.connect(db_path)
    
    try:
        conn.execute(
            "INSERT INTO dependencies (task_id, depends_on_task_id) VALUES (?, ?)",
            (task_id, depends_on_task_id)
        )
        conn.commit()
    except sqlite3.IntegrityError:
        print(f"Dependency already exists or invalid task IDs")
    finally:
        conn.close()


def calculate_task_score(properties: Dict, weights: Dict) -> float:
    """Calculate priority score for a task based on properties and weights."""
    score = 0.0
    
    for prop, weight in weights.items():
        if prop in properties:
            value = properties[prop]
            
            # Handle different value types
            if isinstance(value, bool):
                score += weight if value else 0
            elif isinstance(value, (int, float)):
                score += weight * value
            elif isinstance(value, str):
                # Map string values to numeric scores
                string_scores = {
                    'low': 1, 'medium': 5, 'high': 10,
                    'active': 10, 'completed': 0, 'cancelled': 0, 'on-hold': 2
                }
                score += weight * string_scores.get(value, 0)
    
    return score


def get_weighting_profile(db_path: str, profile_name: str) -> Dict:
    """Get weighting profile by name."""
    conn = sqlite3.connect(db_path)
    
    result = conn.execute(
        "SELECT weights FROM weighting_profiles WHERE name = ?",
        (profile_name,)
    ).fetchone()
    
    conn.close()
    
    if result:
        return json.loads(result[0])
    return {}


def save_weighting_profile(db_path: str, name: str, weights: Dict) -> None:
    """Save or update a weighting profile."""
    conn = sqlite3.connect(db_path)
    
    weights_json = json.dumps(weights)
    conn.execute(
        "INSERT OR REPLACE INTO weighting_profiles (name, weights) VALUES (?, ?)",
        (name, weights_json)
    )
    
    conn.commit()
    conn.close()


def get_available_tasks(db_path: str, profile_name: str = None) -> List[Tuple]:
    """Get tasks that have no unmet dependencies, optionally scored by profile."""
    conn = sqlite3.connect(db_path)
    
    # Get tasks without unmet dependencies
    query = """
        SELECT t.id, t.title, t.description, t.properties, t.status
        FROM tasks t
        WHERE t.status = 'active'
        AND NOT EXISTS (
            SELECT 1 FROM dependencies d
            JOIN tasks dt ON d.depends_on_task_id = dt.id
            WHERE d.task_id = t.id AND dt.status = 'active'
        )
        ORDER BY t.created_at
    """
    
    tasks = conn.execute(query).fetchall()
    conn.close()
    
    if profile_name:
        weights = get_weighting_profile(db_path, profile_name)
        if weights:
            # Score and sort tasks
            scored_tasks = []
            for task in tasks:
                task_id, title, description, properties_json, status = task
                properties = json.loads(properties_json or '{}')
                score = calculate_task_score(properties, weights)
                scored_tasks.append((score, task))
            
            # Sort by score (highest first)
            scored_tasks.sort(key=lambda x: x[0], reverse=True)
            return [task for score, task in scored_tasks]
    
    return tasks


def list_tasks(db_path: str, status: str = "active") -> None:
    """List all tasks with their status."""
    conn = sqlite3.connect(db_path)
    
    query = "SELECT id, title, description, status FROM tasks"
    if status:
        query += f" WHERE status = '{status}'"
    query += " ORDER BY created_at"
    
    tasks = conn.execute(query).fetchall()
    conn.close()
    
    if not tasks:
        print(f"No tasks found with status '{status}'")
        return
    
    for task_id, title, description, task_status in tasks:
        print(f"[{task_id}] {title} ({task_status})")
        if description:
            print(f"    {description}")


def get_all_tasks(db_path: str, status: str = "active") -> List[Dict]:
    """Get all tasks as dictionaries for priority engine."""
    conn = sqlite3.connect(db_path)
    
    query = """
        SELECT id, title, description, properties, status, due_date, created_at
        FROM tasks
    """
    if status:
        query += f" WHERE status = '{status}'"
    
    rows = conn.execute(query).fetchall()
    conn.close()
    
    tasks = []
    for row in rows:
        task_id, title, description, properties_json, status, due_date, created_at = row
        properties = json.loads(properties_json or '{}')
        
        tasks.append({
            "id": str(task_id),
            "title": title,
            "description": description or "",
            "properties": properties,
            "status": status,
            "due_date": due_date,
            "created_at": created_at,
        })
    
    return tasks


def get_task_dependencies(db_path: str, task_id: int) -> List[int]:
    """Get tasks that this task depends on."""
    conn = sqlite3.connect(db_path)
    
    deps = conn.execute(
        "SELECT depends_on_task_id FROM dependencies WHERE task_id = ?",
        (task_id,)
    ).fetchall()
    
    conn.close()
    return [dep[0] for dep in deps]


def get_recent_actions(db_path: str, hours: int = 24) -> List[Tuple[int, str, str]]:
    """Get recent user actions for PDV learning."""
    conn = sqlite3.connect(db_path)
    
    # For now, get all actions (timestamp comparison is tricky with SQLite)
    actions = conn.execute(
        """
        SELECT task_id, action, timestamp
        FROM user_actions
        ORDER BY timestamp DESC
        LIMIT 100
        """
    ).fetchall()
    
    conn.close()
    return actions


class IntelligentTaskManager:
    """Task manager with learned prioritization via PriorityEngine."""
    
    def __init__(self, db_path: str = "tasks.db"):
        self.db_path = db_path
        init_database(db_path)
        
        # Initialize priority engine
        cache_dir = Path(".cache") / "task_manager"
        self.engine = PriorityEngine(
            cache_dir=cache_dir,
            embedding_model="granite-embedding:278m",
            ollama_url="http://localhost:11434"
        )
        
        # Initialize chat analyzer
        self.chat_analyzer = ChatLogAnalyzer(self.engine.embedding_model)
        
        # Load existing state if available
        state_file = cache_dir / "engine_state.json"
        if state_file.exists():
            try:
                self.engine.load_state(state_file)
            except Exception as e:
                print(f"Warning: Could not load engine state: {e}")
    
    def update_embeddings(self):
        """Update task embeddings in the priority engine."""
        tasks = get_all_tasks(self.db_path, status="active")
        if tasks:
            self.engine.update_from_tasks(tasks)
    
    def learn_from_recent_actions(self, hours: int = 24):
        """Learn from recent user actions to update PDV."""
        actions = get_recent_actions(self.db_path, hours)
        
        # Get ALL tasks (including completed) for embedding
        all_tasks_for_learning = get_all_tasks(self.db_path, status=None)
        task_by_id = {t["id"]: t for t in all_tasks_for_learning}
        
        # Embed any tasks we haven't seen before
        for task in all_tasks_for_learning:
            if task["id"] not in self.engine.task_embeddings:
                text = f"{task['title']} {task['description']}"
                emb = self.engine.embedding_model.encode_batch([text])[0]
                self.engine.task_embeddings[task["id"]] = emb
        
        # Separate actions
        kept_ids = []
        removed_ids = []
        
        for task_id, action, timestamp in actions:
            task_id_str = str(task_id)
            if task_id_str in self.engine.task_embeddings:
                if action in ('complete', 'promote'):
                    kept_ids.append(task_id_str)
                elif action in ('snooze', 'demote'):
                    removed_ids.append(task_id_str)
        
        # Update PDV
        if kept_ids or removed_ids:
            self.engine.update_from_actions(kept_ids, removed_ids, weight=1.0)
    
    def get_intelligent_priorities(self) -> List[Dict]:
        """Get task priorities using learned dimensions and PDV."""
        # Update embeddings
        self.update_embeddings()
        
        # Learn from recent actions
        self.learn_from_recent_actions()
        
        # Build metadata for scoring
        tasks = get_all_tasks(self.db_path, status="active")
        task_metadata = {}
        
        for task in tasks:
            task_id = task["id"]
            
            # Calculate due pressure
            due_pressure = 0.0
            if task.get("due_date"):
                try:
                    due_date = datetime.fromisoformat(task["due_date"])
                    time_until_due = (due_date - datetime.now()).total_seconds()
                    # Exponential urgency as deadline approaches (7 days baseline)
                    due_pressure = max(0, 1 - (time_until_due / (7 * 24 * 3600)))
                except:
                    pass
            
            # Check if blocked
            deps = get_task_dependencies(self.db_path, int(task_id))
            blocked = len(deps) > 0
            
            # Graph centrality (simplified - could be enhanced)
            centrality = 0.5
            
            task_metadata[task_id] = {
                "due_pressure": due_pressure,
                "blocked": blocked,
                "graph_centrality": centrality,
            }
        
        # Get rankings
        rankings = self.engine.rank_tasks(task_metadata)
        
        # Build result with full task info
        result = []
        for task_id, score in rankings:
            task = next((t for t in tasks if t["id"] == task_id), None)
            if task:
                result.append({
                    "task": task,
                    "score": score,
                    "metadata": task_metadata[task_id],
                })
        
        return result
    
    async def analyze_chat_logs(self, messages: List[Dict]) -> Dict:
        """Analyze chat logs for passive feedback."""
        tasks = get_all_tasks(self.db_path, status="active")
        
        # Analyze
        analysis = await self.chat_analyzer.analyze_logs(
            messages,
            tasks,
            time_window=timedelta(hours=24)
        )
        
        # Extract preference signals
        positive, negative = await self.chat_analyzer.extract_preference_signals(
            analysis,
            self.engine.task_embeddings
        )
        
        # Update PDV with reduced weight (passive signal)
        if positive or negative:
            self.engine.preference_model.update(
                positive,
                negative,
                weight=0.3
            )
        
        return analysis
    
    def get_dimension_insights(self) -> List[Dict]:
        """Get human-readable dimension insights."""
        return self.engine.get_dimension_info()
    
    def get_stats(self) -> Dict:
        """Get engine statistics."""
        return self.engine.get_stats()
    
    def save_state(self):
        """Save priority engine state."""
        state_file = Path(".cache") / "task_manager" / "engine_state.json"
        self.engine.save_state(state_file)


def complete_task(db_path: str, task_id: int) -> None:
    """Mark a task as completed."""
    conn = sqlite3.connect(db_path)
    
    # Update task status
    conn.execute(
        "UPDATE tasks SET status = 'completed', updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        (task_id,)
    )
    
    # Record action for PDV learning
    conn.execute(
        "INSERT INTO user_actions (task_id, action) VALUES (?, 'complete')",
        (task_id,)
    )
    
    conn.commit()
    conn.close()


def snooze_task(db_path: str, task_id: int, until: str = None) -> None:
    """Snooze a task (lower priority signal)."""
    conn = sqlite3.connect(db_path)
    
    # Record snooze action for PDV learning
    conn.execute(
        "INSERT INTO user_actions (task_id, action) VALUES (?, 'snooze')",
        (task_id,)
    )
    
    conn.commit()
    conn.close()


def record_action(db_path: str, task_id: int, action: str) -> None:
    """Record a user action for PDV learning."""
    conn = sqlite3.connect(db_path)
    
    conn.execute(
        "INSERT INTO user_actions (task_id, action) VALUES (?, ?)",
        (task_id, action)
    )
    
    conn.commit()
    conn.close()


def main(argv: List[str] = None) -> int:
    """Main entry point for task manager CLI."""
    if argv is None:
        argv = sys.argv[1:]
    
    parser = argparse.ArgumentParser(description="Task Manager CLI")
    parser.add_argument("--db", default="tasks.db", help="SQLite database path")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Init command
    init_parser = subparsers.add_parser("init", help="Initialize database")
    
    # Add command
    add_parser = subparsers.add_parser("add", help="Add a new task")
    add_parser.add_argument("title", help="Task title")
    add_parser.add_argument("--description", default="", help="Task description")
    add_parser.add_argument("--properties", help="JSON string of task properties")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List tasks")
    list_parser.add_argument("--status", default="active", help="Filter by status")
    
    # Available command
    avail_parser = subparsers.add_parser("available", help="Show available tasks (no unmet dependencies)")
    avail_parser.add_argument("--profile", help="Weighting profile for prioritization")
    
    # Intelligent priorities command
    smart_parser = subparsers.add_parser("smart", help="Show tasks with intelligent learned priorities")
    smart_parser.add_argument("--top", type=int, default=10, help="Number of top tasks to show")
    
    # Complete command
    complete_parser = subparsers.add_parser("complete", help="Mark task as completed")
    complete_parser.add_argument("task_id", type=int, help="Task ID to complete")
    
    # Dependency command
    dep_parser = subparsers.add_parser("depend", help="Add dependency")
    dep_parser.add_argument("task_id", type=int, help="Task that depends on something")
    dep_parser.add_argument("depends_on", type=int, help="Task that must be completed first")
    
    # Profile commands
    profile_parser = subparsers.add_parser("profile", help="Manage weighting profiles")
    profile_subparsers = profile_parser.add_subparsers(dest="profile_command")
    
    save_profile_parser = profile_subparsers.add_parser("save", help="Save weighting profile")
    save_profile_parser.add_argument("name", help="Profile name")
    save_profile_parser.add_argument("--weights", required=True, help="JSON string of weights")
    
    list_profiles_parser = profile_subparsers.add_parser("list", help="List saved profiles")
    
    args = parser.parse_args(argv)
    
    if not args.command:
        parser.print_help()
        return 2
    
    try:
        if args.command == "init":
            init_database(args.db)
            print(f"Database initialized: {args.db}")
            
        elif args.command == "add":
            properties = {}
            if args.properties:
                properties = json.loads(args.properties)
            
            task_id = add_task(args.db, args.title, args.description, properties)
            print(f"Added task {task_id}: {args.title}")
            
        elif args.command == "list":
            list_tasks(args.db, args.status)
            
        elif args.command == "available":
            tasks = get_available_tasks(args.db, args.profile)
            if not tasks:
                print("No available tasks")
            else:
                print("Available tasks (no unmet dependencies):")
                for task_id, title, description, properties, status in tasks:
                    print(f"[{task_id}] {title}")
                    if description:
                        print(f"    {description}")
        
        elif args.command == "smart":
            # Use intelligent task manager
            manager = IntelligentTaskManager(args.db)
            priorities = manager.get_intelligent_priorities()
            
            if not priorities:
                print("No active tasks")
            else:
                print("Intelligent Priorities (learned from your behavior):")
                print()
                
                for i, item in enumerate(priorities[:args.top], 1):
                    task = item["task"]
                    score = item["score"]
                    metadata = item["metadata"]
                    
                    print(f"{i}. [{task['id']}] {task['title']}")
                    print(f"   Score: {score:.2f}")
                    
                    if task.get("description"):
                        print(f"   {task['description'][:80]}...")
                    
                    if metadata.get("due_pressure") > 0:
                        print(f"   Due pressure: {metadata['due_pressure']:.2f}")
                    
                    if metadata.get("blocked"):
                        print(f"   ⚠️  BLOCKED by dependencies")
                    
                    print()
                
                # Show stats
                stats = manager.get_stats()
                print(f"Engine stats: {stats['pdv_strength']:.2f} PDV strength, "
                      f"{stats['action_count']} actions learned, "
                      f"{stats['cache_stats']['hit_rate']:.0%} cache hit rate")
                
                # Save state for next run
                manager.save_state()
            
            return 0
            
        elif args.command == "complete":
            complete_task(args.db, args.task_id)
            print(f"Task {args.task_id} marked as completed")
            
        elif args.command == "depend":
            add_dependency(args.db, args.task_id, args.depends_on)
            print(f"Added dependency: task {args.task_id} depends on task {args.depends_on}")
            
        elif args.command == "profile":
            if args.profile_command == "save":
                weights = json.loads(args.weights)
                save_weighting_profile(args.db, args.name, weights)
                print(f"Saved weighting profile: {args.name}")
                
            elif args.profile_command == "list":
                conn = sqlite3.connect(args.db)
                profiles = conn.execute("SELECT name FROM weighting_profiles ORDER BY name").fetchall()
                conn.close()
                
                if profiles:
                    print("Available weighting profiles:")
                    for (name,) in profiles:
                        print(f"  {name}")
                else:
                    print("No weighting profiles found")
            
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())