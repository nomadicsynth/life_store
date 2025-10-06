"""
Example integration of priority engine with task manager.

Shows how to use the priority engine to intelligently rank tasks.
"""

import asyncio
from pathlib import Path
from datetime import datetime, timedelta

from priority_engine import PriorityEngine
from chat_log_analyzer import ChatLogAnalyzer


class IntelligentTaskManager:
    """Task manager with learned prioritization"""

    def __init__(self):
        self.engine = PriorityEngine(
            cache_dir=Path(".cache/task_manager"),
            embedding_model="granite-embedding:278m"
        )
        self.chat_analyzer = ChatLogAnalyzer(self.engine.embedding_model)
        self.tasks = {}

    def add_task(self, task_id: str, title: str, description: str = "", **metadata):
        """Add a new task"""
        self.tasks[task_id] = {
            "id": task_id,
            "title": title,
            "description": description,
            "created": datetime.now(),
            "metadata": metadata,
        }
        
        # Update embeddings
        self.engine.update_from_tasks([self.tasks[task_id]])

    def complete_task(self, task_id: str):
        """Mark task as complete and learn from it"""
        if task_id in self.tasks:
            # This is a positive signal - task was important enough to complete
            self.engine.update_from_actions(
                kept_task_ids=[task_id],
                removed_task_ids=[],
                weight=1.0
            )
            self.tasks[task_id]["completed"] = datetime.now()

    def snooze_task(self, task_id: str, until: datetime = None):
        """Snooze a task and learn that it's lower priority"""
        if task_id in self.tasks:
            # Negative signal - task was deprioritized
            self.engine.update_from_actions(
                kept_task_ids=[],
                removed_task_ids=[task_id],
                weight=0.5
            )
            self.tasks[task_id]["snoozed_until"] = until or datetime.now() + timedelta(hours=4)

    def get_priorities(self, context: dict = None):
        """Get current task priorities"""
        # Build metadata for scoring
        task_metadata = {}
        
        for task_id, task in self.tasks.items():
            # Skip completed tasks
            if task.get("completed"):
                continue
            
            # Skip snoozed tasks
            if task.get("snoozed_until") and task["snoozed_until"] > datetime.now():
                continue
            
            # Compute due pressure
            due_pressure = 0.0
            if "due_date" in task["metadata"]:
                due_date = task["metadata"]["due_date"]
                if isinstance(due_date, str):
                    due_date = datetime.fromisoformat(due_date)
                time_until_due = (due_date - datetime.now()).total_seconds()
                # Exponential urgency as deadline approaches
                due_pressure = max(0, 1 - (time_until_due / (7 * 24 * 3600)))
            
            task_metadata[task_id] = {
                "due_pressure": due_pressure,
                "blocked": task["metadata"].get("blocked", False),
                "graph_centrality": task["metadata"].get("centrality", 0.5),
            }
        
        # Get rankings
        rankings = self.engine.rank_tasks(task_metadata)
        
        return [
            {
                "task_id": task_id,
                "score": score,
                "task": self.tasks[task_id],
            }
            for task_id, score in rankings
        ]

    async def analyze_recent_chats(self, messages: list):
        """Analyze chat logs to update priorities"""
        # Analyze logs
        analysis = await self.chat_analyzer.analyze_logs(
            messages,
            list(self.tasks.values()),
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
                weight=0.3  # Lower weight for passive vs. explicit actions
            )
        
        return analysis

    def get_dimension_insights(self):
        """Get human-readable dimension insights"""
        return self.engine.get_dimension_info()

    def save(self, filepath: Path):
        """Save state to disk"""
        self.engine.save_state(filepath / "engine_state.json")
        # Could also save tasks here

    def load(self, filepath: Path):
        """Load state from disk"""
        self.engine.load_state(filepath / "engine_state.json")


async def demo():
    """Demo of intelligent task manager"""
    print("Intelligent Task Manager Demo")
    print("=" * 60)
    
    manager = IntelligentTaskManager()
    
    # Add some tasks
    print("\n1. Adding tasks...")
    manager.add_task("t1", "Fix critical security bug", 
                    "OAuth login vulnerability",
                    due_date=(datetime.now() + timedelta(days=1)).isoformat())
    
    manager.add_task("t2", "Write quarterly report",
                    "Q4 performance summary",
                    due_date=(datetime.now() + timedelta(days=7)).isoformat())
    
    manager.add_task("t3", "Review code changes",
                    "PR #123 - authentication refactor",
                    due_date=(datetime.now() + timedelta(days=2)).isoformat())
    
    manager.add_task("t4", "Plan team offsite",
                    "Organize Q1 planning session")
    
    manager.add_task("t5", "Update documentation",
                    "API reference for new endpoints")
    
    # Get initial priorities
    print("\n2. Initial priorities:")
    for i, item in enumerate(manager.get_priorities(), 1):
        print(f"   {i}. {item['task']['title']:<35} (score: {item['score']:.2f})")
    
    # Simulate user actions
    print("\n3. User completes security bug and snoozes documentation...")
    manager.complete_task("t1")
    manager.snooze_task("t5")
    
    # Get updated priorities
    print("\n4. Updated priorities (learned from actions):")
    for i, item in enumerate(manager.get_priorities(), 1):
        print(f"   {i}. {item['task']['title']:<35} (score: {item['score']:.2f})")
    
    # Simulate chat conversation
    print("\n5. Analyzing chat logs...")
    chat_messages = [
        {
            "role": "user",
            "content": "I'm really stressed about the code review deadline",
            "timestamp": datetime.now().isoformat(),
        },
        {
            "role": "assistant",
            "content": "Let's prioritize the PR review then",
            "timestamp": datetime.now().isoformat(),
        },
    ]
    
    analysis = await manager.analyze_recent_chats(chat_messages)
    print(f"   - Analyzed {analysis['chunk_count']} chat chunks")
    print(f"   - Sentiment: {analysis['sentiment_summary'].get('average', 0):.2f}")
    
    # Final priorities with chat influence
    print("\n6. Final priorities (with chat influence):")
    for i, item in enumerate(manager.get_priorities(), 1):
        print(f"   {i}. {item['task']['title']:<35} (score: {item['score']:.2f})")
    
    # Show dimension insights
    print("\n7. Discovered task dimensions:")
    for dim in manager.get_dimension_insights():
        print(f"   - Dimension {dim['index']}: variance={dim['variance_explained']:.3f}, "
              f"coverage={dim['coverage']:.2f}")


if __name__ == "__main__":
    asyncio.run(demo())
