#!/usr/bin/env python3
"""
Test script for priority engine.

Demonstrates:
- Embedding and caching
- Dimension learning with PCA
- PDV learning from user actions
- Task scoring and ranking
- Chat log analysis
"""

import asyncio
import json
from pathlib import Path

import numpy as np

from priority_engine import PriorityEngine
from chat_log_analyzer import ChatLogAnalyzer


async def test_basic_flow():
    """Test the complete flow"""
    print("=" * 60)
    print("Priority Engine Test")
    print("=" * 60)
    
    # Initialize engine
    print("\n1. Initializing priority engine...")
    engine = PriorityEngine(
        cache_dir=Path(".cache/test"),
        embedding_model="granite-embedding:278m",
        ollama_url="http://localhost:11434"
    )
    
    # Create test tasks
    tasks = [
        {
            "id": "task-1",
            "title": "Fix login bug",
            "description": "Users can't log in with OAuth. Critical security issue.",
        },
        {
            "id": "task-2", 
            "title": "Write documentation",
            "description": "Document the new API endpoints for the team.",
        },
        {
            "id": "task-3",
            "title": "Refactor database layer",
            "description": "Clean up the database access code to improve performance.",
        },
        {
            "id": "task-4",
            "title": "Plan sprint meeting",
            "description": "Organize agenda for next sprint planning session.",
        },
        {
            "id": "task-5",
            "title": "Review pull requests",
            "description": "Review pending PRs from the team, especially the auth changes.",
        },
    ]
    
    # Update engine with tasks
    print("\n2. Embedding tasks...")
    engine.update_from_tasks(tasks)
    
    print(f"   - Embedded {len(tasks)} tasks")
    print(f"   - Cache stats: {engine.embedding_model.cache.stats()}")
    
    # Get dimension info
    print("\n3. Learning dimensions with PCA...")
    dim_info = engine.get_dimension_info()
    print(f"   - Discovered {len(dim_info)} dimensions")
    for i, dim in enumerate(dim_info):
        print(f"   - Dimension {i}: variance={dim['variance_explained']:.3f}, coverage={dim['coverage']:.2f}")
    
    # Simulate user actions (promoting security tasks, demoting docs)
    print("\n4. Learning preferences from user actions...")
    kept_tasks = ["task-1", "task-5"]  # Security/review tasks
    removed_tasks = ["task-2", "task-4"]  # Docs/planning tasks
    
    engine.update_from_actions(kept_tasks, removed_tasks, weight=1.0)
    
    stats = engine.get_stats()
    print(f"   - PDV strength: {stats['pdv_strength']:.3f}")
    print(f"   - Action count: {stats['action_count']}")
    
    # Rank tasks
    print("\n5. Ranking tasks...")
    task_metadata = {
        "task-1": {"due_pressure": 0.9, "blocked": False, "graph_centrality": 0.8},
        "task-2": {"due_pressure": 0.3, "blocked": False, "graph_centrality": 0.2},
        "task-3": {"due_pressure": 0.5, "blocked": False, "graph_centrality": 0.6},
        "task-4": {"due_pressure": 0.4, "blocked": False, "graph_centrality": 0.3},
        "task-5": {"due_pressure": 0.7, "blocked": False, "graph_centrality": 0.7},
    }
    
    rankings = engine.rank_tasks(task_metadata)
    
    print("\n   Rankings:")
    for i, (task_id, score) in enumerate(rankings[:5], 1):
        task = next(t for t in tasks if t["id"] == task_id)
        print(f"   {i}. {task['title']:<30} (score: {score:.2f})")
    
    # Save state
    print("\n6. Saving state...")
    state_file = Path(".cache/test/engine_state.json")
    engine.save_state(state_file)
    print(f"   - Saved to {state_file}")
    
    # Test chat log analysis
    print("\n7. Testing chat log analysis...")
    chat_analyzer = ChatLogAnalyzer(engine.embedding_model)
    
    # Simulate chat messages
    chat_messages = [
        {
            "role": "user",
            "content": "I'm really worried about the login bug, it's affecting users",
            "timestamp": "2025-10-07T10:00:00",
        },
        {
            "role": "assistant",
            "content": "That sounds critical. Let's prioritize fixing the OAuth login issue.",
            "timestamp": "2025-10-07T10:01:00",
        },
        {
            "role": "user",
            "content": "Yeah, and I also need to review those auth PRs before end of day",
            "timestamp": "2025-10-07T10:02:00",
        },
        {
            "role": "assistant",
            "content": "Good idea. The PR reviews are important for the security fixes.",
            "timestamp": "2025-10-07T10:03:00",
        },
        {
            "role": "user",
            "content": "I'm exhausted though, the documentation can wait until tomorrow",
            "timestamp": "2025-10-07T10:04:00",
        },
    ]
    
    analysis = await chat_analyzer.analyze_logs(chat_messages, tasks)
    print(f"   - Analyzed {analysis['chunk_count']} chat chunks")
    print(f"   - Found task mentions: {list(analysis['task_mentions'].keys())}")
    print(f"   - Sentiment: {analysis['sentiment_summary']}")
    
    # Extract preference signals from chat
    positive, negative = await chat_analyzer.extract_preference_signals(
        analysis, engine.task_embeddings
    )
    print(f"   - Extracted {len(positive)} positive, {len(negative)} negative signals")
    
    # Update PDV with chat signals
    if positive or negative:
        engine.preference_model.update(positive, negative, weight=0.5)
        print(f"   - Updated PDV from chat (new strength: {np.linalg.norm(engine.preference_model.pdv):.3f})")
    
    # Re-rank with chat-informed PDV
    print("\n8. Re-ranking with chat-informed preferences...")
    new_rankings = engine.rank_tasks(task_metadata)
    
    print("\n   New rankings:")
    for i, (task_id, score) in enumerate(new_rankings[:5], 1):
        task = next(t for t in tasks if t["id"] == task_id)
        old_rank = next(j for j, (tid, _) in enumerate(rankings, 1) if tid == task_id)
        change = "↑" if i < old_rank else "↓" if i > old_rank else "="
        print(f"   {i}. {task['title']:<30} (score: {score:.2f}) {change}")
    
    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)
    
    # Print final stats
    final_stats = engine.get_stats()
    print("\nFinal Statistics:")
    for key, value in final_stats.items():
        print(f"  {key}: {value}")


async def test_dimension_labeling():
    """Test dimension labeling with vocabulary"""
    print("\n" + "=" * 60)
    print("Dimension Labeling Test")
    print("=" * 60)
    
    engine = PriorityEngine(cache_dir=Path(".cache/test"))
    
    # Create diverse tasks
    tasks = [
        {"id": "t1", "title": "Security audit", "description": "Review security vulnerabilities"},
        {"id": "t2", "title": "Performance optimization", "description": "Improve database query speed"},
        {"id": "t3", "title": "User research", "description": "Conduct user interviews"},
        {"id": "t4", "title": "Bug fixing", "description": "Fix critical bugs in production"},
        {"id": "t5", "title": "Team meeting", "description": "Weekly sync with the team"},
        {"id": "t6", "title": "Code review", "description": "Review code changes"},
        {"id": "t7", "title": "Documentation", "description": "Write technical documentation"},
        {"id": "t8", "title": "Infrastructure", "description": "Set up CI/CD pipeline"},
    ]
    
    engine.update_from_tasks(tasks)
    
    # Load simple vocabulary
    vocab_words = [
        "security", "performance", "user", "bug", "team", "code", 
        "documentation", "infrastructure", "critical", "important",
        "urgent", "research", "review", "meeting", "technical"
    ]
    
    print(f"\n1. Loading vocabulary ({len(vocab_words)} words)...")
    vocab_embeddings = engine.embedding_model.encode_batch(vocab_words)
    engine.dimension_learner.load_vocabulary(vocab_words, vocab_embeddings)
    
    print("\n2. Labeling dimensions...")
    labels = engine.dimension_learner.label_dimensions(top_k=3)
    
    dim_info = engine.get_dimension_info()
    print(f"\n3. Dimension analysis:")
    for i, dim in enumerate(dim_info):
        print(f"\n   Dimension {i}:")
        print(f"     Variance explained: {dim['variance_explained']:.3f}")
        print(f"     Coverage: {dim['coverage']:.2f}")
        print(f"     Labels: {', '.join(dim['labels'])}")


if __name__ == "__main__":
    print("Starting priority engine tests...")
    print("Make sure Ollama is running with granite-embedding:278m model!")
    print()
    
    # Run tests
    asyncio.run(test_basic_flow())
    print("\n\n")
    asyncio.run(test_dimension_labeling())
