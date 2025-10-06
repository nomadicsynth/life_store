#!/usr/bin/env python3
"""
Task Manager Web UI - Gradio interface for task management.
Includes intelligent prioritization with PriorityEngine.
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import gradio as gr

# Import functions from task_manager
from task_manager import (
    init_database, add_task, list_tasks, get_available_tasks,
    complete_task, add_dependency, save_weighting_profile,
    get_weighting_profile, IntelligentTaskManager, snooze_task,
    record_action
)


class TaskManagerUI:
    def __init__(self, db_path: str = "tasks.db"):
        self.db_path = db_path
        init_database(db_path)
        
        # Initialize intelligent task manager
        self.intelligent_manager = IntelligentTaskManager(db_path)
        
    def add_task_ui(self, title: str, description: str, properties_json: str, due_date: str = ""):
        """Add a task via the UI."""
        if not title.strip():
            return "Error: Task title is required", ""
        
        try:
            properties = json.loads(properties_json) if properties_json.strip() else {}
        except json.JSONDecodeError:
            return "Error: Invalid JSON in properties", ""
        
        try:
            due = due_date.strip() if due_date.strip() else None
            task_id = add_task(self.db_path, title.strip(), description.strip(), properties, due)
            
            # Update embeddings
            self.intelligent_manager.update_embeddings()
            
            return f"Added task {task_id}: {title}", ""
        except Exception as e:
            return f"Error: {e}", ""
    
    def get_available_tasks_ui(self, profile_name: str = ""):
        """Get available tasks for display."""
        try:
            profile = profile_name.strip() if profile_name.strip() else None
            tasks = get_available_tasks(self.db_path, profile)
            
            if not tasks:
                return "No available tasks"
            
            result = "Available tasks:\n\n"
            for task_id, title, description, properties_json, status in tasks:
                result += f"[{task_id}] {title}\n"
                if description:
                    result += f"    {description}\n"
                
                # Show properties if they exist
                if properties_json:
                    try:
                        props = json.loads(properties_json)
                        if props:
                            result += f"    Properties: {json.dumps(props, indent=2)}\n"
                    except:
                        pass
                result += "\n"
            
            return result
        except Exception as e:
            return f"Error: {e}"
    
    def complete_task_ui(self, task_id_str: str):
        """Complete a task via the UI."""
        try:
            task_id = int(task_id_str.strip())
            complete_task(self.db_path, task_id)
            
            # Learn from this action
            self.intelligent_manager.learn_from_recent_actions()
            
            return f"Task {task_id} marked as completed"
        except ValueError:
            return "Error: Task ID must be a number"
        except Exception as e:
            return f"Error: {e}"
    
    def snooze_task_ui(self, task_id_str: str):
        """Snooze a task via the UI."""
        try:
            task_id = int(task_id_str.strip())
            snooze_task(self.db_path, task_id)
            
            # Learn from this action
            self.intelligent_manager.learn_from_recent_actions()
            
            return f"Task {task_id} snoozed"
        except ValueError:
            return "Error: Task ID must be a number"
        except Exception as e:
            return f"Error: {e}"
    
    def add_dependency_ui(self, task_id_str: str, depends_on_str: str):
        """Add a dependency via the UI."""
        try:
            task_id = int(task_id_str.strip())
            depends_on = int(depends_on_str.strip())
            add_dependency(self.db_path, task_id, depends_on)
            return f"Added dependency: task {task_id} depends on task {depends_on}"
        except ValueError:
            return "Error: Task IDs must be numbers"
        except Exception as e:
            return f"Error: {e}"
    
    def save_profile_ui(self, profile_name: str, weights_json: str):
        """Save a weighting profile via the UI."""
        if not profile_name.strip():
            return "Error: Profile name is required"
            
        try:
            weights = json.loads(weights_json) if weights_json.strip() else {}
            save_weighting_profile(self.db_path, profile_name.strip(), weights)
            return f"Saved weighting profile: {profile_name}"
        except json.JSONDecodeError:
            return "Error: Invalid JSON in weights"
        except Exception as e:
            return f"Error: {e}"
    
    def get_intelligent_priorities_ui(self, top_n: int = 10):
        """Get intelligent priorities for display."""
        try:
            priorities = self.intelligent_manager.get_intelligent_priorities()
            
            if not priorities:
                return "No active tasks"
            
            result = "🧠 Intelligent Priorities (Learned from Your Behavior)\n"
            result += "=" * 60 + "\n\n"
            
            for i, item in enumerate(priorities[:top_n], 1):
                task = item["task"]
                score = item["score"]
                metadata = item["metadata"]
                
                result += f"{i}. [{task['id']}] {task['title']}\n"
                result += f"   Score: {score:.2f}"
                
                if metadata.get("due_pressure") > 0:
                    result += f" | Due pressure: {metadata['due_pressure']:.1%}"
                
                if metadata.get("blocked"):
                    result += " | ⚠️ BLOCKED"
                
                result += "\n"
                
                if task.get("description"):
                    desc = task["description"][:100]
                    result += f"   {desc}{'...' if len(task['description']) > 100 else ''}\n"
                
                result += "\n"
            
            # Add stats
            stats = self.intelligent_manager.get_stats()
            result += "\n" + "-" * 60 + "\n"
            result += f"PDV Strength: {stats['pdv_strength']:.2f} | "
            result += f"Actions Learned: {stats['action_count']} | "
            result += f"Cache Hit Rate: {stats['cache_stats']['hit_rate']:.0%}\n"
            
            return result
        except Exception as e:
            return f"Error: {e}"
    
    def get_dimension_insights_ui(self):
        """Get dimension insights for display."""
        try:
            dimensions = self.intelligent_manager.get_dimension_insights()
            
            if not dimensions:
                return "No dimensions learned yet. Add more tasks!"
            
            result = "📊 Discovered Task Dimensions\n"
            result += "=" * 60 + "\n\n"
            result += "These are the latent factors automatically discovered\n"
            result += "in your task space using PCA/eigendecomposition:\n\n"
            
            for dim in dimensions[:8]:  # Show top 8
                result += f"Dimension {dim['index']}:\n"
                result += f"  Variance Explained: {dim['variance_explained']:.1%}\n"
                result += f"  Coverage: {dim['coverage']:.2f}\n"
                
                if dim.get('labels'):
                    result += f"  Labels: {', '.join(dim['labels'])}\n"
                
                result += "\n"
            
            return result
        except Exception as e:
            return f"Error: {e}"
    
    async def analyze_chat_logs_ui(self, chat_json: str):
        """Analyze chat logs for passive feedback."""
        if not chat_json.strip():
            return "Error: No chat logs provided"
        
        try:
            messages = json.loads(chat_json)
            
            if not isinstance(messages, list):
                return "Error: Chat logs must be a JSON array"
            
            analysis = await self.intelligent_manager.analyze_chat_logs(messages)
            
            result = "💬 Chat Log Analysis\n"
            result += "=" * 60 + "\n\n"
            result += f"Analyzed {analysis['chunk_count']} conversation chunks\n\n"
            
            if analysis.get('task_mentions'):
                result += "Task Mentions:\n"
                for task_id, mentions in analysis['task_mentions'].items():
                    result += f"  Task {task_id}: {len(mentions)} mentions\n"
                    for mention in mentions[:2]:  # Show first 2
                        result += f"    - Sentiment: {mention['sentiment']:.2f}\n"
                result += "\n"
            
            if analysis.get('sentiment_summary'):
                sent = analysis['sentiment_summary']
                result += f"Overall Sentiment: {sent.get('average', 0):.2f}\n"
                result += f"Recent Trend: {sent.get('trend', 'unknown')}\n\n"
            
            result += "✓ PDV updated with passive signals from conversation\n"
            
            return result
        except json.JSONDecodeError:
            return "Error: Invalid JSON in chat logs"
        except Exception as e:
            return f"Error: {e}"


def create_app(db_path: str) -> gr.Blocks:
    """Create the Gradio application."""
    ui = TaskManagerUI(db_path)
    
    with gr.Blocks(title="Intelligent Task Manager") as app:
        gr.Markdown("# 🧠 Intelligent Task Manager")
        gr.Markdown("Task management with **learned prioritization** - no predefined properties needed!")
        
        with gr.Tab("➕ Add Task"):
            with gr.Row():
                with gr.Column():
                    task_title = gr.Textbox(label="Task Title", placeholder="Enter task title...")
                    task_description = gr.Textbox(
                        label="Description", 
                        placeholder="Optional description...",
                        lines=3
                    )
                    task_due_date = gr.Textbox(
                        label="Due Date (ISO format)",
                        placeholder="2025-10-15T18:00:00 (optional)"
                    )
                    task_properties = gr.Textbox(
                        label="Properties (JSON, optional)",
                        placeholder='{"energy_required": "medium", "health_related": true}',
                        lines=3
                    )
                    add_btn = gr.Button("Add Task", variant="primary")
                
                with gr.Column():
                    add_result = gr.Textbox(label="Result", interactive=False)
            
            add_btn.click(
                ui.add_task_ui,
                inputs=[task_title, task_description, task_properties, task_due_date],
                outputs=[add_result, task_title]
            )
        
        with gr.Tab("🧠 Intelligent Priorities"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("""
                    ### How It Works
                    
                    This uses **eigendecomposition** to discover latent factors in your tasks,
                    and learns your preferences from your actions (complete/snooze).
                    
                    - 📊 Discovers task dimensions automatically
                    - 🎯 Learns what you care about (PDV)
                    - 🔄 Adapts to due dates and context
                    - 💬 Can learn from chat logs!
                    """)
                    
                    top_n_slider = gr.Slider(
                        minimum=5,
                        maximum=20,
                        value=10,
                        step=1,
                        label="Number of tasks to show"
                    )
                    refresh_smart_btn = gr.Button("Get Smart Priorities", variant="primary")
                
                with gr.Column():
                    smart_priorities = gr.Textbox(
                        label="Intelligent Priorities",
                        lines=20,
                        interactive=False
                    )
            
            refresh_smart_btn.click(
                ui.get_intelligent_priorities_ui,
                inputs=[top_n_slider],
                outputs=[smart_priorities]
            )
        
        with gr.Tab("📊 Dimension Insights"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("""
                    ### Discovered Dimensions
                    
                    These are the **latent factors** automatically discovered in your task space.
                    No predefined categories - it learns from your actual tasks!
                    """)
                    
                    refresh_dim_btn = gr.Button("Refresh Dimensions", variant="primary")
                
                with gr.Column():
                    dimensions_display = gr.Textbox(
                        label="Dimension Analysis",
                        lines=20,
                        interactive=False
                    )
            
            refresh_dim_btn.click(
                ui.get_dimension_insights_ui,
                outputs=[dimensions_display]
            )
        
        with gr.Tab("💬 Chat Log Analysis"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("""
                    ### Journaling Sneak-Attack
                    
                    Paste your AI chat logs here. The system will:
                    - Detect task mentions
                    - Analyze sentiment
                    - Extract preference signals
                    - Update PDV passively
                    
                    **Format**: JSON array of messages with `role`, `content`, `timestamp`
                    """)
                    
                    chat_logs_input = gr.Textbox(
                        label="Chat Logs (JSON)",
                        placeholder='[{"role": "user", "content": "...", "timestamp": "..."}]',
                        lines=10
                    )
                    analyze_chat_btn = gr.Button("Analyze Chat Logs", variant="primary")
                
                with gr.Column():
                    chat_analysis_result = gr.Textbox(
                        label="Analysis Result",
                        lines=15,
                        interactive=False
                    )
            
            # Need to wrap async function
            def analyze_chat_wrapper(chat_json):
                return asyncio.run(ui.analyze_chat_logs_ui(chat_json))
            
            analyze_chat_btn.click(
                analyze_chat_wrapper,
                inputs=[chat_logs_input],
                outputs=[chat_analysis_result]
            )
        
        with gr.Tab("📋 View Tasks"):
            with gr.Row():
                with gr.Column():
                    profile_name = gr.Textbox(
                        label="Weighting Profile (optional, legacy)",
                        placeholder="Enter profile name for prioritized view..."
                    )
                    refresh_btn = gr.Button("Refresh Tasks", variant="primary")
                
                with gr.Column():
                    tasks_display = gr.Textbox(
                        label="Available Tasks",
                        lines=15,
                        interactive=False
                    )
            
            refresh_btn.click(
                ui.get_available_tasks_ui,
                inputs=[profile_name],
                outputs=[tasks_display]
            )
        
        with gr.Tab("✅ Complete / Snooze"):
            with gr.Row():
                with gr.Column():
                    complete_task_id = gr.Textbox(label="Task ID to Complete")
                    complete_btn = gr.Button("Complete Task", variant="primary")
                    
                    gr.Markdown("---")
                    
                    snooze_task_id = gr.Textbox(label="Task ID to Snooze")
                    snooze_btn = gr.Button("Snooze Task (Lower Priority)")
                
                with gr.Column():
                    action_result = gr.Textbox(label="Result", interactive=False, lines=10)
            
            complete_btn.click(
                ui.complete_task_ui,
                inputs=[complete_task_id],
                outputs=[action_result]
            )
            
            snooze_btn.click(
                ui.snooze_task_ui,
                inputs=[snooze_task_id],
                outputs=[action_result]
            )
        
        with gr.Tab("Dependencies"):
            with gr.Row():
                with gr.Column():
                    dep_task_id = gr.Textbox(label="Task ID")
                    dep_depends_on = gr.Textbox(label="Depends on Task ID")
                    dep_btn = gr.Button("Add Dependency", variant="primary")
                
                with gr.Column():
                    dep_result = gr.Textbox(label="Result", interactive=False)
            
            dep_btn.click(
                ui.add_dependency_ui,
                inputs=[dep_task_id, dep_depends_on],
                outputs=[dep_result]
            )
        
        with gr.Tab("Weighting Profiles"):
            with gr.Row():
                with gr.Column():
                    profile_name_save = gr.Textbox(label="Profile Name")
                    profile_weights = gr.Textbox(
                        label="Weights (JSON)",
                        placeholder='{"importance": 3.0, "urgency": 2.0, "health_related": 5.0}',
                        lines=5
                    )
                    save_profile_btn = gr.Button("Save Profile", variant="primary")
                
                with gr.Column():
                    profile_result = gr.Textbox(label="Result", interactive=False)
            
            save_profile_btn.click(
                ui.save_profile_ui,
                inputs=[profile_name_save, profile_weights],
                outputs=[profile_result]
            )
    
    return app


def main(argv=None):
    """Main entry point for task manager web UI."""
    if argv is None:
        argv = sys.argv[1:]
    
    parser = argparse.ArgumentParser(description="Task Manager Web UI")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind to")
    parser.add_argument("--db", default="tasks.db", help="SQLite database path")
    parser.add_argument("--cert", help="SSL certificate file")
    parser.add_argument("--key", help="SSL private key file")
    
    args = parser.parse_args(argv)
    
    # Check environment variables (following existing pattern)
    host = os.getenv("LIFESTORE_TASK_HOST", args.host)
    port = int(os.getenv("LIFESTORE_TASK_PORT", args.port))
    cert_file = os.getenv("LIFESTORE_TASK_SSL_CERT", args.cert)
    key_file = os.getenv("LIFESTORE_TASK_SSL_KEY", args.key)
    
    # SSL validation (following existing pattern)
    ssl_context = None
    if cert_file or key_file:
        if not (cert_file and key_file):
            print("Error: Both --cert and --key must be provided for SSL", file=sys.stderr)
            return 1
        
        cert_path = Path(cert_file)
        key_path = Path(key_file)
        
        if not cert_path.exists():
            print(f"Error: Certificate file not found: {cert_file}", file=sys.stderr)
            return 1
        
        if not key_path.exists():
            print(f"Error: Key file not found: {key_file}", file=sys.stderr)
            return 1
        
        ssl_context = (str(cert_path), str(key_path))
    
    # Create and launch the app
    app = create_app(args.db)
    
    print(f"Starting Task Manager on {host}:{port}")
    if ssl_context:
        print("SSL enabled")
    
    app.launch(
        server_name=host,
        server_port=port,
        ssl_keyfile=ssl_context[1] if ssl_context else None,
        ssl_certfile=ssl_context[0] if ssl_context else None,
        ssl_verify=False if ssl_context else True
    )


if __name__ == "__main__":
    sys.exit(main())