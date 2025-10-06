"""
Chat Log Analyzer - Passive feedback from AI conversations.

Embeds chat log chunks and uses them to inform PDV and dimension learning.
This is the "journaling sneak-attack" - learning from natural conversation
instead of requiring structured input.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests

logger = logging.getLogger(__name__)


class ChatLogAnalyzer:
    """Analyzes chat logs to extract task-related insights and preferences"""

    def __init__(self, embedding_model):
        """
        Args:
            embedding_model: EmbeddingModel instance from priority_engine
        """
        self.embedding_model = embedding_model
        self.log_chunks = []  # Stored (timestamp, text, embedding) tuples
        self.task_mentions = {}  # task_id -> [(timestamp, sentiment, context)]
        
    def chunk_conversation(self, messages: List[Dict], 
                          chunk_size: int = 3) -> List[Tuple[datetime, str]]:
        """
        Chunk conversation into overlapping windows.
        
        Args:
            messages: List of message dicts with 'role', 'content', 'timestamp'
            chunk_size: Number of messages per chunk
            
        Returns:
            List of (timestamp, text) tuples
        """
        chunks = []
        
        for i in range(len(messages) - chunk_size + 1):
            window = messages[i:i + chunk_size]
            
            # Combine messages in window
            text_parts = []
            for msg in window:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                text_parts.append(f"{role}: {content}")
            
            combined_text = "\n".join(text_parts)
            
            # Use timestamp of first message in window
            timestamp = window[0].get("timestamp", datetime.now().isoformat())
            if isinstance(timestamp, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp)
                except:
                    timestamp = datetime.now()
            
            chunks.append((timestamp, combined_text))
        
        return chunks

    async def analyze_logs(self, messages: List[Dict], 
                          task_list: Optional[List[Dict]] = None,
                          time_window: Optional[timedelta] = None) -> Dict:
        """
        Analyze chat logs for task-related insights.
        
        Args:
            messages: Chat messages to analyze
            task_list: Optional list of current tasks for mention detection
            time_window: Only analyze logs within this time window
            
        Returns:
            Analysis dict with embeddings, sentiments, task mentions
        """
        # Filter by time window if specified
        if time_window:
            cutoff = datetime.now() - time_window
            messages = [m for m in messages 
                       if self._get_timestamp(m) >= cutoff]
        
        # Chunk conversation
        chunks = self.chunk_conversation(messages)
        
        # Embed chunks
        chunk_embeddings = []
        for timestamp, text in chunks:
            emb = await self.embedding_model.encode(text)
            chunk_embeddings.append(emb)
            self.log_chunks.append((timestamp, text, emb))
        
        # Keep only recent chunks (memory management)
        if len(self.log_chunks) > 1000:
            self.log_chunks = self.log_chunks[-1000:]
        
        # Detect task mentions if task list provided
        task_mentions = {}
        if task_list:
            task_mentions = await self._detect_task_mentions(chunks, task_list)
        
        # Analyze sentiment trends
        sentiment_summary = self._analyze_sentiment_trends(chunks)
        
        return {
            "chunk_count": len(chunks),
            "embeddings": np.array(chunk_embeddings) if chunk_embeddings else np.array([]),
            "task_mentions": task_mentions,
            "sentiment_summary": sentiment_summary,
            "time_range": (chunks[0][0], chunks[-1][0]) if chunks else (None, None),
        }

    async def extract_preference_signals(self, 
                                        analysis: Dict,
                                        task_embeddings: Dict[str, np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Extract preference signals from chat analysis.
        
        Returns:
            (positive_embeddings, negative_embeddings) for PDV update
        """
        positive = []
        negative = []
        
        task_mentions = analysis.get("task_mentions", {})
        
        for task_id, mentions in task_mentions.items():
            if task_id not in task_embeddings:
                continue
            
            task_emb = task_embeddings[task_id]
            
            # Aggregate sentiment
            sentiments = [m["sentiment"] for m in mentions]
            avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
            
            if avg_sentiment > 0.3:
                positive.append(task_emb)
            elif avg_sentiment < -0.3:
                negative.append(task_emb)
        
        # Also look for general positive/negative patterns in embeddings
        chunk_embs = analysis.get("embeddings", np.array([]))
        if len(chunk_embs) > 0:
            # Simple heuristic: chunks mentioning "excited", "looking forward",
            # "important", "priority" are positive signals
            # This would ideally use a sentiment model, but we'll keep it simple
            pass
        
        return positive, negative

    async def _detect_task_mentions(self, 
                                   chunks: List[Tuple[datetime, str]],
                                   task_list: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Detect which tasks are mentioned in chat chunks.
        
        Returns:
            Dict of task_id -> list of mention dicts
        """
        mentions = {}
        
        # Embed tasks
        task_texts = {}
        for task in task_list:
            task_id = task["id"]
            text = f"{task.get('title', '')} {task.get('description', '')}"
            task_texts[task_id] = text
        
        task_embeddings = {}
        for task_id, text in task_texts.items():
            task_embeddings[task_id] = await self.embedding_model.encode(text)
        
        # Check each chunk for task mentions
        for timestamp, chunk_text in chunks:
            chunk_emb = await self.embedding_model.encode(chunk_text)
            
            # Find tasks with high similarity to this chunk
            for task_id, task_emb in task_embeddings.items():
                similarity = self._cosine_similarity(chunk_emb, task_emb)
                
                # Threshold for "mention"
                if similarity > 0.5:
                    # Estimate sentiment (very basic - could use a model)
                    sentiment = self._estimate_sentiment(chunk_text)
                    
                    if task_id not in mentions:
                        mentions[task_id] = []
                    
                    mentions[task_id].append({
                        "timestamp": timestamp.isoformat(),
                        "similarity": float(similarity),
                        "sentiment": sentiment,
                        "context": chunk_text[:200],  # First 200 chars
                    })
        
        return mentions

    def _estimate_sentiment(self, text: str) -> float:
        """
        Simple sentiment estimation (-1 to 1).
        
        This is a placeholder - could be replaced with a proper sentiment model.
        """
        text_lower = text.lower()
        
        positive_words = [
            "excited", "great", "important", "priority", "love", "good",
            "excellent", "perfect", "yes", "looking forward", "can't wait",
            "amazing", "awesome", "wonderful"
        ]
        
        negative_words = [
            "stuck", "blocked", "frustrat", "annoying", "hate", "terrible",
            "awful", "worried", "concerned", "problem", "issue", "struggling",
            "burnout", "exhausted", "overwhelmed"
        ]
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total = positive_count + negative_count
        if total == 0:
            return 0.0
        
        return (positive_count - negative_count) / total

    def _analyze_sentiment_trends(self, chunks: List[Tuple[datetime, str]]) -> Dict:
        """Analyze sentiment trends over time"""
        if not chunks:
            return {}
        
        sentiments = [self._estimate_sentiment(text) for _, text in chunks]
        
        return {
            "average": sum(sentiments) / len(sentiments) if sentiments else 0,
            "recent_average": sum(sentiments[-10:]) / min(10, len(sentiments)) if sentiments else 0,
            "trend": "improving" if len(sentiments) > 5 and sentiments[-5:] > sentiments[:5] else "stable",
        }

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)

    def _get_timestamp(self, message: Dict) -> datetime:
        """Extract timestamp from message"""
        timestamp = message.get("timestamp", datetime.now().isoformat())
        if isinstance(timestamp, str):
            try:
                return datetime.fromisoformat(timestamp)
            except:
                return datetime.now()
        return timestamp

    def get_recent_embeddings(self, 
                            time_window: Optional[timedelta] = None,
                            max_count: int = 100) -> List[np.ndarray]:
        """
        Get recent chat embeddings for dimension learning.
        
        Args:
            time_window: Only return embeddings within this time window
            max_count: Maximum number of embeddings to return
            
        Returns:
            List of embeddings
        """
        filtered = self.log_chunks
        
        if time_window:
            cutoff = datetime.now() - time_window
            filtered = [(ts, txt, emb) for ts, txt, emb in filtered if ts >= cutoff]
        
        # Most recent first
        filtered = sorted(filtered, key=lambda x: x[0], reverse=True)
        
        # Take max_count
        filtered = filtered[:max_count]
        
        return [emb for _, _, emb in filtered]


class OpenWebUIConnector:
    """
    Connector for Open WebUI to fetch chat logs.
    
    This would integrate with Open WebUI's API to fetch conversation history.
    For now, this is a placeholder showing the intended interface.
    """

    def __init__(self, api_url: str = "http://localhost:3000/api"):
        self.api_url = api_url
        self.session = requests.Session()

    def fetch_recent_conversations(self, 
                                  hours: int = 24,
                                  user_id: Optional[str] = None) -> List[Dict]:
        """
        Fetch recent conversations from Open WebUI.
        
        Args:
            hours: How many hours back to fetch
            user_id: Optional user ID to filter by
            
        Returns:
            List of conversation dicts
        """
        # This is a placeholder - actual implementation would depend on
        # Open WebUI's API structure
        try:
            response = self.session.get(
                f"{self.api_url}/conversations",
                params={"hours": hours, "user_id": user_id},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to fetch conversations: {e}")
            return []

    def fetch_conversation_messages(self, conversation_id: str) -> List[Dict]:
        """Fetch messages from a specific conversation"""
        try:
            response = self.session.get(
                f"{self.api_url}/conversations/{conversation_id}/messages",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to fetch messages: {e}")
            return []
