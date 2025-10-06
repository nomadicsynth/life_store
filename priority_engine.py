"""
Priority Engine - Learns what matters from task embeddings and user behavior.

Uses eigendecomposition to discover latent "dimensions" in the task space,
learns preferences from user actions, and scores tasks adaptively based on
current context and research gaps.

Heavily inspired by the deep_research_tool.py approach.
"""

import hashlib
import json
import logging
import pickle
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

import numpy as np
import requests
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """Cache for embeddings to avoid redundant API calls"""

    def __init__(self, cache_dir: Path, max_size: int = 10000):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "embeddings.pkl"
        self.max_size = max_size
        self.hit_count = 0
        self.miss_count = 0
        
        # Load cache from disk
        self.cache = self._load_cache()
        self.access_order = list(self.cache.keys())

    def _load_cache(self) -> Dict:
        """Load cache from disk"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return {}

    def _save_cache(self):
        """Save cache to disk"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def _generate_key(self, text: str) -> str:
        """Generate SHA-256 cache key from text"""
        text_sample = text[:2000] if len(text) > 2000 else text
        return hashlib.sha256(text_sample.encode("utf-8")).hexdigest()[:16]

    def get(self, text: str) -> Optional[np.ndarray]:
        """Get embedding from cache"""
        key = self._generate_key(text)
        result = self.cache.get(key)
        
        if result is not None:
            self.hit_count += 1
            # Move to end (most recently used)
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            return np.array(result)
        
        self.miss_count += 1
        return None

    def set(self, text: str, embedding: np.ndarray):
        """Store embedding in cache"""
        key = self._generate_key(text)
        
        # If key exists, update and move to end
        if key in self.cache:
            self.cache[key] = embedding.tolist()
            self.access_order.remove(key)
            self.access_order.append(key)
            return
        
        # Add new entry
        self.cache[key] = embedding.tolist()
        self.access_order.append(key)
        
        # LRU eviction if too large
        while len(self.cache) > self.max_size:
            old_key = self.access_order.pop(0)
            del self.cache[old_key]
        
        # Save periodically (every 10 additions)
        if len(self.cache) % 10 == 0:
            self._save_cache()

    def stats(self) -> Dict:
        """Return cache statistics"""
        total = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total if total > 0 else 0
        return {
            "size": len(self.cache),
            "hits": self.hit_count,
            "misses": self.miss_count,
            "hit_rate": hit_rate,
        }


class EmbeddingModel:
    """Wrapper for Ollama embeddings with caching"""

    def __init__(self, model: str = "granite-embedding:278m", 
                 ollama_url: str = "http://localhost:11434",
                 cache_dir: Path = Path(".cache/embeddings")):
        self.model = model
        self.ollama_url = ollama_url
        self.cache = EmbeddingCache(cache_dir)
        self.embedding_dim = 768  # granite-embedding:278m dimension

    async def encode(self, text: str) -> np.ndarray:
        """Get embedding for text, using cache when available"""
        # Check cache first
        cached = self.cache.get(text)
        if cached is not None:
            return cached
        
        # Fetch from Ollama
        try:
            response = requests.post(
                f"{self.ollama_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
                timeout=30
            )
            response.raise_for_status()
            embedding = np.array(response.json()["embedding"])
            
            # Cache it
            self.cache.set(text, embedding)
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to get embedding: {e}")
            # Return zero vector as fallback
            return np.zeros(self.embedding_dim)

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode multiple texts (sync version for now)"""
        embeddings = []
        for text in texts:
            # Note: This is sync, but we'll make it async-compatible later
            cached = self.cache.get(text)
            if cached is not None:
                embeddings.append(cached)
            else:
                try:
                    response = requests.post(
                        f"{self.ollama_url}/api/embeddings",
                        json={"model": self.model, "prompt": text},
                        timeout=30
                    )
                    response.raise_for_status()
                    emb = np.array(response.json()["embedding"])
                    self.cache.set(text, emb)
                    embeddings.append(emb)
                except Exception as e:
                    logger.error(f"Failed to get embedding: {e}")
                    embeddings.append(np.zeros(self.embedding_dim))
        
        return np.array(embeddings)


class DimensionLearner:
    """Learns latent dimensions from task embeddings using PCA/eigendecomposition"""

    def __init__(self, n_components: int = 10, decay_factor: float = 0.95):
        self.n_components = n_components
        self.decay_factor = decay_factor
        self.pca = None
        self.embeddings_history = []
        self.coverage = None  # Track coverage per dimension
        self.dimension_labels = []  # Human-readable labels
        self.vocabulary = None  # For translation
        self.vocab_embeddings = None
        
    def update(self, embeddings: np.ndarray, quality_factors: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update dimensions with new embeddings.
        
        Args:
            embeddings: Array of shape (n_samples, embedding_dim)
            quality_factors: Optional weights for each embedding
            
        Returns:
            (eigenvectors, coverage_vector)
        """
        if embeddings.shape[0] == 0:
            logger.warning("No embeddings to learn from")
            return np.array([]), np.array([])
        
        # Apply exponential decay to old embeddings
        if len(self.embeddings_history) > 0:
            decayed = [emb * (self.decay_factor ** i) 
                      for i, emb in enumerate(reversed(self.embeddings_history))]
            all_embeddings = np.vstack(decayed + [embeddings])
        else:
            all_embeddings = embeddings
        
        # Keep only recent history
        self.embeddings_history.append(embeddings)
        if len(self.embeddings_history) > 100:
            self.embeddings_history.pop(0)
        
        # Fit PCA
        n_comp = min(self.n_components, all_embeddings.shape[0], all_embeddings.shape[1])
        self.pca = PCA(n_components=n_comp)
        
        try:
            self.pca.fit(all_embeddings)
        except Exception as e:
            logger.error(f"PCA failed: {e}")
            return np.array([]), np.array([])
        
        # Compute coverage
        if quality_factors is None:
            quality_factors = np.ones(embeddings.shape[0])
        
        # Project current embeddings onto dimensions
        coords = self.pca.transform(embeddings)
        
        # Coverage = weighted sum of absolute coordinates
        # Reset coverage if number of components changed
        if self.coverage is None or len(self.coverage) != n_comp:
            self.coverage = np.zeros(n_comp)
        
        weighted_coords = np.abs(coords) * quality_factors[:, np.newaxis]
        self.coverage = self.coverage * self.decay_factor + weighted_coords.sum(axis=0)
        
        return self.pca.components_, self.coverage

    def get_gap_vector(self, target_profile: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate gap vector showing under-explored dimensions.
        
        Args:
            target_profile: Desired coverage profile (default: uniform)
            
        Returns:
            Gap vector in original embedding space
        """
        if self.pca is None or self.coverage is None:
            return np.array([])
        
        if target_profile is None:
            # Default: uniform coverage
            target_profile = np.ones_like(self.coverage) * self.coverage.mean()
        
        # Gap in dimension space
        gap_coords = target_profile - self.coverage
        
        # Project back to embedding space
        gap_vector = self.pca.inverse_transform(gap_coords)
        
        return gap_vector

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Transform embeddings to dimension coordinates"""
        if self.pca is None:
            return embeddings
        return self.pca.transform(embeddings)

    def load_vocabulary(self, words: List[str], word_embeddings: np.ndarray):
        """Load vocabulary for dimension labeling"""
        self.vocabulary = words
        self.vocab_embeddings = word_embeddings

    def label_dimensions(self, top_k: int = 5) -> List[List[str]]:
        """
        Translate eigenvectors to human-readable labels.
        
        Returns:
            List of top-k words for each dimension
        """
        if self.pca is None or self.vocabulary is None:
            return []
        
        labels = []
        for component in self.pca.components_:
            # Find nearest vocabulary words
            similarities = cosine_similarity(
                component.reshape(1, -1),
                self.vocab_embeddings
            )[0]
            
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            top_words = [self.vocabulary[i] for i in top_indices]
            labels.append(top_words)
        
        self.dimension_labels = labels
        return labels

    def get_dimension_info(self) -> List[Dict]:
        """Get comprehensive dimension information"""
        if self.pca is None:
            return []
        
        info = []
        for i in range(len(self.pca.components_)):
            dim_info = {
                "index": i,
                "variance_explained": float(self.pca.explained_variance_ratio_[i]),
                "coverage": float(self.coverage[i]) if self.coverage is not None else 0.0,
                "labels": self.dimension_labels[i] if i < len(self.dimension_labels) else [],
            }
            info.append(dim_info)
        
        return info


class PreferenceModel:
    """Learns user preferences from actions (PDV - Preference Direction Vector)"""

    def __init__(self, embedding_dim: int = 768, decay_factor: float = 0.9):
        self.embedding_dim = embedding_dim
        self.decay_factor = decay_factor
        self.pdv = np.zeros(embedding_dim)
        self.action_count = 0
        self.action_history = []  # For debugging/analysis

    def update(self, 
               kept_embeddings: List[np.ndarray],
               removed_embeddings: List[np.ndarray],
               weight: float = 1.0):
        """
        Update PDV from user actions.
        
        Args:
            kept_embeddings: Embeddings of promoted/completed tasks
            removed_embeddings: Embeddings of snoozed/demoted tasks
            weight: Importance weight for this update
        """
        if len(kept_embeddings) == 0 and len(removed_embeddings) == 0:
            return
        
        # Compute direction: toward kept, away from removed
        kept_mean = np.mean(kept_embeddings, axis=0) if len(kept_embeddings) > 0 else np.zeros(self.embedding_dim)
        removed_mean = np.mean(removed_embeddings, axis=0) if len(removed_embeddings) > 0 else np.zeros(self.embedding_dim)
        
        direction = kept_mean - removed_mean
        
        # Normalize
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm
        
        # Update PDV with decay
        self.pdv = self.pdv * self.decay_factor + direction * weight
        
        # Re-normalize PDV
        pdv_norm = np.linalg.norm(self.pdv)
        if pdv_norm > 0:
            self.pdv = self.pdv / pdv_norm
        
        self.action_count += 1
        self.action_history.append({
            "timestamp": datetime.now().isoformat(),
            "kept_count": len(kept_embeddings),
            "removed_count": len(removed_embeddings),
            "weight": weight,
        })

    def alignment(self, embedding: np.ndarray) -> float:
        """Calculate alignment of an embedding with user preferences"""
        if np.linalg.norm(self.pdv) == 0:
            return 0.0
        
        similarity = cosine_similarity(
            embedding.reshape(1, -1),
            self.pdv.reshape(1, -1)
        )[0][0]
        
        return float(similarity)

    def get_pdv(self) -> np.ndarray:
        """Get current preference direction vector"""
        return self.pdv.copy()


class TaskScorer:
    """Scores tasks using learned dimensions, PDV, and context"""

    def __init__(self):
        self.weights = {
            "dimensions": 1.0,
            "pdv": 1.5,
            "gap": 1.2,
            "due_pressure": 2.0,
            "blocked": -3.0,
            "graph_centrality": 0.8,
        }
        self.exploration_rate = 0.1

    def score(self,
              task_embedding: np.ndarray,
              dimension_learner: DimensionLearner,
              preference_model: PreferenceModel,
              due_pressure: float = 0.0,
              blocked: bool = False,
              graph_centrality: float = 0.0,
              gap_alignment: float = 0.0) -> float:
        """
        Score a task based on multiple factors.
        
        Args:
            task_embedding: Task embedding vector
            dimension_learner: Trained dimension learner
            preference_model: Trained preference model
            due_pressure: Urgency score (0-1)
            blocked: Whether task is blocked
            graph_centrality: Graph importance (0-1)
            gap_alignment: Alignment with research gaps (0-1)
            
        Returns:
            Composite score
        """
        score = 0.0
        
        # Dimension contribution (coverage-weighted)
        if dimension_learner.pca is not None:
            coords = dimension_learner.transform(task_embedding.reshape(1, -1))[0]
            dim_score = np.sum(np.abs(coords))
            score += self.weights["dimensions"] * dim_score
        
        # PDV alignment
        pdv_score = preference_model.alignment(task_embedding)
        score += self.weights["pdv"] * pdv_score
        
        # Gap alignment (helps explore under-covered areas)
        score += self.weights["gap"] * gap_alignment
        
        # Due pressure
        score += self.weights["due_pressure"] * due_pressure
        
        # Blocked penalty
        if blocked:
            score += self.weights["blocked"]
        
        # Graph centrality
        score += self.weights["graph_centrality"] * graph_centrality
        
        # Exploration noise
        if np.random.random() < self.exploration_rate:
            score += np.random.normal(0, 0.5)
        
        return score

    def rank_tasks(self,
                   task_embeddings: Dict[str, np.ndarray],
                   dimension_learner: DimensionLearner,
                   preference_model: PreferenceModel,
                   task_metadata: Dict[str, Dict] = None) -> List[Tuple[str, float]]:
        """
        Rank all tasks by score.
        
        Args:
            task_embeddings: Dict of task_id -> embedding
            dimension_learner: Trained dimension learner
            preference_model: Trained preference model
            task_metadata: Optional metadata (due_pressure, blocked, etc.)
            
        Returns:
            List of (task_id, score) tuples, sorted descending
        """
        if task_metadata is None:
            task_metadata = {}
        
        # Calculate gap vector
        gap_vector = dimension_learner.get_gap_vector()
        
        scores = []
        for task_id, embedding in task_embeddings.items():
            metadata = task_metadata.get(task_id, {})
            
            # Gap alignment
            gap_alignment = 0.0
            if len(gap_vector) > 0:
                gap_alignment = cosine_similarity(
                    embedding.reshape(1, -1),
                    gap_vector.reshape(1, -1)
                )[0][0]
            
            score = self.score(
                task_embedding=embedding,
                dimension_learner=dimension_learner,
                preference_model=preference_model,
                due_pressure=metadata.get("due_pressure", 0.0),
                blocked=metadata.get("blocked", False),
                graph_centrality=metadata.get("graph_centrality", 0.0),
                gap_alignment=gap_alignment,
            )
            
            scores.append((task_id, score))
        
        # Sort descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores


class PriorityEngine:
    """
    Main priority engine that orchestrates dimension learning,
    preference learning, and task scoring.
    """

    def __init__(self,
                 cache_dir: Path = Path(".cache"),
                 embedding_model: str = "granite-embedding:278m",
                 ollama_url: str = "http://localhost:11434"):
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.embedding_model = EmbeddingModel(
            model=embedding_model,
            ollama_url=ollama_url,
            cache_dir=self.cache_dir / "embeddings"
        )
        self.dimension_learner = DimensionLearner(n_components=10)
        self.preference_model = PreferenceModel(embedding_dim=768)
        self.scorer = TaskScorer()
        
        # State
        self.task_embeddings: Dict[str, np.ndarray] = {}
        self.last_update = None

    def update_from_tasks(self, tasks: List[Dict], force_update: bool = False):
        """
        Update internal state from task list.
        
        Args:
            tasks: List of task dicts with 'id', 'title', 'description', etc.
            force_update: Force re-embedding even if cached
        """
        # Generate embeddings
        new_embeddings = {}
        texts = []
        task_ids = []
        
        for task in tasks:
            task_id = task["id"]
            # Create text representation
            text = f"{task.get('title', '')} {task.get('description', '')}"
            
            if force_update or task_id not in self.task_embeddings:
                texts.append(text)
                task_ids.append(task_id)
        
        # Batch encode new tasks
        if texts:
            embeddings = self.embedding_model.encode_batch(texts)
            for task_id, emb in zip(task_ids, embeddings):
                self.task_embeddings[task_id] = emb
                new_embeddings[task_id] = emb
        
        # Update dimensions
        if len(self.task_embeddings) > 0:
            all_embs = np.array(list(self.task_embeddings.values()))
            self.dimension_learner.update(all_embs)
        
        self.last_update = datetime.now()

    def update_from_actions(self, 
                           kept_task_ids: List[str],
                           removed_task_ids: List[str],
                           weight: float = 1.0):
        """
        Update preferences from user actions.
        
        Args:
            kept_task_ids: IDs of tasks that were promoted/completed
            removed_task_ids: IDs of tasks that were snoozed/demoted
            weight: Importance weight
        """
        kept_embs = [self.task_embeddings[tid] for tid in kept_task_ids 
                    if tid in self.task_embeddings]
        removed_embs = [self.task_embeddings[tid] for tid in removed_task_ids 
                       if tid in self.task_embeddings]
        
        self.preference_model.update(kept_embs, removed_embs, weight)

    def rank_tasks(self, task_metadata: Dict[str, Dict] = None) -> List[Tuple[str, float]]:
        """
        Rank all tasks by current priority.
        
        Args:
            task_metadata: Optional metadata for scoring
            
        Returns:
            List of (task_id, score) tuples, sorted by priority
        """
        return self.scorer.rank_tasks(
            self.task_embeddings,
            self.dimension_learner,
            self.preference_model,
            task_metadata
        )

    def get_dimension_info(self) -> List[Dict]:
        """Get information about current dimensions"""
        return self.dimension_learner.get_dimension_info()

    def get_stats(self) -> Dict:
        """Get engine statistics"""
        return {
            "cache_stats": self.embedding_model.cache.stats(),
            "task_count": len(self.task_embeddings),
            "dimension_count": len(self.dimension_learner.pca.components_) if self.dimension_learner.pca else 0,
            "pdv_strength": float(np.linalg.norm(self.preference_model.pdv)),
            "action_count": self.preference_model.action_count,
            "last_update": self.last_update.isoformat() if self.last_update else None,
        }

    def save_state(self, filepath: Path):
        """Save engine state to disk"""
        state = {
            "task_embeddings": {k: v.tolist() for k, v in self.task_embeddings.items()},
            "dimension_learner": {
                "components": self.dimension_learner.pca.components_.tolist() if self.dimension_learner.pca else None,
                "explained_variance_ratio": self.dimension_learner.pca.explained_variance_ratio_.tolist() if self.dimension_learner.pca else None,
                "coverage": self.dimension_learner.coverage.tolist() if self.dimension_learner.coverage is not None else None,
                "dimension_labels": self.dimension_learner.dimension_labels,
            },
            "preference_model": {
                "pdv": self.preference_model.pdv.tolist(),
                "action_count": self.preference_model.action_count,
                "action_history": self.preference_model.action_history,
            },
            "scorer_weights": self.scorer.weights,
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

    def load_state(self, filepath: Path):
        """Load engine state from disk"""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        # Restore task embeddings
        self.task_embeddings = {
            k: np.array(v) for k, v in state["task_embeddings"].items()
        }
        
        # Restore dimension learner (partially - PCA needs refitting)
        dim_state = state["dimension_learner"]
        if dim_state["coverage"]:
            self.dimension_learner.coverage = np.array(dim_state["coverage"])
        self.dimension_learner.dimension_labels = dim_state["dimension_labels"]
        
        # Restore preference model
        pref_state = state["preference_model"]
        self.preference_model.pdv = np.array(pref_state["pdv"])
        self.preference_model.action_count = pref_state["action_count"]
        self.preference_model.action_history = pref_state["action_history"]
        
        # Restore scorer weights
        self.scorer.weights = state["scorer_weights"]
