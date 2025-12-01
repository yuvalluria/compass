"""
Stage 1: Advanced Task Embedding

Converts user task descriptions into high-quality embeddings
using E5 or BGE models for better semantic understanding.
"""

import json
from typing import Dict, List, Tuple, Optional
import numpy as np

from .config import EmbeddingConfig, default_config

# Lazy imports for optional dependencies
SentenceTransformer = None

def _load_sentence_transformer():
    global SentenceTransformer
    if SentenceTransformer is None:
        from sentence_transformers import SentenceTransformer as ST
        SentenceTransformer = ST
    return SentenceTransformer

class TaskEmbedder:
    """
    Advanced task embedding using E5/BGE models.
    
    Key improvements over MiniLM:
    - Better semantic understanding of task intent
    - Instruction-aware embeddings
    - Higher dimensional space (768 vs 384)
    """
    
    # Predefined use cases with descriptions for matching
    PREDEFINED_TASKS = {
        "code_completion": {
            "description": "Fast code completion and autocomplete for IDEs. Real-time code suggestions, IntelliSense, typing assistance. Latency-critical for smooth developer experience.",
            "experience_class": "instant",
            "typical_prompt_tokens": (50, 200),
            "typical_output_tokens": (10, 100),
        },
        "chatbot_conversational": {
            "description": "Real-time conversational chatbots. Customer service, personal assistants, interactive dialogue. Natural conversation flow with quick responses.",
            "experience_class": "conversational",
            "typical_prompt_tokens": (100, 500),
            "typical_output_tokens": (50, 300),
        },
        "code_generation_detailed": {
            "description": "Detailed code generation with explanations. Software development, code with comments and documentation. Quality over speed.",
            "experience_class": "interactive",
            "typical_prompt_tokens": (200, 1000),
            "typical_output_tokens": (100, 500),
        },
        "translation": {
            "description": "Document translation and localization. Multilingual text conversion. Non-interactive, accuracy prioritized.",
            "experience_class": "interactive",
            "typical_prompt_tokens": (200, 2000),
            "typical_output_tokens": (200, 2000),
        },
        "content_generation": {
            "description": "Content creation, marketing copy, articles, blog posts. Creative writing and marketing materials.",
            "experience_class": "interactive",
            "typical_prompt_tokens": (100, 1000),
            "typical_output_tokens": (200, 1000),
        },
        "summarization_short": {
            "description": "Short document summarization. Executive summaries, brief text condensation. Prefill-heavy workload.",
            "experience_class": "deferred",
            "typical_prompt_tokens": (500, 4000),
            "typical_output_tokens": (50, 200),
        },
        "document_analysis_rag": {
            "description": "RAG-based document Q&A. Retrieval-augmented generation, answering questions from documents and knowledge bases.",
            "experience_class": "deferred",
            "typical_prompt_tokens": (1000, 8000),
            "typical_output_tokens": (100, 500),
        },
        "long_document_summarization": {
            "description": "Long document summarization. Processing extensive research papers, large text files. High prefill, medium output.",
            "experience_class": "deferred",
            "typical_prompt_tokens": (4000, 32000),
            "typical_output_tokens": (200, 1000),
        },
        "research_legal_analysis": {
            "description": "Research and legal document analysis. Academic research, legal document review, in-depth analysis. Batch processing, quality prioritized.",
            "experience_class": "batch",
            "typical_prompt_tokens": (8000, 64000),
            "typical_output_tokens": (500, 2000),
        },
    }
    
    def __init__(self, config: EmbeddingConfig = None):
        self.config = config or default_config.embedding
        self.model = None
        self._task_embeddings = None
        
    def _load_model(self):
        """Lazy load the embedding model"""
        if self.model is None:
            ST = _load_sentence_transformer()
            print(f"Loading embedding model: {self.config.model_name}")
            self.model = ST(self.config.model_name, device=self.config.device)
            print(f"  ✓ Model loaded ({self.config.embedding_dimension} dimensions)")
        return self.model
    
    def _get_task_embeddings(self) -> Dict[str, np.ndarray]:
        """Get or compute embeddings for all predefined tasks"""
        if self._task_embeddings is None:
            model = self._load_model()
            
            # For E5 models, add instruction prefix
            is_e5 = "e5" in self.config.model_name.lower()
            
            self._task_embeddings = {}
            for task_name, task_info in self.PREDEFINED_TASKS.items():
                text = task_info["description"]
                if is_e5:
                    # E5 models use query/passage prefixes
                    text = f"query: {text}"
                
                embedding = model.encode([text], show_progress_bar=False)[0]
                self._task_embeddings[task_name] = embedding
                
            print(f"  ✓ Generated embeddings for {len(self._task_embeddings)} predefined tasks")
            
        return self._task_embeddings
    
    def embed_task(self, task_description: str) -> np.ndarray:
        """
        Generate embedding for a user task description.
        
        Args:
            task_description: User's plain text task description
            
        Returns:
            Embedding vector (numpy array)
        """
        model = self._load_model()
        
        # For E5 models, add instruction prefix
        is_e5 = "e5" in self.config.model_name.lower()
        text = f"query: {task_description}" if is_e5 else task_description
        
        embedding = model.encode([text], show_progress_bar=False)[0]
        return embedding
    
    def find_similar_tasks(
        self, 
        task_description: str, 
        top_k: int = 3
    ) -> List[Tuple[str, float, Dict]]:
        """
        Find similar predefined tasks using cosine similarity.
        
        Args:
            task_description: User's task description
            top_k: Number of top matches to return
            
        Returns:
            List of (task_name, similarity_score, task_info) tuples
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Get user embedding
        user_embedding = self.embed_task(task_description).reshape(1, -1)
        
        # Get predefined task embeddings
        task_embeddings = self._get_task_embeddings()
        
        # Calculate similarities
        similarities = []
        for task_name, task_embedding in task_embeddings.items():
            task_embedding_2d = task_embedding.reshape(1, -1)
            similarity = cosine_similarity(user_embedding, task_embedding_2d)[0][0]
            task_info = self.PREDEFINED_TASKS[task_name].copy()
            similarities.append((task_name, float(similarity), task_info))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def analyze_task(self, task_description: str) -> Dict:
        """
        Analyze a task description and return structured JSON.
        
        Args:
            task_description: User's plain text task description
            
        Returns:
            Dictionary with task analysis
        """
        # Find similar predefined tasks
        similar_tasks = self.find_similar_tasks(task_description, top_k=5)
        
        # Get the best match
        best_match_name, best_similarity, best_info = similar_tasks[0]
        
        # Estimate token ranges based on best match
        prompt_min, prompt_max = best_info["typical_prompt_tokens"]
        output_min, output_max = best_info["typical_output_tokens"]
        
        # Determine experience class
        experience_class = best_info["experience_class"]
        
        # Build result
        result = {
            "task": {
                "description": task_description,
                "primary_category": best_match_name,
                "experience_class": experience_class,
                "estimated_prompt_tokens": {
                    "min": prompt_min,
                    "max": prompt_max,
                    "typical": (prompt_min + prompt_max) // 2
                },
                "estimated_output_tokens": {
                    "min": output_min,
                    "max": output_max,
                    "typical": (output_min + output_max) // 2
                },
            },
            "matched_tasks": [
                {
                    "name": name,
                    "similarity": round(score, 4),
                    "experience_class": info["experience_class"]
                }
                for name, score, info in similar_tasks
            ],
            "embedding_model": self.config.model_name,
        }
        
        return result
    
    def save_task_analysis(self, task_description: str, output_path: str):
        """Analyze task and save to JSON file"""
        result = self.analyze_task(task_description)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        
        print(f"  ✓ Saved task analysis to {output_path}")
        return result


# Convenience function
def analyze_task(task_description: str) -> Dict:
    """Quick function to analyze a task"""
    embedder = TaskEmbedder()
    return embedder.analyze_task(task_description)

