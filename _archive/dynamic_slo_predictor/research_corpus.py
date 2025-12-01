"""
Research Corpus with ChromaDB Vector Database

Uses the SAME E5 embedding model as task detection for:
1. Indexing research documents (papers, benchmarks)
2. Semantic search to find relevant research for a query

This provides research-backed justification for SLO values.
"""

from typing import Dict, List, Optional
from pathlib import Path

from .config import EmbeddingConfig, default_config
from .research_data import RESEARCH_CORPUS, get_all_documents

# Lazy imports
chromadb = None
SentenceTransformer = None


def _load_chromadb():
    global chromadb
    if chromadb is None:
        import chromadb as cdb
        chromadb = cdb
    return chromadb


def _load_sentence_transformer():
    global SentenceTransformer
    if SentenceTransformer is None:
        from sentence_transformers import SentenceTransformer as ST
        SentenceTransformer = ST
    return SentenceTransformer


class ResearchCorpus:
    """
    Vector database for research knowledge using ChromaDB.
    
    Uses E5 model (same as task detection) for semantic search
    over academic papers and industry benchmarks.
    """
    
    def __init__(self, embedding_config: EmbeddingConfig = None):
        self.embedding_config = embedding_config or default_config.embedding
        self.client = None
        self.collection = None
        self.embedding_model = None
        self._initialized = False
        
    def _get_embedding_function(self):
        """Create embedding function for ChromaDB using E5 model"""
        if self.embedding_model is None:
            ST = _load_sentence_transformer()
            print(f"  Loading E5 model for RAG: {self.embedding_config.model_name}")
            self.embedding_model = ST(
                self.embedding_config.model_name, 
                device=self.embedding_config.device
            )
        
        class E5EmbeddingFunction:
            """Custom embedding function for ChromaDB with E5 model"""
            def __init__(self, model, model_name: str):
                self.model = model
                self._model_name = model_name
                self.is_e5 = "e5" in model_name.lower()
            
            def name(self) -> str:
                return self._model_name
            
            def _encode(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
                if self.is_e5:
                    prefix = "query: " if is_query else "passage: "
                    texts = [f"{prefix}{text}" for text in texts]
                embeddings = self.model.encode(texts, show_progress_bar=False)
                return embeddings.tolist()
                
            def __call__(self, input: List[str]) -> List[List[float]]:
                return self._encode(input, is_query=False)
            
            def embed_documents(self, input: List[str]) -> List[List[float]]:
                return self._encode(input, is_query=False)
            
            def embed_query(self, input: str) -> List[List[float]]:
                return self._encode([input], is_query=True)
        
        return E5EmbeddingFunction(self.embedding_model, self.embedding_config.model_name)
    
    def initialize(self, force_rebuild: bool = False):
        """Initialize the ChromaDB vector database"""
        if self._initialized and not force_rebuild:
            return
        
        cdb = _load_chromadb()
        
        # Use persistent storage
        db_path = Path("./chroma_db")
        print(f"  Initializing ChromaDB at: {db_path}")
        
        self.client = cdb.PersistentClient(path=str(db_path))
        
        # Get or create collection
        collection_name = "slo_research_corpus"
        embedding_fn = self._get_embedding_function()
        
        try:
            # Try to get existing collection
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=embedding_fn
            )
            doc_count = self.collection.count()
            print(f"  ✓ Loaded existing collection: {doc_count} documents")
            
            # Rebuild if empty or force rebuild
            if doc_count == 0 or force_rebuild:
                self._rebuild_collection(collection_name, embedding_fn)
                
        except Exception:
            # Create new collection
            self._rebuild_collection(collection_name, embedding_fn)
        
        self._initialized = True
    
    def _rebuild_collection(self, collection_name: str, embedding_fn):
        """Rebuild the collection with research documents"""
        print(f"  Building research corpus...")
        
        # Delete if exists
        try:
            self.client.delete_collection(name=collection_name)
        except Exception:
            pass
        
        # Create new collection
        self.collection = self.client.create_collection(
            name=collection_name,
            embedding_function=embedding_fn,
            metadata={"description": "SLO research papers and benchmarks"}
        )
        
        # Index all documents
        documents = get_all_documents()
        
        ids = []
        texts = []
        metadatas = []
        
        for doc in documents:
            ids.append(doc["id"])
            texts.append(doc["text"].strip())
            metadatas.append({
                "source": doc["source"],
                "type": doc["type"],
                "task_types": ",".join(doc.get("task_types", [])),
                "confidence": doc.get("confidence", "medium"),
            })
        
        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas
        )
        
        print(f"  ✓ Indexed {len(documents)} research documents")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for relevant research documents.
        
        Args:
            query: The search query (task description)
            top_k: Number of results to return
            
        Returns:
            List of matching documents with similarity scores
        """
        if not self._initialized:
            self.initialize()
        
        # Query the collection
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted = []
        
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                # Convert distance to similarity (cosine distance to similarity)
                distance = results["distances"][0][i]
                similarity = 1 - (distance / 2) if distance <= 2 else 0
                
                # Get the full document text
                full_text = results["documents"][0][i]
                
                # Extract title (first non-empty line) and first content line (with "-")
                lines = [l.strip() for l in full_text.strip().split('\n') if l.strip()]
                
                # Get the title/header (first line without "-")
                title = ""
                for l in lines:
                    if not l.startswith('-') and ':' in l:
                        title = l.rstrip(':').strip()
                        break
                
                # Get the first actual content line (starts with "- ")
                first_content = ""
                for l in lines:
                    if l.startswith('- '):
                        first_content = l[2:].strip()  # Remove "- " prefix
                        break
                
                # Build preview: Title + First content line
                if title and first_content:
                    chunk_preview = f"{title}: {first_content}"
                elif title:
                    chunk_preview = title
                elif first_content:
                    chunk_preview = first_content
                else:
                    chunk_preview = ' '.join(lines[:2])[:200] if lines else full_text[:200]
                
                if len(chunk_preview) > 200:
                    chunk_preview = chunk_preview[:200] + "..."
                
                formatted.append({
                    "id": doc_id,
                    "paper_name": results["metadatas"][0][i].get("source", "Unknown"),
                    "type": results["metadatas"][0][i].get("type", "unknown"),
                    "chunk_text": chunk_preview,
                    "full_text": full_text.strip(),
                    "similarity": round(similarity, 3),
                    "confidence": results["metadatas"][0][i].get("confidence", "medium"),
                })
        
        return formatted
    
    def get_research_for_task(self, task_description: str) -> Dict:
        """
        Get relevant research for a task description.
        
        Returns a summary with sources and their relevance.
        """
        results = self.search(task_description, top_k=5)
        
        return {
            "query": task_description,
            "sources": results,
            "total_sources": len(results),
        }
