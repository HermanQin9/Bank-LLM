"""RAG system module initialization."""

# Import simplified RAG components (no TensorFlow dependency)
from src.rag_system.simple_vector_store import SimpleVectorStore
from src.rag_system.simple_rag_pipeline import SimpleRAGPipeline

# Advanced RAG components with heavy dependencies (lazy load only when needed)
ADVANCED_RAG_AVAILABLE = False

__all__ = [
    "SimpleVectorStore",
    "SimpleRAGPipeline",
    "ADVANCED_RAG_AVAILABLE",
]
