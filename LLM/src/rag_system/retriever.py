"""Document retriever for RAG system."""

from typing import List, Dict, Optional, Any
from src.rag_system.vector_store import VectorStore
from src.utils import logger


class Retriever:
    """Retriever for finding relevant documents."""
    
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        top_k: int = 5
    ):
        """
        Initialize retriever.
        
        Args:
            vector_store: VectorStore instance
            top_k: Number of top results to retrieve
        """
        self.vector_store = vector_store or VectorStore()
        self.top_k = top_k
        self.logger = logger
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Query text
            top_k: Number of results (overrides default)
            filter_metadata: Optional metadata filter
            
        Returns:
            List of dictionaries containing document info
        """
        k = top_k or self.top_k
        
        self.logger.info(f"Retrieving top {k} documents for query: {query[:50]}...")
        
        results = self.vector_store.search(
            query=query,
            n_results=k,
            where=filter_metadata
        )
        
        # Format results
        documents = []
        for i in range(len(results['documents'][0])):
            documents.append({
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                'distance': results['distances'][0][i] if 'distances' in results else None,
                'id': results['ids'][0][i]
            })
        
        return documents
    
    def retrieve_with_scores(
        self,
        query: str,
        top_k: Optional[int] = None,
        similarity_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents with similarity scores.
        
        Args:
            query: Query text
            top_k: Number of results
            similarity_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of documents with scores above threshold
        """
        documents = self.retrieve(query, top_k)
        
        # Filter by threshold and add similarity score
        filtered_docs = []
        for doc in documents:
            if doc['distance'] is not None:
                # Convert distance to similarity (assuming cosine distance)
                similarity = 1 - doc['distance']
                if similarity >= similarity_threshold:
                    doc['similarity'] = similarity
                    filtered_docs.append(doc)
        
        return filtered_docs
    
    def retrieve_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        max_context_length: Optional[int] = None
    ) -> str:
        """
        Retrieve and concatenate document context.
        
        Args:
            query: Query text
            top_k: Number of documents to retrieve
            max_context_length: Maximum total context length
            
        Returns:
            Concatenated context string
        """
        documents = self.retrieve(query, top_k)
        
        context_parts = []
        total_length = 0
        
        for i, doc in enumerate(documents):
            doc_text = doc['text']
            doc_length = len(doc_text)
            
            if max_context_length and total_length + doc_length > max_context_length:
                # Truncate last document if needed
                remaining = max_context_length - total_length
                if remaining > 100:  # Only add if meaningful amount left
                    context_parts.append(f"[Document {i+1}]\n{doc_text[:remaining]}...")
                break
            
            context_parts.append(f"[Document {i+1}]\n{doc_text}")
            total_length += doc_length
        
        return "\n\n".join(context_parts)
