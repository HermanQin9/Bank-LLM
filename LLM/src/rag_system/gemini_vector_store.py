"""Vector store using Google Gemini embeddings for semantic search."""

import numpy as np
from typing import List, Dict, Optional, Any
import json
import os
from pathlib import Path
import requests
from src.utils import Config, logger


class GeminiVectorStore:
    """
    Production-grade vector store using Google Gemini embeddings.
    
    Features:
    - High-quality semantic embeddings (768 dimensions)
    - Cosine similarity search
    - Persistent storage
    - Batch embedding support
    """
    
    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """
        Initialize Gemini vector store.
        
        Args:
            collection_name: Name of the collection
            persist_directory: Directory to persist data
            api_key: Google API key (uses Config.GOOGLE_API_KEY if not provided)
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory or "data/vector_store"
        self.api_key = api_key or Config.GOOGLE_API_KEY
        
        if not self.api_key:
            raise ValueError("Google API key is required. Set GOOGLE_API_KEY in .env")
        
        # Create directory
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        self.documents = []
        self.metadatas = []
        self.embeddings = []
        
        self.embedding_model = "models/embedding-001"
        self.embedding_dim = 768
        
        self.logger = logger
        self.logger.info(f"Initialized Gemini Vector Store: {collection_name}")
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding from Gemini API.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        url = f"https://generativelanguage.googleapis.com/v1beta/{self.embedding_model}:embedContent"
        
        headers = {
            "Content-Type": "application/json"
        }
        
        params = {
            "key": self.api_key
        }
        
        data = {
            "model": self.embedding_model,
            "content": {
                "parts": [{"text": text}]
            }
        }
        
        try:
            response = requests.post(url, headers=headers, params=params, json=data, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            embedding = result["embedding"]["values"]
            
            return np.array(embedding)
            
        except Exception as e:
            self.logger.error(f"Error getting embedding: {e}")
            # Return zero vector as fallback
            return np.zeros(self.embedding_dim)
    
    def _get_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> List[np.ndarray]:
        """
        Get embeddings for multiple texts in batches.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for API calls
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            self.logger.info(f"Embedding batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            
            for text in batch:
                embedding = self._get_embedding(text)
                embeddings.append(embedding)
        
        return embeddings
    
    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> int:
        """
        Add documents to the store.
        
        Args:
            documents: List of document texts
            metadatas: Optional metadata for each document
            
        Returns:
            Number of documents added
        """
        if not documents:
            return 0
        
        self.logger.info(f"Adding {len(documents)} documents...")
        
        # Get embeddings from Gemini
        new_embeddings = self._get_embeddings_batch(documents)
        
        # Add to store
        self.documents.extend(documents)
        self.embeddings.extend(new_embeddings)
        
        if metadatas:
            self.metadatas.extend(metadatas)
        else:
            self.metadatas.extend([{}] * len(documents))
        
        self.logger.info(f"Successfully added {len(documents)} documents")
        
        return len(documents)
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents using semantic similarity.
        
        Args:
            query: Query text
            top_k: Number of results to return
            score_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of results with documents, scores, and metadata
        """
        if not self.documents:
            return []
        
        self.logger.info(f"Searching for: {query[:50]}...")
        
        # Get query embedding
        query_vector = self._get_embedding(query)
        
        # Compute cosine similarity with all documents
        similarities = []
        for i, doc_vector in enumerate(self.embeddings):
            # Cosine similarity
            similarity = np.dot(query_vector, doc_vector) / (
                np.linalg.norm(query_vector) * np.linalg.norm(doc_vector) + 1e-8
            )
            similarities.append((similarity, i))
        
        # Sort by similarity
        similarities.sort(reverse=True, key=lambda x: x[0])
        
        # Get top k results above threshold
        results = []
        for score, idx in similarities[:top_k]:
            if score >= score_threshold:
                results.append({
                    'document': self.documents[idx],
                    'score': float(score),
                    'metadata': self.metadatas[idx],
                    'index': idx
                })
        
        self.logger.info(f"Found {len(results)} relevant documents")
        
        return results
    
    def save(self):
        """Save the vector store to disk."""
        self.logger.info("Saving vector store...")
        
        # Convert embeddings to list for JSON serialization
        embeddings_list = [emb.tolist() for emb in self.embeddings]
        
        data = {
            'documents': self.documents,
            'metadatas': self.metadatas,
            'embeddings': embeddings_list,
            'embedding_model': self.embedding_model,
            'embedding_dim': self.embedding_dim
        }
        
        filepath = os.path.join(self.persist_directory, f"{self.collection_name}_gemini.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Saved to {filepath}")
    
    def load(self) -> bool:
        """Load the vector store from disk."""
        filepath = os.path.join(self.persist_directory, f"{self.collection_name}_gemini.json")
        
        if not os.path.exists(filepath):
            self.logger.warning(f"No saved data found at {filepath}")
            return False
        
        self.logger.info(f"Loading vector store from {filepath}...")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.documents = data['documents']
        self.metadatas = data['metadatas']
        
        # Convert embeddings back to numpy arrays
        self.embeddings = [np.array(emb) for emb in data['embeddings']]
        
        self.embedding_model = data.get('embedding_model', self.embedding_model)
        self.embedding_dim = data.get('embedding_dim', self.embedding_dim)
        
        self.logger.info(f"Loaded {len(self.documents)} documents")
        
        return True
    
    def clear(self):
        """Clear all documents."""
        self.documents = []
        self.metadatas = []
        self.embeddings = []
        self.logger.info("Cleared vector store")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        return {
            'num_documents': len(self.documents),
            'embedding_model': self.embedding_model,
            'embedding_dim': self.embedding_dim,
            'collection_name': self.collection_name,
            'total_size_mb': sum(emb.nbytes for emb in self.embeddings) / (1024 * 1024)
        }
    
    def get_document_by_index(self, index: int) -> Optional[Dict[str, Any]]:
        """
        Get document by index.
        
        Args:
            index: Document index
            
        Returns:
            Document with metadata
        """
        if 0 <= index < len(self.documents):
            return {
                'document': self.documents[index],
                'metadata': self.metadatas[index],
                'embedding_shape': self.embeddings[index].shape
            }
        return None
    
    def delete_documents(self, indices: List[int]):
        """
        Delete documents by indices.
        
        Args:
            indices: List of document indices to delete
        """
        # Sort in reverse to avoid index shifting
        for idx in sorted(indices, reverse=True):
            if 0 <= idx < len(self.documents):
                del self.documents[idx]
                del self.metadatas[idx]
                del self.embeddings[idx]
        
        self.logger.info(f"Deleted {len(indices)} documents")
    
    def update_document(self, index: int, new_text: str, new_metadata: Optional[Dict[str, Any]] = None):
        """
        Update a document.
        
        Args:
            index: Document index
            new_text: New document text
            new_metadata: New metadata (optional)
        """
        if 0 <= index < len(self.documents):
            # Get new embedding
            new_embedding = self._get_embedding(new_text)
            
            # Update
            self.documents[index] = new_text
            self.embeddings[index] = new_embedding
            
            if new_metadata is not None:
                self.metadatas[index] = new_metadata
            
            self.logger.info(f"Updated document at index {index}")
        else:
            self.logger.error(f"Invalid index: {index}")
