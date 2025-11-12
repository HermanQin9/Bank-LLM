"""Simplified vector store without heavy dependencies."""

import numpy as np
from typing import List, Dict, Optional, Any
import json
import os
from pathlib import Path


class SimpleVectorStore:
    """
    Lightweight vector store using simple TF-IDF and cosine similarity.
    No TensorFlow/PyTorch dependencies required.
    """
    
    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: Optional[str] = None
    ):
        """Initialize simple vector store."""
        self.collection_name = collection_name
        self.persist_directory = persist_directory or "data/vector_store"
        
        # Create directory
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        self.documents = []
        self.metadatas = []
        self.embeddings = []
        self.vocab = {}
        self.idf = {}
        
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        # Convert to lowercase and split by whitespace
        text = text.lower()
        # Remove punctuation
        import string
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text.split()
    
    def _compute_tf(self, tokens: List[str]) -> Dict[str, float]:
        """Compute term frequency."""
        tf = {}
        total = len(tokens)
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1
        # Normalize
        for token in tf:
            tf[token] = tf[token] / total
        return tf
    
    def _compute_idf(self):
        """Compute inverse document frequency."""
        n_docs = len(self.documents)
        if n_docs == 0:
            return
        
        # Count document frequency
        df = {}
        for doc in self.documents:
            tokens = set(self._tokenize(doc))
            for token in tokens:
                df[token] = df.get(token, 0) + 1
        
        # Compute IDF
        self.idf = {}
        for token, freq in df.items():
            self.idf[token] = np.log(n_docs / freq)
    
    def _compute_tfidf_vector(self, text: str) -> np.ndarray:
        """Compute TF-IDF vector for text."""
        tokens = self._tokenize(text)
        tf = self._compute_tf(tokens)
        
        # Create vector
        vector = np.zeros(len(self.vocab))
        for token, tf_value in tf.items():
            if token in self.vocab:
                idx = self.vocab[token]
                idf_value = self.idf.get(token, 0)
                vector[idx] = tf_value * idf_value
        
        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector
    
    def _build_vocab(self):
        """Build vocabulary from all documents."""
        self.vocab = {}
        idx = 0
        for doc in self.documents:
            tokens = self._tokenize(doc)
            for token in tokens:
                if token not in self.vocab:
                    self.vocab[token] = idx
                    idx += 1
    
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
        
        self.documents.extend(documents)
        
        if metadatas:
            self.metadatas.extend(metadatas)
        else:
            self.metadatas.extend([{}] * len(documents))
        
        # Rebuild vocabulary and embeddings
        self._build_vocab()
        self._compute_idf()
        
        # Compute embeddings for all documents
        self.embeddings = []
        for doc in self.documents:
            vector = self._compute_tfidf_vector(doc)
            self.embeddings.append(vector)
        
        return len(documents)
    
    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query: Query text
            top_k: Number of results to return
            
        Returns:
            List of results with documents, scores, and metadata
        """
        if not self.documents:
            return []
        
        # Compute query vector
        query_vector = self._compute_tfidf_vector(query)
        
        # Compute cosine similarity with all documents
        similarities = []
        for i, doc_vector in enumerate(self.embeddings):
            similarity = np.dot(query_vector, doc_vector)
            similarities.append((similarity, i))
        
        # Sort by similarity
        similarities.sort(reverse=True, key=lambda x: x[0])
        
        # Get top k results
        results = []
        for score, idx in similarities[:top_k]:
            if score > 0:  # Only return relevant results
                results.append({
                    'document': self.documents[idx],
                    'score': float(score),
                    'metadata': self.metadatas[idx],
                    'index': idx
                })
        
        return results
    
    def save(self):
        """Save the vector store to disk."""
        data = {
            'documents': self.documents,
            'metadatas': self.metadatas,
            'vocab': self.vocab,
            'idf': self.idf
        }
        
        filepath = os.path.join(self.persist_directory, f"{self.collection_name}.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self):
        """Load the vector store from disk."""
        filepath = os.path.join(self.persist_directory, f"{self.collection_name}.json")
        
        if not os.path.exists(filepath):
            return False
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.documents = data['documents']
        self.metadatas = data['metadatas']
        self.vocab = data['vocab']
        self.idf = data['idf']
        
        # Rebuild embeddings
        self.embeddings = []
        for doc in self.documents:
            vector = self._compute_tfidf_vector(doc)
            self.embeddings.append(vector)
        
        return True
    
    def clear(self):
        """Clear all documents."""
        self.documents = []
        self.metadatas = []
        self.embeddings = []
        self.vocab = {}
        self.idf = {}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        return {
            'num_documents': len(self.documents),
            'vocab_size': len(self.vocab),
            'collection_name': self.collection_name
        }
