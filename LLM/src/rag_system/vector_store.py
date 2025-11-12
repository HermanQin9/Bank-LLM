"""Vector store for document embeddings."""

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except Exception:
    CHROMADB_AVAILABLE = False

import numpy as np
import faiss
from typing import List, Dict, Optional, Any
from sentence_transformers import SentenceTransformer
from src.utils import Config, logger, ensure_dir
import json
import os


class VectorStore:
    """Vector store with support for both ChromaDB and FAISS."""
    
    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: Optional[str] = None,
        embedding_model: Optional[str] = None,
        store_type: str = "auto"  # "chromadb", "faiss", or "auto"
    ):
        """
        Initialize vector store.
        
        Args:
            collection_name: Name of the collection
            persist_directory: Directory to persist the database
            embedding_model: Name of the embedding model
            store_type: Type of vector store ("chromadb", "faiss", or "auto")
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory or Config.VECTOR_DB_PATH
        self.embedding_model_name = embedding_model or Config.EMBEDDING_MODEL
        self.logger = logger
        
        # Determine which store to use
        if store_type == "auto":
            self.store_type = "chromadb" if CHROMADB_AVAILABLE else "faiss"
        else:
            self.store_type = store_type
        
        # Ensure directory exists
        ensure_dir(self.persist_directory)
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize the chosen vector store
        if self.store_type == "chromadb" and CHROMADB_AVAILABLE:
            self._init_chromadb()
        else:
            self._init_faiss()
            
        self.logger.info(f"Initialized VectorStore ({self.store_type}) with collection: {collection_name}")
    
    def _init_chromadb(self):
        """Initialize ChromaDB store."""
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def _init_faiss(self):
        """Initialize FAISS store."""
        self.index_path = os.path.join(self.persist_directory, f"{self.collection_name}.faiss")
        self.metadata_path = os.path.join(self.persist_directory, f"{self.collection_name}_meta.json")
        
        # Load existing index or create new
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                self.metadata_store = json.load(f)
        else:
            # Use IndexFlatL2 for simplicity (L2 distance)
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.metadata_store = {"documents": [], "metadatas": [], "ids": []}
    
    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document texts
            metadatas: Optional metadata for each document
            ids: Optional custom IDs for documents
        """
        if not documents:
            self.logger.warning("No documents to add")
            return
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(documents)
        
        if self.store_type == "chromadb":
            self._add_documents_chromadb(documents, embeddings, metadatas, ids)
        else:
            self._add_documents_faiss(documents, embeddings, metadatas, ids)
        
        self.logger.info(f"Added {len(documents)} documents to vector store")
    
    def _add_documents_chromadb(self, documents, embeddings, metadatas, ids):
        """Add documents to ChromaDB."""
        if ids is None:
            existing_count = self.collection.count()
            ids = [f"doc_{existing_count + i}" for i in range(len(documents))]
        
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
    
    def _add_documents_faiss(self, documents, embeddings, metadatas, ids):
        """Add documents to FAISS."""
        # Generate IDs if not provided
        if ids is None:
            existing_count = len(self.metadata_store["ids"])
            ids = [f"doc_{existing_count + i}" for i in range(len(documents))]
        
        # Add to FAISS index
        embeddings_np = np.array(embeddings).astype('float32')
        self.index.add(embeddings_np)
        
        # Store metadata
        self.metadata_store["documents"].extend(documents)
        self.metadata_store["metadatas"].extend(metadatas or [{}] * len(documents))
        self.metadata_store["ids"].extend(ids)
        
        # Persist
        self._save_faiss()
    
    def _save_faiss(self):
        """Save FAISS index and metadata to disk."""
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata_store, f, ensure_ascii=False, indent=2)
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List]:
        """
        Search for similar documents.
        
        Args:
            query: Query text
            n_results: Number of results to return
            where: Optional metadata filter
            
        Returns:
            Dictionary containing documents, distances, and metadatas
        """
        if self.store_type == "chromadb":
            return self._search_chromadb(query, n_results, where)
        else:
            return self._search_faiss(query, n_results, where)
    
    def _search_chromadb(self, query, n_results, where):
        """Search in ChromaDB."""
        query_embedding = self.embedding_model.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where=where
        )
        return results
    
    def _search_faiss(self, query, n_results, where):
        """Search in FAISS."""
        query_embedding = self.embedding_model.encode([query])
        query_embedding_np = np.array([query_embedding]).astype('float32')
        
        # Search
        distances, indices = self.index.search(query_embedding_np, n_results)
        
        # Format results
        results = {
            "documents": [[]],
            "distances": [[]],
            "metadatas": [[]]
        }
        
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.metadata_store["documents"]):
                results["documents"][0].append(self.metadata_store["documents"][idx])
                results["distances"][0].append(float(dist))
                results["metadatas"][0].append(self.metadata_store["metadatas"][idx])
        
        return results
    
    def delete_documents(self, ids: List[str]) -> None:
        """
        Delete documents by IDs.
        
        Args:
            ids: List of document IDs to delete
        """
        if self.store_type == "chromadb":
            self.collection.delete(ids=ids)
        else:
            # FAISS doesn't support deletion, would need rebuild
            self.logger.warning("FAISS doesn't support deletion. Consider rebuilding index.")
        
        self.logger.info(f"Requested deletion of {len(ids)} documents from vector store")
    
    def get_document_count(self) -> int:
        """Get total number of documents in the collection."""
        if self.store_type == "chromadb":
            return self.collection.count()
        else:
            return len(self.metadata_store["documents"])
    
    def clear(self) -> None:
        """Clear all documents from the collection."""
        if self.store_type == "chromadb":
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        else:
            # Reset FAISS index
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.metadata_store = {"documents": [], "metadatas": [], "ids": []}
            self._save_faiss()
        
        self.logger.info(f"Cleared collection: {self.collection_name}")
