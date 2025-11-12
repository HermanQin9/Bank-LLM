"""Simplified RAG pipeline without heavy dependencies."""

from typing import Dict, List, Optional, Any
from src.rag_system.simple_vector_store import SimpleVectorStore
from src.llm_engine import UniversalLLMClient


class SimpleRAGPipeline:
    """
    Lightweight RAG pipeline using simple TF-IDF vector store.
    No TensorFlow/PyTorch dependencies required.
    """
    
    def __init__(
        self,
        llm_provider: str = 'auto',
        collection_name: str = 'documents'
    ):
        """
        Initialize simplified RAG pipeline.
        
        Args:
            llm_provider: LLM provider to use ('auto', 'gemini', 'groq', etc.)
            collection_name: Name for the document collection
        """
        self.vector_store = SimpleVectorStore(collection_name=collection_name)
        self.llm_client = UniversalLLMClient(provider=llm_provider, enable_fallback=True)
        self.chunk_size = 500
        self.chunk_overlap = 50
    
    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk = ' '.join(words[i:i + self.chunk_size])
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    def index_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        chunk_documents: bool = True
    ) -> int:
        """
        Index documents into the vector store.
        
        Args:
            documents: List of document texts
            metadatas: Optional metadata for each document
            chunk_documents: Whether to chunk documents
            
        Returns:
            Number of chunks indexed
        """
        all_chunks = []
        all_metadatas = []
        
        for i, doc in enumerate(documents):
            if chunk_documents:
                # Chunk the document
                chunks = self._chunk_text(doc)
                
                for j, chunk in enumerate(chunks):
                    all_chunks.append(chunk)
                    
                    # Add chunk metadata
                    chunk_metadata = {
                        'doc_id': i,
                        'chunk_id': j,
                        'source': f"doc_{i}_chunk_{j}"
                    }
                    
                    # Merge with user metadata
                    if metadatas and i < len(metadatas):
                        chunk_metadata.update(metadatas[i])
                    
                    all_metadatas.append(chunk_metadata)
            else:
                all_chunks.append(doc)
                
                if metadatas and i < len(metadatas):
                    all_metadatas.append(metadatas[i])
                else:
                    all_metadatas.append({'doc_id': i})
        
        # Add to vector store
        num_added = self.vector_store.add_documents(all_chunks, all_metadatas)
        
        return num_added
    
    def query(
        self,
        question: str,
        top_k: int = 3,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Query the RAG system.
        
        Args:
            question: Question to answer
            top_k: Number of documents to retrieve
            include_sources: Whether to include source documents
            
        Returns:
            Dictionary with answer and optional sources
        """
        # Retrieve relevant documents
        search_results = self.vector_store.search(question, top_k=top_k)
        
        if not search_results:
            return {
                'answer': "I don't have enough information to answer this question.",
                'num_sources': 0,
                'sources': []
            }
        
        # Build context from retrieved documents
        context_parts = []
        for i, result in enumerate(search_results):
            context_parts.append(f"Document {i+1}:\n{result['document']}")
        
        context = "\n\n".join(context_parts)
        
        # Build prompt
        prompt = f"""Based on the following documents, please answer the question.

Documents:
{context}

Question: {question}

Answer: Please provide a clear and concise answer based only on the information in the documents above. If the documents don't contain enough information to answer the question, please say so."""
        
        # Generate answer
        answer = self.llm_client.generate(prompt)
        
        # Prepare response
        response = {
            'answer': answer,
            'num_sources': len(search_results)
        }
        
        if include_sources:
            sources = []
            for result in search_results:
                sources.append({
                    'document': result['document'][:200] + '...' if len(result['document']) > 200 else result['document'],
                    'score': result['score'],
                    'metadata': result['metadata']
                })
            response['sources'] = sources
        
        return response
    
    def clear(self):
        """Clear all indexed documents."""
        self.vector_store.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG system."""
        return self.vector_store.get_stats()
