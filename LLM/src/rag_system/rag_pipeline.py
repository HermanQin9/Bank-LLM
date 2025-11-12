"""Complete RAG pipeline implementation."""

from typing import Dict, List, Optional, Any
from src.rag_system.vector_store import VectorStore
from src.rag_system.retriever import Retriever
from src.rag_system.generator import Generator
from src.document_parser import TextPreprocessor
from src.utils import Config, logger


class RAGPipeline:
    """End-to-end RAG pipeline for document Q&A."""
    
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        retriever: Optional[Retriever] = None,
        generator: Optional[Generator] = None,
        chunk_size: int = None,
        chunk_overlap: int = None
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            vector_store: VectorStore instance
            retriever: Retriever instance
            generator: Generator instance
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.vector_store = vector_store or VectorStore()
        self.retriever = retriever or Retriever(self.vector_store)
        self.generator = generator or Generator()
        self.preprocessor = TextPreprocessor()
        
        self.chunk_size = chunk_size or Config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP
        
        self.logger = logger
        self.logger.info("Initialized RAG pipeline")
    
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
        self.logger.info(f"Indexing {len(documents)} documents")
        
        all_chunks = []
        all_metadatas = []
        
        for i, doc in enumerate(documents):
            if chunk_documents:
                # Chunk the document
                chunks = self.preprocessor.chunk_text(
                    doc,
                    self.chunk_size,
                    self.chunk_overlap
                )
                
                for chunk in chunks:
                    all_chunks.append(chunk['text'])
                    
                    # Add chunk metadata
                    chunk_metadata = {
                        'doc_id': i,
                        'chunk_id': chunk['chunk_id'],
                        'start_pos': chunk['start_pos']
                    }
                    
                    # Merge with document metadata if provided
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
        self.vector_store.add_documents(all_chunks, all_metadatas)
        
        self.logger.info(f"Indexed {len(all_chunks)} chunks from {len(documents)} documents")
        
        return len(all_chunks)
    
    def query(
        self,
        question: str,
        top_k: int = None,
        return_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Query the RAG system.
        
        Args:
            question: User question
            top_k: Number of documents to retrieve
            return_sources: Whether to include source documents
            
        Returns:
            Dictionary with answer and optional sources
        """
        self.logger.info(f"Processing query: {question}")
        
        # Retrieve relevant documents
        documents = self.retriever.retrieve(question, top_k)
        
        # Build context from retrieved documents
        context = self.retriever.retrieve_context(question, top_k)
        
        # Generate answer
        result = self.generator.generate_with_metadata(
            question,
            context,
            include_sources=return_sources
        )
        
        # Add retrieved documents if requested
        if return_sources:
            result['sources'] = [
                {
                    'text': doc['text'][:200] + "...",  # Truncate for display
                    'metadata': doc['metadata'],
                    'similarity': doc.get('distance')
                }
                for doc in documents
            ]
        
        result['num_sources'] = len(documents)
        
        return result
    
    def batch_query(
        self,
        questions: List[str],
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Process multiple queries in batch.
        
        Args:
            questions: List of questions
            top_k: Number of documents to retrieve per question
            
        Returns:
            List of results for each question
        """
        results = []
        
        for question in questions:
            result = self.query(question, top_k, return_sources=False)
            results.append(result)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the RAG system.
        
        Returns:
            Dictionary with system statistics
        """
        return {
            'total_documents': self.vector_store.get_document_count(),
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'embedding_model': self.vector_store.embedding_model_name,
            'llm_model': self.generator.llm_client.model_name
        }
