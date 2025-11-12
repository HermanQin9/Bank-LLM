"""Enhanced RAG pipeline using Gemini embeddings for semantic search."""

from typing import Dict, List, Optional, Any
from src.rag_system.gemini_vector_store import GeminiVectorStore
from src.llm_engine.universal_client import UniversalLLMClient
from src.document_parser.preprocessor import TextPreprocessor
from src.utils import Config, logger


class GeminiRAGPipeline:
    """
    Production RAG pipeline using Google Gemini embeddings.
    
    Features:
    - Semantic search with Gemini embeddings (768-dim)
    - Context-aware answer generation
    - Source tracking and citation
    - Configurable retrieval parameters
    """
    
    def __init__(
        self,
        llm_provider: str = "auto",
        collection_name: str = "documents",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize Gemini RAG pipeline.
        
        Args:
            llm_provider: LLM provider for generation
            collection_name: Name of the vector store collection
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.vector_store = GeminiVectorStore(collection_name=collection_name)
        self.llm_client = UniversalLLMClient(provider=llm_provider)
        self.preprocessor = TextPreprocessor()
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.logger = logger
        self.logger.info("Initialized Gemini RAG Pipeline")
    
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
        self.logger.info(f"Indexing {len(documents)} documents...")
        
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
                        'start_pos': chunk['start_pos'],
                        'source': 'chunked'
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
        
        # Add to vector store (will use Gemini embeddings)
        num_indexed = self.vector_store.add_documents(all_chunks, all_metadatas)
        
        self.logger.info(f"Indexed {num_indexed} chunks from {len(documents)} documents")
        
        return num_indexed
    
    def query(
        self,
        question: str,
        top_k: int = 5,
        include_sources: bool = True,
        score_threshold: float = 0.3
    ) -> Dict[str, Any]:
        """
        Query the RAG system with semantic search.
        
        Args:
            question: User question
            top_k: Number of documents to retrieve
            include_sources: Whether to include source documents
            score_threshold: Minimum similarity score (0-1)
            
        Returns:
            Dictionary with answer and optional sources
        """
        self.logger.info(f"Processing query: {question[:50]}...")
        
        # Retrieve relevant documents using Gemini embeddings
        search_results = self.vector_store.search(
            question,
            top_k=top_k,
            score_threshold=score_threshold
        )
        
        if not search_results:
            return {
                'answer': "I couldn't find relevant information to answer this question.",
                'sources': [],
                'num_sources': 0,
                'query': question
            }
        
        # Build context from retrieved documents
        context_parts = []
        for i, result in enumerate(search_results):
            context_parts.append(f"[Source {i+1}] {result['document']}")
        
        context = "\n\n".join(context_parts)
        
        # Generate answer with context
        prompt = f"""Based on the following context, answer the question. If the context doesn't contain enough information, say so.

Context:
{context}

Question: {question}

Answer:"""
        
        try:
            result = self.llm_client.generate_with_metadata(prompt, temperature=0.3)
            
            response = {
                'answer': result['text'],
                'query': question,
                'num_sources': len(search_results),
                'tokens': {
                    'prompt': result.get('prompt_tokens', 0),
                    'response': result.get('response_tokens', 0),
                    'total': result.get('prompt_tokens', 0) + result.get('response_tokens', 0)
                }
            }
            
            # Add sources if requested
            if include_sources:
                response['sources'] = [
                    {
                        'document': res['document'][:300] + "..." if len(res['document']) > 300 else res['document'],
                        'score': res['score'],
                        'metadata': res['metadata']
                    }
                    for res in search_results
                ]
            
            self.logger.info(f"Generated answer with {len(search_results)} sources")
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating answer: {e}")
            return {
                'answer': f"Error generating answer: {str(e)}",
                'sources': [],
                'num_sources': len(search_results),
                'query': question,
                'error': str(e)
            }
    
    def batch_query(
        self,
        questions: List[str],
        top_k: int = 5
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
        
        for i, question in enumerate(questions):
            self.logger.info(f"Processing question {i+1}/{len(questions)}")
            result = self.query(question, top_k, include_sources=False)
            results.append(result)
        
        return results
    
    def save(self):
        """Save the vector store to disk."""
        self.vector_store.save()
    
    def load(self) -> bool:
        """Load the vector store from disk."""
        return self.vector_store.load()
    
    def clear(self):
        """Clear all indexed documents."""
        self.vector_store.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the RAG system.
        
        Returns:
            Dictionary with system statistics
        """
        vector_stats = self.vector_store.get_stats()
        
        return {
            'vector_store': vector_stats,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'llm_model': self.llm_client.model_name,
            'llm_provider': self.llm_client.provider,
            'embedding_type': 'google-gemini-embeddings'
        }
    
    def semantic_search_only(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search without generation (useful for retrieval testing).
        
        Args:
            query: Search query
            top_k: Number of results
            score_threshold: Minimum score
            
        Returns:
            List of search results
        """
        return self.vector_store.search(query, top_k, score_threshold)
    
    def rerank_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Rerank search results using LLM (advanced feature).
        
        Args:
            query: Original query
            results: Search results to rerank
            top_k: Number of top results to return
            
        Returns:
            Reranked results
        """
        if not results or len(results) <= top_k:
            return results
        
        # Create reranking prompt
        docs_text = "\n\n".join([
            f"Document {i+1}:\n{res['document'][:200]}"
            for i, res in enumerate(results)
        ])
        
        prompt = f"""Given the query and documents below, rank the documents by relevance to the query.
Return ONLY the document numbers in order, separated by commas (e.g., "3,1,5,2,4").

Query: {query}

{docs_text}

Ranking (most relevant first):"""
        
        try:
            ranking_str = self.llm_client.generate(prompt, temperature=0.1)
            
            # Parse ranking
            ranking_indices = [int(x.strip()) - 1 for x in ranking_str.split(',') if x.strip().isdigit()]
            
            # Reorder results
            reranked = []
            for idx in ranking_indices[:top_k]:
                if 0 <= idx < len(results):
                    reranked.append(results[idx])
            
            self.logger.info(f"Reranked {len(reranked)} results")
            
            return reranked
            
        except Exception as e:
            self.logger.error(f"Reranking failed: {e}")
            return results[:top_k]
