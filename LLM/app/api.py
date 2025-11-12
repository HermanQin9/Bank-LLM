"""FastAPI REST API for Document Intelligence System."""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.document_parser import PDFParser, TextPreprocessor
from src.llm_engine import GeminiClient, PromptTemplates
from src.rag_system.gemini_rag_pipeline import GeminiRAGPipeline

# Initialize FastAPI app
app = FastAPI(
    title="Document Intelligence API",
    description="API for document understanding using Google Gemini",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
client = GeminiClient()
rag_pipeline = GeminiRAGPipeline()
pdf_parser = PDFParser()
preprocessor = TextPreprocessor()


# Request/Response models
class ClassificationRequest(BaseModel):
    text: str
    categories: List[str]


class ClassificationResponse(BaseModel):
    classification: str
    confidence: Optional[float] = None
    tokens_used: int


class ExtractionRequest(BaseModel):
    text: str
    fields: List[str]


class ExtractionResponse(BaseModel):
    extracted_data: Dict[str, str]
    tokens_used: int


class EntityRequest(BaseModel):
    text: str
    entity_types: Optional[List[str]] = None


class EntityResponse(BaseModel):
    entities: str
    tokens_used: int


class IndexDocumentsRequest(BaseModel):
    documents: List[str]
    metadatas: Optional[List[Dict[str, Any]]] = None


class IndexDocumentsResponse(BaseModel):
    num_indexed: int
    message: str


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class QueryResponse(BaseModel):
    answer: str
    sources: Optional[List[Dict[str, Any]]] = None
    tokens_used: int


# API endpoints
@app.get("/")
def read_root():
    """Root endpoint."""
    return {
        "message": "Document Intelligence API with Gemini",
        "version": "1.0.0",
        "endpoints": {
            "classification": "/classify",
            "extraction": "/extract",
            "entities": "/entities",
            "rag_index": "/rag/index",
            "rag_query": "/rag/query",
            "health": "/health"
        }
    }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model": client.model_name}


@app.post("/classify", response_model=ClassificationResponse)
def classify_document(request: ClassificationRequest):
    """Classify a document into categories."""
    try:
        prompt = PromptTemplates.document_classification(
            request.text,
            request.categories
        )
        result = client.generate_with_metadata(prompt)
        
        return ClassificationResponse(
            classification=result['text'].strip(),
            tokens_used=result.get('prompt_tokens', 0) + result.get('response_tokens', 0)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract", response_model=ExtractionResponse)
def extract_information(request: ExtractionRequest):
    """Extract structured information from document."""
    try:
        prompt = PromptTemplates.information_extraction(
            request.text,
            request.fields
        )
        result = client.generate_with_metadata(prompt)
        
        # Parse extracted information
        extracted = {}
        for line in result['text'].split('\n'):
            for field in request.fields:
                if line.startswith(f"{field}:"):
                    value = line.replace(f"{field}:", "").strip()
                    extracted[field] = value
                    break
        
        return ExtractionResponse(
            extracted_data=extracted,
            tokens_used=result.get('prompt_tokens', 0) + result.get('response_tokens', 0)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/entities", response_model=EntityResponse)
def extract_entities(request: EntityRequest):
    """Extract named entities from text."""
    try:
        prompt = PromptTemplates.entity_extraction(
            request.text,
            request.entity_types
        )
        result = client.generate_with_metadata(prompt)
        
        return EntityResponse(
            entities=result['text'],
            tokens_used=result.get('prompt_tokens', 0) + result.get('response_tokens', 0)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/index", response_model=IndexDocumentsResponse)
def index_documents(request: IndexDocumentsRequest):
    """Index documents for RAG system."""
    try:
        num_indexed = rag_pipeline.index_documents(
            request.documents,
            request.metadatas
        )
        
        return IndexDocumentsResponse(
            num_indexed=num_indexed,
            message=f"Successfully indexed {num_indexed} document chunks"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/query", response_model=QueryResponse)
def query_documents(request: QueryRequest):
    """Query RAG system."""
    try:
        result = rag_pipeline.query(
            request.question,
            top_k=request.top_k,
            return_sources=True
        )
        
        return QueryResponse(
            answer=result['answer'],
            sources=result.get('sources'),
            tokens_used=result['tokens']['prompt'] + result['tokens']['response']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/rag/stats")
def get_rag_stats():
    """Get RAG system statistics."""
    try:
        stats = rag_pipeline.get_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
