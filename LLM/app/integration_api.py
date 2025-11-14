"""
FastAPI Integration Endpoints
Provides LLM services to Java transaction processing system

Endpoints:
- POST /api/analyze-transaction: Analyze transaction with LLM + document context
- POST /api/search-documents: RAG search for customer documents
- POST /api/generate-report: Generate compliance reports
- GET /health: Health check
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, root_validator
import psycopg2
from psycopg2.extras import RealDictCursor

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_engine.universal_client import UniversalLLMClient
from rag_system.gemini_rag_pipeline import GeminiRAGPipeline

app = FastAPI(title="Financial Intelligence API")

# Initialize LLM and RAG
llm_client = UniversalLLMClient()
rag_pipeline = GeminiRAGPipeline() if Path("data/vector_store").exists() else None

# Database connection config
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'frauddb',
    'user': 'postgres',
    'password': 'postgres'
}


# Request models
def _coerce_aliases(values: Dict, alias_map: Dict[str, List[str]]) -> Dict:
    """Allow camelCase payloads by copying alias values to snake_case keys."""
    for canonical, aliases in alias_map.items():
        if canonical in values:
            continue
        for alias in aliases:
            if alias in values:
                values[canonical] = values.pop(alias)
                break
    return values


class TransactionAnalysisRequest(BaseModel):
    transaction_id: str
    customer_id: str
    amount: float
    merchant_name: str

    @root_validator(pre=True)
    def _support_aliases(cls, values: Dict) -> Dict:
        alias_map = {
            'transaction_id': ['transactionId'],
            'customer_id': ['customerId'],
            'merchant_name': ['merchantName'],
        }
        return _coerce_aliases(values, alias_map)


class DocumentSearchRequest(BaseModel):
    customer_id: str
    query: str
    top_k: int = 5

    @root_validator(pre=True)
    def _support_aliases(cls, values: Dict) -> Dict:
        alias_map = {
            'customer_id': ['customerId'],
            'top_k': ['topK'],
        }
        return _coerce_aliases(values, alias_map)


class ReportGenerationRequest(BaseModel):
    customer_id: str
    report_type: str  # SAR, CTR, CDD

    @root_validator(pre=True)
    def _support_aliases(cls, values: Dict) -> Dict:
        alias_map = {
            'customer_id': ['customerId'],
            'report_type': ['reportType'],
        }
        return _coerce_aliases(values, alias_map)


def get_db_connection():
    """Get PostgreSQL connection"""
    return psycopg2.connect(**DB_CONFIG)


def get_customer_profile(customer_id: str) -> Optional[Dict]:
    """Fetch customer profile from database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("""
            SELECT * FROM customer_profiles 
            WHERE customer_id = %s
        """, (customer_id,))
        
        profile = cursor.fetchone()
        cursor.close()
        conn.close()
        
        return dict(profile) if profile else None
    except Exception as e:
        print(f"Error fetching customer profile: {e}")
        return None


def get_transaction_history(customer_id: str, limit: int = 10) -> List[Dict]:
    """Fetch recent transaction history"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("""
            SELECT transaction_id, amount, merchant_name, transaction_date
            FROM transactions 
            WHERE customer_id = %s
            ORDER BY transaction_date DESC
            LIMIT %s
        """, (customer_id, limit))
        
        transactions = cursor.fetchall()
        cursor.close()
        conn.close()
        
        return [dict(tx) for tx in transactions]
    except Exception as e:
        print(f"Error fetching transaction history: {e}")
        return []


@app.post("/api/analyze-transaction")
async def analyze_transaction(request: TransactionAnalysisRequest):
    """
    Analyze transaction using LLM with document context
    Called by Java fraud detection service
    """
    try:
        # Get customer profile from database
        profile = get_customer_profile(request.customer_id)
        
        # Get transaction history for context
        history = get_transaction_history(request.customer_id, limit=5)
        
        # Calculate statistical baseline
        if history:
            avg_amount = sum(tx['amount'] for tx in history) / len(history)
            deviation = (request.amount - avg_amount) / avg_amount if avg_amount > 0 else 0
        else:
            deviation = 0
        
        # Search for relevant documents via RAG
        supporting_docs = []
        if rag_pipeline:
            search_query = f"customer {request.customer_id} merchant {request.merchant_name} transaction {request.amount}"
            rag_results = rag_pipeline.semantic_search_only(search_query, top_k=3)
            supporting_docs = [
                {
                    'source': doc.get('metadata', {}).get('source', 'Unknown'),
                    'excerpt': doc['document'][:200] + '...',
                    'relevance_score': doc.get('score', 0)
                }
                for doc in rag_results
            ]
        
        # LLM analysis prompt
        prompt = f"""Analyze this financial transaction for fraud risk:

Transaction Details:
- ID: {request.transaction_id}
- Customer: {request.customer_id}
- Amount: ${request.amount:,.2f}
- Merchant: {request.merchant_name}

Customer Profile:
{profile if profile else "No profile found"}

Transaction History (last 5):
{history}

Statistical Analysis:
- Deviation from average: {deviation:.1%}

Supporting Documents:
{supporting_docs}

Provide:
1. Risk score (0.0 to 1.0)
2. Reasoning for the score
3. Recommended action

Format response as JSON:
{{
    "risk_score": <float>,
    "reasoning": "<explanation>",
    "recommended_action": "<action>",
    "key_risk_factors": ["factor1", "factor2"]
}}
"""
        
        # Get LLM analysis
        llm_response = llm_client.generate(prompt, temperature=0.3, max_tokens=500)
        
        # Parse LLM response (expecting JSON)
        import json
        try:
            analysis = json.loads(llm_response)
        except:
            # Fallback if LLM doesn't return valid JSON
            analysis = {
                'risk_score': 0.5,
                'reasoning': llm_response,
                'recommended_action': 'MANUAL_REVIEW',
                'key_risk_factors': []
            }
        
        # Add complete context to response
        analysis['transaction_id'] = request.transaction_id
        analysis['supporting_documents'] = supporting_docs
        analysis['profile'] = profile
        analysis['deviation_from_average'] = deviation
        analysis['history_count'] = len(history)
        
        return analysis
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/api/search-documents")
async def search_documents(request: DocumentSearchRequest):
    """
    Search customer documents using RAG
    Called by Java services for document evidence retrieval
    """
    try:
        if not rag_pipeline:
            raise HTTPException(status_code=503, detail="RAG system not initialized")
        
        # Perform semantic search
        results = rag_pipeline.semantic_search_only(request.query, top_k=request.top_k)
        
        # Format results
        documents = [
            {
                'source': doc.get('metadata', {}).get('source', 'Unknown'),
                'content': doc['document'],
                'relevance_score': doc.get('score', 0),
                'metadata': doc.get('metadata', {})
            }
            for doc in results
        ]
        
        return {
            'customer_id': request.customer_id,
            'query': request.query,
            'document_count': len(documents),
            'documents': documents
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/api/generate-report")
async def generate_report(request: ReportGenerationRequest):
    """
    Generate compliance report combining transaction data + document analysis
    Called by Java compliance system
    """
    try:
        # Get suspicious transactions from database
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("""
            SELECT t.*, ta.risk_score, ta.risk_level, ta.analysis_details
            FROM transactions t
            LEFT JOIN transaction_alerts ta ON t.transaction_id = ta.transaction_id
            WHERE t.customer_id = %s 
              AND ta.risk_level IN ('HIGH', 'CRITICAL')
            ORDER BY t.transaction_date DESC
            LIMIT 20
        """, (request.customer_id,))
        
        suspicious_txns = [dict(tx) for tx in cursor.fetchall()]
        cursor.close()
        conn.close()
        
        # Get customer profile
        profile = get_customer_profile(request.customer_id)
        
        # Search for related documents
        doc_results = []
        if rag_pipeline:
            search_query = f"customer {request.customer_id} suspicious activity compliance"
            doc_results = rag_pipeline.semantic_search_only(search_query, top_k=5)
        
        # Generate report with LLM
        prompt = f"""Generate a {request.report_type} (Suspicious Activity Report) for regulatory filing:

Customer Profile:
{profile}

Suspicious Transactions ({len(suspicious_txns)} total):
{suspicious_txns[:10]}  # Include first 10

Total Suspicious Amount: ${sum(tx['amount'] for tx in suspicious_txns):,.2f}

Supporting Documents:
{[doc['document'][:200] for doc in doc_results]}

Generate a professional compliance report with:
1. Executive Summary
2. Customer Information
3. Suspicious Activity Description
4. Pattern Analysis
5. Supporting Documentation
6. Recommended Action

Format professionally for regulatory submission.
"""
        
        report_content = llm_client.generate(prompt, temperature=0.2, max_tokens=2000)
        
        # Store report in database
        conn = get_db_connection()
        cursor = conn.cursor()
        
        report_id = f"{request.report_type}_{request.customer_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        cursor.execute("""
            INSERT INTO compliance_reports 
            (report_id, customer_id, report_type, suspicious_transaction_count, 
             total_suspicious_amount, report_content, generated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            report_id,
            request.customer_id,
            request.report_type,
            len(suspicious_txns),
            sum(tx['amount'] for tx in suspicious_txns),
            report_content,
            datetime.now()
        ))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return {
            'report_id': report_id,
            'report_type': request.report_type,
            'customer_id': request.customer_id,
            'report_content': report_content,
            'transaction_count': len(suspicious_txns),
            'generated_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint for Java service"""
    return {
        'status': 'healthy',
        'llm_available': llm_client is not None,
        'rag_available': rag_pipeline is not None,
        'timestamp': datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
