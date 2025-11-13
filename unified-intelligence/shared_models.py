"""
Shared Data Models - Single Source of Truth
============================================

These models are used by BOTH Java and Python systems.
Java uses Jackson for serialization, Python uses Pydantic.
Schema defined here ensures consistency across the stack.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    """Shared risk classification - used in DB, Java, Python, LLM prompts"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class DetectionMethod(str, Enum):
    """How fraud was detected - tracks integration effectiveness"""
    RULE_BASED = "RULE_BASED"              # Pure Java/Scala rules
    ML_MODEL = "ML_MODEL"                  # PyTorch deep learning
    LLM_ANALYSIS = "LLM_ANALYSIS"          # Pure LLM reasoning
    HYBRID_RULE_ML = "HYBRID_RULE_ML"      # Rules + ML ensemble
    HYBRID_RULE_LLM = "HYBRID_RULE_LLM"    # Rules + LLM (current)
    UNIFIED_INTELLIGENCE = "UNIFIED_INTELLIGENCE"  # All three combined


class Transaction(BaseModel):
    """
    Core transaction model shared across ALL systems
    - Java ETL writes to DB
    - Python ML reads for feature engineering
    - LLM uses for context in analysis
    """
    transaction_id: str
    customer_id: str
    amount: float
    merchant_name: str
    merchant_category: Optional[str] = None
    transaction_date: datetime
    location: Optional[str] = None
    device_id: Optional[str] = None
    
    # Derived fields (populated by different systems)
    fraud_score: Optional[float] = None  # Scala rules or Python ML
    risk_level: Optional[RiskLevel] = None
    detection_method: Optional[DetectionMethod] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "transaction_id": "TXN_20251113_001",
                "customer_id": "CUST_12345",
                "amount": 8500.00,
                "merchant_name": "Overseas Electronics Ltd",
                "transaction_date": "2025-11-13T15:30:00Z",
                "fraud_score": 0.85,
                "risk_level": "HIGH"
            }
        }


class CustomerProfile(BaseModel):
    """
    Customer intelligence aggregated from multiple sources:
    - Java: Transaction history statistics
    - Python: ML-derived behavior patterns
    - LLM: Extracted from KYC documents, emails, support tickets
    """
    customer_id: str
    
    # Traditional banking data (Java source)
    account_open_date: Optional[datetime] = None
    account_type: Optional[str] = None
    credit_limit: Optional[float] = None
    
    # Statistical features (computed by Java/Scala)
    avg_transaction_amount: Optional[float] = None
    transaction_count_30d: Optional[int] = None
    max_transaction_amount: Optional[float] = None
    
    # ML-derived features (Python source)
    behavior_cluster: Optional[int] = None  # K-means clustering
    anomaly_score: Optional[float] = None   # Isolation forest
    spending_pattern: Optional[str] = None  # "consistent", "erratic", "seasonal"
    
    # LLM-extracted features (document intelligence)
    occupation: Optional[str] = None
    income_bracket: Optional[str] = None
    risk_tolerance: Optional[str] = None
    expected_transaction_types: Optional[List[str]] = None
    kyc_summary: Optional[str] = None  # LLM-generated summary from documents
    
    # Unified risk assessment (combines all sources)
    unified_risk_score: Optional[float] = None
    last_updated: datetime = Field(default_factory=datetime.now)


class DocumentEvidence(BaseModel):
    """
    Links unstructured documents to structured transactions
    Enables "show me the evidence" for any fraud alert
    """
    evidence_id: str
    transaction_id: str
    customer_id: str
    
    # Document metadata
    document_source: str  # "email", "kyc_form", "support_ticket", "compliance_report"
    document_type: str
    document_path: Optional[str] = None
    
    # LLM processing results
    extracted_text: str
    key_entities: Dict[str, Any]  # NER results
    sentiment: Optional[str] = None
    risk_indicators: List[str] = []
    
    # Linking metadata
    relevance_score: float  # From RAG semantic search
    llm_reasoning: str  # Why this document is relevant
    
    created_at: datetime = Field(default_factory=datetime.now)


class FraudAlert(BaseModel):
    """
    Unified alert combining ALL detection methods
    Single object flows through Java → Python → LLM → back to Java
    """
    alert_id: str
    transaction_id: str
    customer_id: str
    
    # Multi-source scoring
    rule_based_score: Optional[float] = None  # Scala functional rules
    ml_model_score: Optional[float] = None    # PyTorch neural network
    llm_risk_score: Optional[float] = None    # LLM analysis
    
    # Final unified score (weighted ensemble)
    final_risk_score: float
    risk_level: RiskLevel
    detection_method: DetectionMethod
    
    # Rich context from all systems
    rules_triggered: List[str] = []  # From Scala
    ml_features: Dict[str, float] = {}  # Feature importance from PyTorch
    llm_reasoning: str = ""  # Natural language explanation
    supporting_documents: List[DocumentEvidence] = []  # From RAG
    
    # Workflow tracking
    status: str = "PENDING"  # PENDING, UNDER_REVIEW, RESOLVED, FALSE_POSITIVE
    assigned_to: Optional[str] = None
    resolution_notes: Optional[str] = None
    
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class InvestigationWorkflow(BaseModel):
    """
    Multi-step investigation orchestrated across Java and Python
    Tracks state as it moves through different systems
    """
    workflow_id: str
    alert_id: str
    customer_id: str
    
    # Workflow steps (each can be handled by different system)
    steps: List[Dict[str, Any]] = []
    # Example steps:
    # 1. Java: Pull transaction history from DB
    # 2. Python ML: Generate risk features
    # 3. Python RAG: Search for relevant documents
    # 4. LLM: Analyze combined context
    # 5. Java: Update alert status in DB
    # 6. Python: Generate compliance report if needed
    
    current_step: int = 0
    status: str = "IN_PROGRESS"
    
    # Results accumulate as workflow progresses
    investigation_findings: Dict[str, Any] = {}
    
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None


class ComplianceReport(BaseModel):
    """
    Generated by LLM but uses data from Java DB
    Shows bidirectional integration
    """
    report_id: str
    customer_id: str
    report_type: str  # SAR, CTR, CDD
    
    # Data sources used
    transaction_count: int  # From Java DB query
    suspicious_transactions: List[Transaction] = []  # Java → Python
    supporting_documents: List[DocumentEvidence] = []  # From RAG
    
    # LLM-generated content
    executive_summary: str
    detailed_analysis: str
    recommended_action: str
    regulatory_citations: List[str] = []
    
    # Metadata
    generated_by: str = "UNIFIED_SYSTEM"
    generated_at: datetime = Field(default_factory=datetime.now)
    approved_by: Optional[str] = None
    filed_at: Optional[datetime] = None


# Export all models
__all__ = [
    'RiskLevel',
    'DetectionMethod',
    'Transaction',
    'CustomerProfile',
    'DocumentEvidence',
    'FraudAlert',
    'InvestigationWorkflow',
    'ComplianceReport',
]
