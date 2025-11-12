"""
Fraud Investigation Multi-Agent Workflow

This module integrates transaction analysis (Java/Scala) with document intelligence (LLM)
to automate fraud investigations. Real-world use case: When a high-risk transaction is
flagged, the system automatically gathers evidence from multiple sources and generates
a comprehensive investigation report.

Architecture:
    Transaction Alert (PostgreSQL) 
        → Agent 1: Scala Rule Engine (existing Java service)
        → Agent 2: ML Risk Scorer (PyTorch model)
        → Agent 3: Document Evidence Collector (RAG system)
        → Agent 4: Report Generator (LLM reasoning)

Author: Herman Qin
Date: November 2025
"""

import os
import sys
import asyncio
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'LLM', 'src'))

# LangGraph imports (optional, will use fallback if not available)
try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    logging.warning("LangGraph not installed. Using simplified workflow.")

# Local imports
from llm_engine.universal_client import UniversalLLMClient
from rag_system.gemini_rag_pipeline import GeminiRAGPipeline

# Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Transaction:
    """Transaction data model matching Java Transaction class."""
    transaction_id: str
    customer_id: str
    amount: float
    merchant_name: str
    merchant_category: str
    transaction_date: str
    location: str
    raw_data: Dict[str, Any]


@dataclass
class InvestigationState:
    """State object for investigation workflow."""
    # Input
    transaction: Optional[Transaction] = None
    
    # Agent 1: Rule engine results
    rule_violations: List[str] = None
    rule_score: float = 0.0
    
    # Agent 2: ML model results
    ml_fraud_probability: float = 0.0
    ml_confidence: float = 0.0
    
    # Agent 3: Document evidence
    related_documents: List[Dict] = None
    document_summaries: List[str] = None
    
    # Agent 4: Final assessment
    overall_risk_score: float = 0.0
    risk_level: str = "UNKNOWN"
    investigation_summary: str = ""
    recommended_actions: List[str] = None
    
    # Workflow metadata
    investigation_id: str = ""
    completed: bool = False
    error: Optional[str] = None
    
    def __post_init__(self):
        """Initialize empty lists."""
        if self.rule_violations is None:
            self.rule_violations = []
        if self.related_documents is None:
            self.related_documents = []
        if self.document_summaries is None:
            self.document_summaries = []
        if self.recommended_actions is None:
            self.recommended_actions = []
        if not self.investigation_id:
            self.investigation_id = f"INV-{datetime.now().strftime('%Y%m%d%H%M%S')}"


class RuleEngineAgent:
    """
    Agent 1: Connects to Scala-based rule engine (existing Java service).
    
    In production, this would make HTTP requests to the Java API endpoint.
    For demo, we simulate the Scala rule engine logic.
    """
    
    def __init__(self, java_service_url: str = "http://localhost:8080/api/fraud/analyze"):
        self.service_url = java_service_url
        logger.info("Initialized Rule Engine Agent")
    
    async def analyze(self, state: InvestigationState) -> InvestigationState:
        """Run rule-based fraud detection."""
        logger.info(f"🔍 Rule Engine Agent analyzing transaction {state.transaction.transaction_id}")
        
        tx = state.transaction
        violations = []
        score = 0.0
        
        # Simulate Scala rule engine (in production: HTTP call to Java service)
        # Rule 1: High value transaction
        if tx.amount > 5000:
            violations.append("HIGH_VALUE_TRANSACTION")
            score += 25.0
            logger.info(f"   ⚠️  High value: ${tx.amount:.2f}")
        
        # Rule 2: Unusual merchant category
        risky_categories = ["GAMBLING", "WIRE_TRANSFER", "MONEY_ORDER", "CRYPTOCURRENCY"]
        if tx.merchant_category.upper() in risky_categories:
            violations.append("RISKY_CATEGORY")
            score += 20.0
            logger.info(f"   ⚠️  Risky category: {tx.merchant_category}")
        
        # Rule 3: Foreign transaction
        if "FOREIGN" in tx.location.upper() or "INTERNATIONAL" in tx.location.upper():
            violations.append("FOREIGN_TRANSACTION")
            score += 15.0
            logger.info(f"   ⚠️  Foreign location: {tx.location}")
        
        # Rule 4: New merchant (simulate with "Unknown" in name)
        if "UNKNOWN" in tx.merchant_name.upper() or "UNREGISTERED" in tx.merchant_name.upper():
            violations.append("NEW_MERCHANT")
            score += 10.0
            logger.info(f"   ⚠️  New merchant: {tx.merchant_name}")
        
        # Rule 5: Time-based (from transaction_date string)
        try:
            tx_time = datetime.fromisoformat(tx.transaction_date.replace('Z', '+00:00'))
            if 2 <= tx_time.hour <= 5:
                violations.append("UNUSUAL_TIME")
                score += 15.0
                logger.info(f"   ⚠️  Unusual hour: {tx_time.hour}:00")
        except Exception as e:
            logger.warning(f"   Could not parse transaction time: {e}")
        
        state.rule_violations = violations
        state.rule_score = min(score, 100.0)
        
        logger.info(f"   ✅ Rule engine score: {state.rule_score:.1f}%, violations: {len(violations)}")
        return state


class MLRiskScorerAgent:
    """
    Agent 2: Machine learning risk scoring using PyTorch model.
    
    In production, this loads the trained fraud detection model.
    For demo, we use a simplified heuristic-based approach.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None  # In production: load PyTorch model
        logger.info("Initialized ML Risk Scorer Agent")
    
    async def analyze(self, state: InvestigationState) -> InvestigationState:
        """Generate ML-based risk score."""
        logger.info(f"🤖 ML Agent analyzing transaction {state.transaction.transaction_id}")
        
        tx = state.transaction
        
        # Simulate ML model inference
        # In production: generate embeddings + run through trained PyTorch model
        
        # Features that increase fraud probability
        fraud_indicators = 0
        
        # Amount-based features
        if tx.amount > 10000:
            fraud_indicators += 3
        elif tx.amount > 5000:
            fraud_indicators += 2
        elif tx.amount > 2000:
            fraud_indicators += 1
        
        # Categorical features
        if any(word in tx.merchant_name.upper() for word in ["UNKNOWN", "TEMP", "UNREGISTERED"]):
            fraud_indicators += 2
        
        if any(word in tx.location.upper() for word in ["FOREIGN", "INTERNATIONAL", "OFFSHORE"]):
            fraud_indicators += 2
        
        # Calculate probability (0-1)
        base_prob = 0.05  # Baseline 5% fraud rate
        fraud_prob = min(base_prob + (fraud_indicators * 0.10), 0.95)
        confidence = 0.75 + (fraud_indicators * 0.05)  # Higher indicators = higher confidence
        
        state.ml_fraud_probability = fraud_prob
        state.ml_confidence = min(confidence, 1.0)
        
        logger.info(f"   ✅ ML fraud probability: {fraud_prob*100:.1f}%, confidence: {confidence*100:.1f}%")
        return state


class DocumentEvidenceAgent:
    """
    Agent 3: Searches document store for relevant compliance evidence.
    
    Uses RAG system to find related:
    - Previous investigation reports
    - Customer due diligence documents
    - Regulatory filings
    - Customer communication records
    """
    
    def __init__(self, rag_pipeline: Optional[GeminiRAGPipeline] = None):
        if rag_pipeline:
            self.rag = rag_pipeline
        else:
            # Initialize RAG system with document store
            try:
                self.rag = GeminiRAGPipeline(llm_provider="gemini")
                logger.info("Initialized Document Evidence Agent with RAG pipeline")
            except Exception as e:
                logger.warning(f"Could not initialize RAG: {e}. Document search will be limited.")
                self.rag = None
    
    async def analyze(self, state: InvestigationState) -> InvestigationState:
        """Search for relevant documents."""
        logger.info(f"📄 Document Agent searching evidence for {state.transaction.transaction_id}")
        
        tx = state.transaction
        
        if self.rag is None:
            logger.warning("   ⚠️  RAG system not available. Skipping document search.")
            state.related_documents = []
            state.document_summaries = ["RAG system not initialized"]
            return state
        
        # Construct search queries
        queries = [
            f"customer {tx.customer_id} previous fraud investigations",
            f"merchant {tx.merchant_name} suspicious activity reports",
            f"{tx.merchant_category} money laundering patterns",
            f"high value transactions {tx.location} compliance"
        ]
        
        all_documents = []
        for query in queries:
            try:
                # Semantic search in document store
                results = self.rag.semantic_search_only(query, top_k=2)
                all_documents.extend(results)
                logger.info(f"   📎 Found {len(results)} documents for: {query[:50]}...")
            except Exception as e:
                logger.warning(f"   ⚠️  Search failed for '{query}': {e}")
        
        # Remove duplicates and store
        unique_docs = {doc['document'][:100]: doc for doc in all_documents}.values()
        state.related_documents = list(unique_docs)
        
        # Generate summaries
        state.document_summaries = [
            f"Document {i+1}: Score {doc['score']:.3f} - {doc['document'][:200]}..."
            for i, doc in enumerate(state.related_documents)
        ]
        
        logger.info(f"   ✅ Collected {len(state.related_documents)} unique documents")
        return state


class ReportGeneratorAgent:
    """
    Agent 4: Synthesizes all evidence and generates comprehensive investigation report.
    
    Uses LLM to:
    - Combine transaction data + rule violations + ML scores + document evidence
    - Generate human-readable investigation summary
    - Recommend specific actions based on risk level
    """
    
    def __init__(self, llm_client: Optional[UniversalLLMClient] = None):
        self.llm = llm_client or UniversalLLMClient()
        logger.info("Initialized Report Generator Agent")
    
    async def analyze(self, state: InvestigationState) -> InvestigationState:
        """Generate final investigation report."""
        logger.info(f"📝 Report Agent generating summary for {state.transaction.transaction_id}")
        
        tx = state.transaction
        
        # Calculate overall risk score (weighted average)
        rule_weight = 0.30
        ml_weight = 0.50
        doc_weight = 0.20
        
        overall_score = (
            (state.rule_score * rule_weight) +
            (state.ml_fraud_probability * 100 * ml_weight) +
            (len(state.related_documents) * 5 * doc_weight)  # More documents = more evidence
        )
        state.overall_risk_score = min(overall_score, 100.0)
        
        # Determine risk level
        if state.overall_risk_score >= 80:
            state.risk_level = "CRITICAL"
        elif state.overall_risk_score >= 60:
            state.risk_level = "HIGH"
        elif state.overall_risk_score >= 40:
            state.risk_level = "MEDIUM"
        elif state.overall_risk_score >= 20:
            state.risk_level = "LOW"
        else:
            state.risk_level = "MINIMAL"
        
        # Prepare context for LLM
        context = f"""
Transaction Investigation Analysis

TRANSACTION DETAILS:
- ID: {tx.transaction_id}
- Customer: {tx.customer_id}
- Amount: ${tx.amount:,.2f}
- Merchant: {tx.merchant_name} ({tx.merchant_category})
- Location: {tx.location}
- Date: {tx.transaction_date}

RULE-BASED ANALYSIS (Scala Engine):
- Score: {state.rule_score:.1f}%
- Violations: {', '.join(state.rule_violations) if state.rule_violations else 'None'}

MACHINE LEARNING ANALYSIS (PyTorch Model):
- Fraud Probability: {state.ml_fraud_probability*100:.1f}%
- Model Confidence: {state.ml_confidence*100:.1f}%

DOCUMENT EVIDENCE:
{chr(10).join(state.document_summaries[:3]) if state.document_summaries else 'No relevant documents found'}

OVERALL ASSESSMENT:
- Combined Risk Score: {state.overall_risk_score:.1f}%
- Risk Level: {state.risk_level}
"""
        
        # Generate investigation summary with LLM
        prompt = f"""{context}

Based on the above multi-source analysis, provide:

1. **Summary**: Brief overview of the investigation findings (2-3 sentences)
2. **Key Risk Factors**: What makes this transaction suspicious?
3. **Recommended Actions**: Specific next steps for the investigation team
4. **Compliance Considerations**: Any regulatory requirements to address

Format as a professional investigation report."""
        
        try:
            report = self.llm.generate(
                prompt,
                max_tokens=1000,
                temperature=0.3  # Lower temperature for factual report
            )
            state.investigation_summary = report
            logger.info(f"   ✅ Generated {len(report)} character report")
        except Exception as e:
            logger.error(f"   ❌ LLM report generation failed: {e}")
            state.investigation_summary = f"Error generating report: {e}\n\n{context}"
        
        # Generate recommended actions based on risk level
        if state.risk_level in ["CRITICAL", "HIGH"]:
            state.recommended_actions = [
                "IMMEDIATE: Block transaction and freeze account",
                "Contact customer for identity verification",
                "File Suspicious Activity Report (SAR) with FinCEN",
                "Notify compliance team for enhanced due diligence",
                "Review all transactions in last 30 days"
            ]
        elif state.risk_level == "MEDIUM":
            state.recommended_actions = [
                "Manual review by fraud analyst",
                "Request additional documentation from customer",
                "Monitor account for 48 hours",
                "Flag for enhanced monitoring"
            ]
        else:
            state.recommended_actions = [
                "Approve transaction",
                "Update customer risk profile",
                "No further action required"
            ]
        
        state.completed = True
        logger.info(f"   ✅ Investigation complete. Risk level: {state.risk_level}")
        return state


class FraudInvestigationWorkflow:
    """
    Main workflow orchestrator using LangGraph (if available) or sequential execution.
    
    This class coordinates the 4 agents to perform a complete fraud investigation:
    1. Rule Engine Agent → Check Scala-based rules
    2. ML Risk Scorer → PyTorch model inference
    3. Document Evidence → RAG system search
    4. Report Generator → LLM synthesis
    """
    
    def __init__(self, use_langgraph: bool = True):
        self.use_langgraph = use_langgraph and LANGGRAPH_AVAILABLE
        
        # Initialize agents
        self.rule_engine = RuleEngineAgent()
        self.ml_scorer = MLRiskScorerAgent()
        self.document_agent = DocumentEvidenceAgent()
        self.report_generator = ReportGeneratorAgent()
        
        # Build workflow
        if self.use_langgraph:
            self.workflow = self._build_langgraph_workflow()
            logger.info("✅ Initialized LangGraph workflow")
        else:
            logger.info("✅ Initialized sequential workflow")
    
    def _build_langgraph_workflow(self) -> StateGraph:
        """Construct LangGraph state machine."""
        # Define workflow graph
        workflow = StateGraph(InvestigationState)
        
        # Add nodes (agents)
        workflow.add_node("rule_engine", self.rule_engine.analyze)
        workflow.add_node("ml_scorer", self.ml_scorer.analyze)
        workflow.add_node("document_evidence", self.document_agent.analyze)
        workflow.add_node("report_generator", self.report_generator.analyze)
        
        # Define edges (execution flow)
        workflow.set_entry_point("rule_engine")
        workflow.add_edge("rule_engine", "ml_scorer")
        workflow.add_edge("ml_scorer", "document_evidence")
        workflow.add_edge("document_evidence", "report_generator")
        workflow.add_edge("report_generator", END)
        
        # Compile with memory
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)
    
    async def investigate(self, transaction: Transaction) -> InvestigationState:
        """
        Run complete investigation workflow.
        
        Args:
            transaction: Transaction object to investigate
        
        Returns:
            InvestigationState with all analysis results
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"🚨 Starting investigation for transaction {transaction.transaction_id}")
        logger.info(f"{'='*70}\n")
        
        # Initialize state
        state = InvestigationState(transaction=transaction)
        
        if self.use_langgraph:
            # Use LangGraph execution
            config = {"configurable": {"thread_id": state.investigation_id}}
            result = await self.workflow.ainvoke(state, config)
            return result
        else:
            # Sequential execution
            state = await self.rule_engine.analyze(state)
            state = await self.ml_scorer.analyze(state)
            state = await self.document_evidence.analyze(state)
            state = await self.report_generator.analyze(state)
            return state
    
    def investigate_sync(self, transaction: Transaction) -> InvestigationState:
        """Synchronous wrapper for investigate()."""
        return asyncio.run(self.investigate(transaction))


# Example usage
if __name__ == "__main__":
    # Sample high-risk transaction
    suspicious_tx = Transaction(
        transaction_id="TXN-2025-11-12-7890",
        customer_id="CUST-45678",
        amount=12500.00,
        merchant_name="Unknown Offshore Merchant",
        merchant_category="WIRE_TRANSFER",
        transaction_date="2025-11-12T03:45:00Z",
        location="Foreign - Cayman Islands",
        raw_data={
            "ip_address": "203.45.67.89",
            "device_id": "unknown",
            "channel": "online"
        }
    )
    
    # Run investigation
    workflow = FraudInvestigationWorkflow(use_langgraph=LANGGRAPH_AVAILABLE)
    result = workflow.investigate_sync(suspicious_tx)
    
    # Print results
    print("\n" + "="*70)
    print("📊 INVESTIGATION RESULTS")
    print("="*70)
    print(f"\nInvestigation ID: {result.investigation_id}")
    print(f"Transaction: {result.transaction.transaction_id}")
    print(f"\n🎯 RISK ASSESSMENT:")
    print(f"   Overall Score: {result.overall_risk_score:.1f}%")
    print(f"   Risk Level: {result.risk_level}")
    print(f"\n⚖️  RULE ENGINE:")
    print(f"   Score: {result.rule_score:.1f}%")
    print(f"   Violations: {', '.join(result.rule_violations)}")
    print(f"\n🤖 MACHINE LEARNING:")
    print(f"   Fraud Probability: {result.ml_fraud_probability*100:.1f}%")
    print(f"   Confidence: {result.ml_confidence*100:.1f}%")
    print(f"\n📄 DOCUMENT EVIDENCE:")
    print(f"   Documents Found: {len(result.related_documents)}")
    print(f"\n📝 INVESTIGATION SUMMARY:")
    print(result.investigation_summary)
    print(f"\n✅ RECOMMENDED ACTIONS:")
    for i, action in enumerate(result.recommended_actions, 1):
        print(f"   {i}. {action}")
    print("\n" + "="*70)
    
    # Export to JSON
    output_path = "investigation_results.json"
    with open(output_path, 'w') as f:
        # Convert to dict (handle dataclass serialization)
        result_dict = {
            'investigation_id': result.investigation_id,
            'transaction': asdict(result.transaction),
            'rule_violations': result.rule_violations,
            'rule_score': result.rule_score,
            'ml_fraud_probability': result.ml_fraud_probability,
            'ml_confidence': result.ml_confidence,
            'document_count': len(result.related_documents),
            'overall_risk_score': result.overall_risk_score,
            'risk_level': result.risk_level,
            'investigation_summary': result.investigation_summary,
            'recommended_actions': result.recommended_actions,
            'completed': result.completed,
            'timestamp': datetime.now().isoformat()
        }
        json.dump(result_dict, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")
