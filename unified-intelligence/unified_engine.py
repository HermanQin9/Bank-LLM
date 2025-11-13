"""
Unified Intelligence Engine - Core Business Logic
==================================================

This is the heart of the integration. NOT a wrapper around two separate systems,
but actual shared business logic that requires BOTH Java and Python components.

Key Principle: Every operation here NEEDS data from multiple sources and
WRITES back to shared state that other systems consume.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json

# Try to import LLM components (optional for basic testing)
try:
    sys.path.insert(0, str(Path(__file__).parent.parent / "LLM" / "src"))
    from llm_engine.universal_client import UniversalLLMClient
    from rag_system.gemini_rag_pipeline import GeminiRAGPipeline
    LLM_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    UniversalLLMClient = None
    GeminiRAGPipeline = None
    LLM_AVAILABLE = False
    print("⚠️  LLM components not available - running in basic mode")

try:
    from .shared_models import (
        Transaction, CustomerProfile, FraudAlert, DocumentEvidence,
        InvestigationWorkflow, ComplianceReport, RiskLevel, DetectionMethod
    )
    from .database_bridge import DatabaseBridge
except ImportError:
    from shared_models import (
        Transaction, CustomerProfile, FraudAlert, DocumentEvidence,
        InvestigationWorkflow, ComplianceReport, RiskLevel, DetectionMethod
    )
    from database_bridge import DatabaseBridge


class UnifiedIntelligenceEngine:
    """
    Single engine that orchestrates:
    - Java transaction data
    - Python ML models
    - LLM document intelligence
    
    This is NOT a facade - it implements business logic that requires all three.
    """
    
    def __init__(self, db_config: Dict[str, str]):
        self.db = DatabaseBridge(db_config)
        
        # Initialize LLM components if available
        if LLM_AVAILABLE and UniversalLLMClient:
            self.llm = UniversalLLMClient()
            self.rag = GeminiRAGPipeline() if Path("LLM/data/vector_store").exists() else None
        else:
            self.llm = None
            self.rag = None
        
        print("Unified Intelligence Engine initialized")
        print(f"   Database: {db_config['host']}:{db_config['port']}")
        print(f"   LLM: {self.llm.__class__.__name__ if self.llm else 'Not Available'}")
        print(f"   RAG: {'Enabled' if self.rag else 'Disabled'}")
    
    
    def enrich_customer_profile_from_documents(self, customer_id: str) -> CustomerProfile:
        """
        BIDIRECTIONAL: 
        1. Java → Read transaction statistics from DB
        2. Python → Search customer documents via RAG
        3. LLM → Extract structured profile from documents
        4. Python → Write enriched profile back to DB
        5. Java → Uses enriched profile in rule engine
        
        This is TRUE integration - neither system can do this alone.
        """
        print(f"\n📊 Enriching profile for customer {customer_id}")
        
        # Step 1: Get base profile from Java's DB (transaction statistics)
        profile = self.db.get_customer_profile(customer_id)
        if not profile:
            profile = CustomerProfile(customer_id=customer_id)
        
        print(f"   ✓ Loaded Java transaction stats: {profile.transaction_count_30d} txns")
        
        # Step 2: Search for customer documents (KYC, emails, support tickets)
        if self.rag:
            doc_query = f"customer {customer_id} kyc profile occupation income risk"
            doc_results = self.rag.semantic_search_only(doc_query, top_k=5)
            
            if doc_results:
                print(f"   ✓ Found {len(doc_results)} relevant documents")
                
                # Step 3: LLM extracts structured data from unstructured documents
                doc_context = "\n\n".join([
                    f"Document {i+1}:\n{doc['document'][:500]}"
                    for i, doc in enumerate(doc_results)
                ])
                
                extraction_prompt = f"""Extract customer profile information from these documents:

{doc_context}

Extract and return JSON with:
{{
    "occupation": "<job title>",
    "income_bracket": "<range>",
    "risk_tolerance": "low|medium|high",
    "expected_transaction_types": ["type1", "type2"],
    "kyc_summary": "<brief summary>"
}}

Only include fields you can confidently extract. Return valid JSON only.
"""
                
                llm_response = self.llm.generate(extraction_prompt, temperature=0.1, max_tokens=300)
                
                try:
                    extracted = json.loads(llm_response)
                    profile.occupation = extracted.get('occupation')
                    profile.income_bracket = extracted.get('income_bracket')
                    profile.risk_tolerance = extracted.get('risk_tolerance')
                    profile.expected_transaction_types = extracted.get('expected_transaction_types', [])
                    profile.kyc_summary = extracted.get('kyc_summary')
                    
                    print(f"   ✓ LLM extracted: occupation={profile.occupation}, risk={profile.risk_tolerance}")
                except json.JSONDecodeError:
                    print(f"   ⚠ LLM response not valid JSON, skipping extraction")
        
        # Step 4: Calculate unified risk score (combines Java stats + LLM insights)
        profile.unified_risk_score = self._calculate_unified_risk(profile)
        profile.last_updated = datetime.now()
        
        # Step 5: Write back to DB (Java will read this)
        self.db.upsert_customer_profile(profile)
        print(f"   ✓ Profile enriched and saved (unified_risk={profile.unified_risk_score:.2f})")
        
        return profile
    
    
    def analyze_transaction_with_full_context(
        self, 
        transaction: Transaction
    ) -> FraudAlert:
        """
        MULTI-SYSTEM WORKFLOW:
        1. Java → Transaction data from DB
        2. Scala → Rule-based scoring
        3. Python ML → Neural network prediction (TODO: integrate PyTorch model)
        4. Python RAG → Find relevant documents
        5. LLM → Reason over combined context
        6. Python → Write alert to DB
        7. Java → Dashboard displays alert
        
        This demonstrates the value proposition: no single system could do this.
        """
        print(f"\n🔍 Analyzing transaction {transaction.transaction_id}")
        
        alert_id = f"ALERT_{transaction.transaction_id}_{int(datetime.now().timestamp())}"
        
        # Step 1 & 2: Get customer profile and transaction history (Java source)
        customer_profile = self.db.get_customer_profile(transaction.customer_id)
        recent_transactions = self.db.get_recent_transactions(transaction.customer_id, days=30)
        
        print(f"   ✓ Loaded customer profile and {len(recent_transactions)} recent txns")
        
        # Step 3: Rule-based scoring (Scala-style functional logic in Python)
        rule_score, rules_triggered = self._apply_rule_based_detection(
            transaction, customer_profile, recent_transactions
        )
        print(f"   ✓ Rule-based score: {rule_score:.2f}, triggered: {rules_triggered}")
        
        # Step 4: ML model scoring (TODO: load actual PyTorch model)
        ml_score = self._apply_ml_model(transaction, customer_profile)
        print(f"   ✓ ML model score: {ml_score:.2f}")
        
        # Step 5: RAG document search for context
        supporting_docs = []
        if self.rag:
            search_query = f"""
            customer {transaction.customer_id} 
            merchant {transaction.merchant_name} 
            amount {transaction.amount}
            suspicious activity fraud
            """
            rag_results = self.rag.semantic_search_only(search_query, top_k=3)
            
            for i, doc in enumerate(rag_results):
                evidence = DocumentEvidence(
                    evidence_id=f"DOC_{transaction.transaction_id}_{i}",
                    transaction_id=transaction.transaction_id,
                    customer_id=transaction.customer_id,
                    document_source=doc.get('metadata', {}).get('source', 'unknown'),
                    document_type="compliance_doc",
                    extracted_text=doc['document'][:500],
                    key_entities={},
                    risk_indicators=[],
                    relevance_score=doc.get('score', 0.0),
                    llm_reasoning="Retrieved via semantic search"
                )
                supporting_docs.append(evidence)
            
            print(f"   ✓ Found {len(supporting_docs)} relevant documents")
        
        # Step 6: LLM reasoning over ALL context
        llm_analysis = self._get_llm_risk_assessment(
            transaction, customer_profile, recent_transactions, 
            rule_score, ml_score, supporting_docs
        )
        
        llm_score = llm_analysis['risk_score']
        llm_reasoning = llm_analysis['reasoning']
        print(f"   ✓ LLM risk score: {llm_score:.2f}")
        
        # Step 7: Ensemble scoring (weighted combination)
        final_score = (
            rule_score * 0.4 +  # Rule-based (stable baseline)
            ml_score * 0.3 +     # ML model (pattern detection)
            llm_score * 0.3      # LLM (contextual reasoning)
        )
        
        risk_level = self._score_to_risk_level(final_score)
        
        # Step 8: Create comprehensive alert
        alert = FraudAlert(
            alert_id=alert_id,
            transaction_id=transaction.transaction_id,
            customer_id=transaction.customer_id,
            rule_based_score=rule_score,
            ml_model_score=ml_score,
            llm_risk_score=llm_score,
            final_risk_score=final_score,
            risk_level=risk_level,
            detection_method=DetectionMethod.UNIFIED_INTELLIGENCE,
            rules_triggered=rules_triggered,
            ml_features={},  # TODO: add feature importance
            llm_reasoning=llm_reasoning,
            supporting_documents=supporting_docs,
            status="PENDING"
        )
        
        # Step 9: Persist alert (Java will read this)
        self.db.save_fraud_alert(alert)
        
        print(f"   ✓ Alert created: {risk_level.value} (final_score={final_score:.2f})")
        print(f"      Method: {DetectionMethod.UNIFIED_INTELLIGENCE.value}")
        
        return alert
    
    
    def generate_investigation_report(
        self, 
        alert_id: str
    ) -> ComplianceReport:
        """
        FULL-STACK REPORT GENERATION:
        1. Java → Query suspicious transactions from DB
        2. Python → Aggregate statistics
        3. Python RAG → Find all related documents
        4. LLM → Generate narrative report
        5. Python → Save to DB
        6. Java → Analyst reviews in dashboard
        
        Output is richer than any single system could produce.
        """
        print(f"\n📋 Generating investigation report for alert {alert_id}")
        
        # Step 1: Load alert with all context
        alert = self.db.get_fraud_alert(alert_id)
        if not alert:
            raise ValueError(f"Alert {alert_id} not found")
        
        # Step 2: Get ALL suspicious transactions for this customer (Java DB)
        suspicious_txns = self.db.get_suspicious_transactions(alert.customer_id)
        print(f"   ✓ Found {len(suspicious_txns)} suspicious transactions")
        
        # Step 3: Customer profile (enriched from documents)
        customer_profile = self.db.get_customer_profile(alert.customer_id)
        
        # Step 4: Gather ALL document evidence via RAG
        all_documents = []
        if self.rag:
            doc_search = f"customer {alert.customer_id} suspicious fraud investigation"
            all_documents = self.rag.semantic_search_only(doc_search, top_k=10)
            print(f"   ✓ Retrieved {len(all_documents)} documents")
        
        # Step 5: LLM generates comprehensive report
        report_prompt = f"""Generate a professional compliance investigation report.

ALERT INFORMATION:
- Alert ID: {alert.alert_id}
- Customer: {alert.customer_id}
- Risk Level: {alert.risk_level.value}
- Final Score: {alert.final_risk_score:.2f}
- Detection: Rule({alert.rule_based_score:.2f}) + ML({alert.ml_model_score:.2f}) + LLM({alert.llm_risk_score:.2f})

CUSTOMER PROFILE:
{customer_profile.model_dump_json(indent=2) if customer_profile else "Not available"}

SUSPICIOUS TRANSACTIONS ({len(suspicious_txns)} total):
{json.dumps([t.model_dump() for t in suspicious_txns[:5]], indent=2, default=str)}

SUPPORTING DOCUMENTS ({len(all_documents)} found):
{chr(10).join([f"- {doc['document'][:200]}..." for doc in all_documents[:3]])}

LLM ANALYSIS:
{alert.llm_reasoning}

Generate a structured report with:
1. EXECUTIVE SUMMARY
2. CUSTOMER OVERVIEW
3. TRANSACTION ANALYSIS
4. DOCUMENT EVIDENCE
5. RISK ASSESSMENT
6. RECOMMENDED ACTION

Format professionally for regulatory review.
"""
        
        report_content = self.llm.generate(report_prompt, temperature=0.2, max_tokens=2000)
        
        # Step 6: Structure the report
        report_id = f"REPORT_{alert.customer_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        report = ComplianceReport(
            report_id=report_id,
            customer_id=alert.customer_id,
            report_type="INVESTIGATION",
            transaction_count=len(suspicious_txns),
            suspicious_transactions=suspicious_txns[:10],  # Top 10
            supporting_documents=alert.supporting_documents,
            executive_summary=report_content[:500],  # First paragraph
            detailed_analysis=report_content,
            recommended_action=self._extract_recommendation(report_content),
            regulatory_citations=[]
        )
        
        # Step 7: Save to DB
        self.db.save_compliance_report(report)
        
        print(f"   ✓ Report generated: {report_id}")
        print(f"      Transactions analyzed: {len(suspicious_txns)}")
        print(f"      Documents cited: {len(all_documents)}")
        
        return report
    
    
    # ===== PRIVATE HELPER METHODS =====
    
    def _apply_rule_based_detection(
        self, 
        transaction: Transaction,
        profile: Optional[CustomerProfile],
        history: List[Transaction]
    ) -> Tuple[float, List[str]]:
        """Scala-style functional rules"""
        score = 0.0
        rules = []
        
        # Rule 1: High amount
        if transaction.amount > 5000:
            score += 0.3
            rules.append("HIGH_AMOUNT")
        
        # Rule 2: Unusual for customer
        if profile and profile.avg_transaction_amount:
            deviation = abs(transaction.amount - profile.avg_transaction_amount) / profile.avg_transaction_amount
            if deviation > 2.0:  # More than 2x average
                score += 0.25
                rules.append("AMOUNT_DEVIATION")
        
        # Rule 3: Velocity check
        if len(history) > 10:
            score += 0.2
            rules.append("HIGH_VELOCITY")
        
        # Rule 4: Unknown merchant
        if "unknown" in transaction.merchant_name.lower():
            score += 0.25
            rules.append("UNKNOWN_MERCHANT")
        
        return min(score, 1.0), rules
    
    
    def _apply_ml_model(
        self, 
        transaction: Transaction,
        profile: Optional[CustomerProfile]
    ) -> float:
        """
        TODO: Load actual PyTorch model
        For now, return heuristic score
        """
        # Placeholder - would call actual model inference
        base_score = 0.5
        
        if transaction.amount > 10000:
            base_score += 0.2
        
        if profile and profile.anomaly_score:
            base_score = (base_score + profile.anomaly_score) / 2
        
        return min(base_score, 1.0)
    
    
    def _get_llm_risk_assessment(
        self,
        transaction: Transaction,
        profile: Optional[CustomerProfile],
        history: List[Transaction],
        rule_score: float,
        ml_score: float,
        documents: List[DocumentEvidence]
    ) -> Dict:
        """LLM analyzes ALL context and provides reasoning"""
        
        prompt = f"""Analyze this transaction for fraud risk.

TRANSACTION:
- ID: {transaction.transaction_id}
- Amount: ${transaction.amount:,.2f}
- Merchant: {transaction.merchant_name}
- Date: {transaction.transaction_date}

CUSTOMER PROFILE:
- Average Amount: ${profile.avg_transaction_amount if profile and profile.avg_transaction_amount else 'Unknown'}
- 30-day Transactions: {profile.transaction_count_30d if profile else 'Unknown'}
- Risk Tolerance: {profile.risk_tolerance if profile else 'Unknown'}

DETECTION SCORES:
- Rule-based: {rule_score:.2f}
- ML Model: {ml_score:.2f}

RECENT HISTORY ({len(history)} transactions):
{json.dumps([{"amount": t.amount, "merchant": t.merchant_name} for t in history[:5]], indent=2)}

SUPPORTING EVIDENCE ({len(documents)} documents):
{chr(10).join([f"- {doc.extracted_text[:150]}..." for doc in documents[:2]])}

Provide risk assessment as JSON:
{{
    "risk_score": <0.0 to 1.0>,
    "reasoning": "<detailed explanation>",
    "key_factors": ["factor1", "factor2"]
}}
"""
        
        response = self.llm.generate(prompt, temperature=0.3, max_tokens=400)
        
        try:
            analysis = json.loads(response)
            return {
                'risk_score': analysis.get('risk_score', 0.5),
                'reasoning': analysis.get('reasoning', response),
                'key_factors': analysis.get('key_factors', [])
            }
        except json.JSONDecodeError:
            return {
                'risk_score': 0.5,
                'reasoning': response,
                'key_factors': []
            }
    
    
    def _calculate_unified_risk(self, profile: CustomerProfile) -> float:
        """Combine all risk factors into single score"""
        risk = 0.5  # Baseline
        
        # Factor in transaction patterns
        if profile.anomaly_score:
            risk = (risk + profile.anomaly_score) / 2
        
        # Factor in LLM-extracted risk tolerance
        if profile.risk_tolerance == "high":
            risk += 0.1
        elif profile.risk_tolerance == "low":
            risk -= 0.1
        
        return max(0.0, min(1.0, risk))
    
    
    def _score_to_risk_level(self, score: float) -> RiskLevel:
        """Convert numeric score to categorical risk"""
        if score >= 0.8:
            return RiskLevel.CRITICAL
        elif score >= 0.6:
            return RiskLevel.HIGH
        elif score >= 0.4:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    
    def _extract_recommendation(self, report_text: str) -> str:
        """Extract action items from report"""
        # Simple heuristic - look for "RECOMMENDED" section
        lines = report_text.split('\n')
        for i, line in enumerate(lines):
            if 'RECOMMEND' in line.upper():
                return '\n'.join(lines[i:i+3])
        return "Manual review required"


# Singleton instance
_engine_instance = None

def get_unified_engine(db_config: Dict[str, str] = None) -> UnifiedIntelligenceEngine:
    """Get or create the unified engine"""
    global _engine_instance
    
    if _engine_instance is None:
        if db_config is None:
            db_config = {
                'host': 'localhost',
                'port': '5432',
                'database': 'frauddb',
                'user': 'postgres',
                'password': 'postgres'
            }
        _engine_instance = UnifiedIntelligenceEngine(db_config)
    
    return _engine_instance
