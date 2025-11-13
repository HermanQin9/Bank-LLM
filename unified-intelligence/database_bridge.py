"""
Database Bridge - Bidirectional Data Flow
==========================================

This layer shows that data flows BOTH ways:
- Java writes transactions → Python reads for ML
- Python writes enriched profiles → Java reads for rules
- LLM writes document evidence → Both systems read

Single source of truth: PostgreSQL
"""

import psycopg2
from psycopg2.extras import RealDictCursor
from typing import Dict, List, Optional
from datetime import datetime, timedelta

try:
    from .shared_models import (
        Transaction, CustomerProfile, FraudAlert, 
        DocumentEvidence, ComplianceReport, RiskLevel, DetectionMethod
    )
    from .schema_adapter import SchemaAdapter
except ImportError:
    from shared_models import (
        Transaction, CustomerProfile, FraudAlert,
        DocumentEvidence, ComplianceReport, RiskLevel, DetectionMethod
    )
    from schema_adapter import SchemaAdapter
class DatabaseBridge:
    """
    NOT just a data access layer - this enables bidirectional integration.
    Every method here represents data flowing between Java and Python.
    """
    
    def __init__(self, config: Dict[str, str]):
        self.config = config
        self._connection = None
    
    def _get_connection(self):
        """Lazy connection with auto-reconnect"""
        if self._connection is None or self._connection.closed:
            self._connection = psycopg2.connect(**self.config)
        return self._connection
    
    
    # ===== READ OPERATIONS (Python reads what Java wrote) =====
    
    def get_transaction(self, transaction_id: str) -> Optional[Transaction]:
        """Python reads transaction that Java ETL wrote"""
        conn = self._get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("""
            SELECT transaction_id, customer_id, amount, merchant_name,
                   merchant_category, transaction_date
            FROM transactions
            WHERE transaction_id = %s
        """, (transaction_id,))
        
        row = cursor.fetchone()
        cursor.close()
        
        if row:
            # Use adapter to convert DB row to model format
            adapted = SchemaAdapter.transaction_from_db(dict(row))
            return Transaction(**adapted)
        return None
    
    
    def get_recent_transactions(
        self, 
        customer_id: str, 
        days: int = 30
    ) -> List[Transaction]:
        """
        Python reads customer history that Java accumulated.
        Used for velocity checks, pattern analysis.
        """
        conn = self._get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cutoff = datetime.now() - timedelta(days=days)
        
        cursor.execute("""
            SELECT transaction_id, customer_id, amount, merchant_name,
                   merchant_category, transaction_date, 
                   location_city, location_country, device_fingerprint
            FROM transactions
            WHERE customer_id = %s 
              AND transaction_date >= %s
            ORDER BY transaction_date DESC
        """, (customer_id, cutoff))
        
        rows = cursor.fetchall()
        cursor.close()
        
        # Use adapter for each row
        transactions = []
        for row in rows:
            adapted = SchemaAdapter.transaction_from_db(dict(row))
            transactions.append(Transaction(**adapted))
        
        return transactions
    
    
    def get_customer_profile(self, customer_id: str) -> Optional[CustomerProfile]:
        """
        Read profile enriched by BOTH systems:
        - Java: transaction statistics
        - Python ML: behavior clusters, anomaly scores
        - LLM: extracted from documents
        
        Uses SchemaAdapter to map existing DB schema to unified model.
        """
        conn = self._get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Query existing columns only
        cursor.execute("""
            SELECT customer_id, business_type, expected_monthly_volume,
                   expected_min_amount, expected_max_amount,
                   geographic_scope, risk_indicators, kyc_document_source,
                   confidence_score, created_at, updated_at
            FROM customer_profiles
            WHERE customer_id = %s
        """, (customer_id,))
        
        row = cursor.fetchone()
        cursor.close()
        
        if row:
            # Use adapter to convert DB row to unified model
            adapted = SchemaAdapter.customer_profile_from_db(dict(row))
            return CustomerProfile(**adapted)
        return None
    
    
    def get_suspicious_transactions(
        self, 
        customer_id: str
    ) -> List[Transaction]:
        """
        Get all transactions flagged by ANY detection method.
        Shows how different systems contribute to the same dataset.
        """
        conn = self._get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("""
            SELECT DISTINCT t.transaction_id, t.customer_id, t.amount, 
                   t.merchant_name, t.merchant_category, t.transaction_date,
                   t.location_city, t.location_country, t.device_fingerprint,
                   fa.fraud_score, fa.risk_level, fa.alert_type
            FROM transactions t
            INNER JOIN fraud_alerts fa ON t.transaction_id = fa.transaction_id
            WHERE t.customer_id = %s
              AND fa.risk_level IN ('HIGH', 'CRITICAL')
            ORDER BY fa.fraud_score DESC, t.transaction_date DESC
            LIMIT 50
        """, (customer_id,))
        
        rows = cursor.fetchall()
        cursor.close()
        
        transactions = []
        for row in rows:
            # Use adapter for base transaction data
            adapted = SchemaAdapter.transaction_from_db(dict(row))
            # Add fraud-specific fields
            adapted['fraud_score'] = row.get('fraud_score')
            adapted['risk_level'] = RiskLevel(row['risk_level']) if row.get('risk_level') else None
            adapted['detection_method'] = DetectionMethod.RULE_BASED  # Map from alert_type if needed
            
            transactions.append(Transaction(**adapted))
        
        return transactions
    
    
    def get_fraud_alert(self, alert_id: str) -> Optional[FraudAlert]:
        """Load alert with full context"""
        conn = self._get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Query existing columns only
        cursor.execute("""
            SELECT alert_id, transaction_id, customer_id, alert_type,
                   fraud_score, risk_level, rules_triggered,
                   description, status, created_at, reviewed_at, reviewed_by
            FROM fraud_alerts
            WHERE alert_id = %s
        """, (alert_id,))
        
        row = cursor.fetchone()
        cursor.close()
        
        if row:
            # Use adapter to convert DB row to unified model
            adapted = SchemaAdapter.fraud_alert_from_db(dict(row))
            # Load supporting documents (if table exists)
            try:
                adapted['supporting_documents'] = self._get_alert_documents(alert_id)
            except:
                adapted['supporting_documents'] = []
            
            return FraudAlert(**adapted)
        return None
    
    
    def _get_alert_documents(self, alert_id: str) -> List[DocumentEvidence]:
        """Load document evidence linked to alert"""
        conn = self._get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("""
            SELECT evidence_id, transaction_id, customer_id,
                   document_source, document_type, document_path,
                   extracted_text, key_entities, sentiment, risk_indicators,
                   relevance_score, llm_reasoning, created_at
            FROM document_evidence
            WHERE alert_id = %s
            ORDER BY relevance_score DESC
        """, (alert_id,))
        
        rows = cursor.fetchall()
        cursor.close()
        
        documents = []
        for row in rows:
            data = dict(row)
            import json
            if isinstance(data.get('key_entities'), str):
                data['key_entities'] = json.loads(data['key_entities'])
            if isinstance(data.get('risk_indicators'), str):
                data['risk_indicators'] = json.loads(data['risk_indicators'])
            documents.append(DocumentEvidence(**data))
        
        return documents
    
    
    # ===== WRITE OPERATIONS (Python writes, Java reads) =====
    
    def upsert_customer_profile(self, profile: CustomerProfile) -> None:
        """
        Python writes enriched profile → Java rules engine reads it.
        This is KEY integration: LLM-extracted data becomes input to Java logic.
        
        Uses SchemaAdapter to map unified model to existing DB schema.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Convert unified model to DB format
        db_data = SchemaAdapter.customer_profile_to_db(profile.dict())
        
        cursor.execute("""
            INSERT INTO customer_profiles (
                customer_id, business_type, expected_monthly_volume,
                expected_min_amount, expected_max_amount,
                geographic_scope, risk_indicators, kyc_document_source,
                confidence_score, updated_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            ON CONFLICT (customer_id) DO UPDATE SET
                business_type = EXCLUDED.business_type,
                expected_monthly_volume = EXCLUDED.expected_monthly_volume,
                expected_min_amount = EXCLUDED.expected_min_amount,
                expected_max_amount = EXCLUDED.expected_max_amount,
                geographic_scope = EXCLUDED.geographic_scope,
                risk_indicators = EXCLUDED.risk_indicators,
                kyc_document_source = EXCLUDED.kyc_document_source,
                confidence_score = EXCLUDED.confidence_score,
                updated_at = EXCLUDED.updated_at
        """, (
            db_data['customer_id'],
            db_data['business_type'],
            db_data['expected_monthly_volume'],
            db_data['expected_min_amount'],
            db_data['expected_max_amount'],
            db_data['geographic_scope'],
            db_data['risk_indicators'],
            db_data['kyc_document_source'],
            db_data['confidence_score'],
            db_data['updated_at']
        ))
        
        conn.commit()
        cursor.close()
        print(f"   Customer profile saved to DB (Java can now read it)")
    
    
    def save_fraud_alert(self, alert: FraudAlert) -> None:
        """
        Python writes alert → Java dashboard displays it.
        Alert combines data from ALL detection methods.
        
        Uses SchemaAdapter to map unified model to existing DB schema.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Convert unified model to DB format
        db_data = SchemaAdapter.fraud_alert_to_db(alert.dict())
        
        cursor.execute("""
            INSERT INTO fraud_alerts (
                alert_id, transaction_id, customer_id, alert_type,
                fraud_score, risk_level, rules_triggered,
                description, status, created_at, reviewed_at, reviewed_by
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            ON CONFLICT (alert_id) DO UPDATE SET
                fraud_score = EXCLUDED.fraud_score,
                risk_level = EXCLUDED.risk_level,
                status = EXCLUDED.status,
                reviewed_by = EXCLUDED.reviewed_by,
                reviewed_at = EXCLUDED.reviewed_at
        """, (
            db_data['alert_id'],
            db_data['transaction_id'],
            db_data['customer_id'],
            db_data['alert_type'],
            db_data['fraud_score'],
            db_data['risk_level'],
            db_data['rules_triggered'],
            db_data['description'],
            db_data['status'],
            db_data['created_at'],
            db_data['reviewed_at'],
            db_data['reviewed_by']
        ))
        
        # Also save supporting documents (if table exists)
        try:
            for doc in alert.supporting_documents:
                self._save_document_evidence(doc, alert.alert_id)
        except:
            pass  # document_evidence table may not exist yet
        
        conn.commit()
        cursor.close()
        print(f"   Fraud alert saved to DB (Java dashboard will show it)")
    
    
    
    def _save_document_evidence(self, doc: DocumentEvidence, alert_id: str) -> None:
        """Save document evidence linked to alert"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        import json
        
        cursor.execute("""
            INSERT INTO document_evidence (
                evidence_id, alert_id, transaction_id, customer_id,
                document_source, document_type, document_path,
                extracted_text, key_entities, sentiment, risk_indicators,
                relevance_score, llm_reasoning, created_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            ON CONFLICT (evidence_id) DO NOTHING
        """, (
            doc.evidence_id,
            alert_id,
            doc.transaction_id,
            doc.customer_id,
            doc.document_source,
            doc.document_type,
            doc.document_path,
            doc.extracted_text,
            json.dumps(doc.key_entities),
            doc.sentiment,
            json.dumps(doc.risk_indicators),
            doc.relevance_score,
            doc.llm_reasoning,
            doc.created_at
        ))
        
        conn.commit()
        cursor.close()
    
    
    def save_compliance_report(self, report: ComplianceReport) -> None:
        """Python generates report → Java compliance team reviews it"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO compliance_reports (
                report_id, customer_id, report_type,
                transaction_count, executive_summary, detailed_analysis,
                recommended_action, generated_by, generated_at,
                approved_by, filed_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            ON CONFLICT (report_id) DO UPDATE SET
                approved_by = EXCLUDED.approved_by,
                filed_at = EXCLUDED.filed_at
        """, (
            report.report_id,
            report.customer_id,
            report.report_type,
            report.transaction_count,
            report.executive_summary,
            report.detailed_analysis,
            report.recommended_action,
            report.generated_by,
            report.generated_at,
            report.approved_by,
            report.filed_at
        ))
        
        conn.commit()
        cursor.close()
        print(f"   💾 Compliance report saved to DB")
    
    
    def close(self):
        """Clean shutdown"""
        if self._connection:
            self._connection.close()
