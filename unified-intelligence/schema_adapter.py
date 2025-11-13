"""
Schema Adapter - Bridge between existing DB schema and unified models
适配现有数据库 schema 和统一模型之间的差异
"""

from typing import Dict, Any, Optional
from datetime import datetime


class SchemaAdapter:
    """Adapt between existing DB columns and unified models"""
    
    @staticmethod
    def transaction_from_db(row: Dict[str, Any]) -> Dict[str, Any]:
        """Convert DB transaction row to unified Transaction model format"""
        return {
            'transaction_id': row['transaction_id'],
            'customer_id': row['customer_id'],
            'amount': float(row['amount']),
            'merchant_name': row.get('merchant_name', 'Unknown'),
            'merchant_category': row.get('merchant_category'),
            'transaction_date': row['transaction_date'],
            # Optional fields (may not exist in current schema)
            'location': row.get('location_city') or row.get('location_country'),
            'device_id': row.get('device_fingerprint')
        }
    
    @staticmethod
    def customer_profile_from_db(row: Dict[str, Any]) -> Dict[str, Any]:
        """Convert DB customer_profiles row to unified CustomerProfile format"""
        return {
            'customer_id': row['customer_id'],
            # Map existing fields to new model
            'account_type': row.get('business_type'),
            'avg_transaction_amount': row.get('expected_monthly_volume'),
            # These fields don't exist yet - will be None
            'account_open_date': None,
            'credit_limit': None,
            'transaction_count_30d': None,
            'max_transaction_amount': row.get('expected_max_amount'),
            'behavior_cluster': None,
            'anomaly_score': None,
            'spending_pattern': None,
            'occupation': None,
            'income_bracket': None,
            'risk_tolerance': None,
            'expected_transaction_types': None,
            'kyc_summary': row.get('kyc_document_source'),
            'unified_risk_score': row.get('confidence_score'),
            'last_updated': row.get('updated_at', datetime.now())
        }
    
    @staticmethod
    def customer_profile_to_db(profile: Dict[str, Any]) -> Dict[str, Any]:
        """Convert unified CustomerProfile to existing DB format"""
        return {
            'customer_id': profile['customer_id'],
            'business_type': profile.get('account_type') or profile.get('occupation'),
            'expected_monthly_volume': profile.get('avg_transaction_amount'),
            'expected_min_amount': None,  # Could calculate from history
            'expected_max_amount': profile.get('max_transaction_amount'),
            'geographic_scope': None,  # Could be populated later
            'risk_indicators': [],  # Could be populated from risk_tolerance
            'kyc_document_source': profile.get('kyc_summary'),
            'confidence_score': profile.get('unified_risk_score'),
            'updated_at': datetime.now()
        }
    
    @staticmethod
    def fraud_alert_from_db(row: Dict[str, Any]) -> Dict[str, Any]:
        """Convert DB fraud_alerts row to unified FraudAlert format"""
        import json
        
        return {
            'alert_id': row['alert_id'],
            'transaction_id': row['transaction_id'],
            'customer_id': row['customer_id'],
            # Map fields
            'rule_based_score': row.get('fraud_score'),  # Currently only one score
            'ml_model_score': None,  # Will be added by unified system
            'llm_risk_score': None,  # Will be added by unified system
            'final_risk_score': row.get('fraud_score', 0.0),
            'risk_level': row.get('risk_level', 'MEDIUM'),
            'detection_method': row.get('alert_type', 'RULE_BASED'),
            'rules_triggered': json.loads(row['rules_triggered']) if isinstance(row.get('rules_triggered'), str) else (row.get('rules_triggered') or []),
            'ml_features': {},
            'llm_reasoning': row.get('description', ''),
            'supporting_documents': [],
            'status': row.get('status', 'PENDING'),
            'assigned_to': row.get('reviewed_by'),
            'resolution_notes': None,
            'created_at': row['created_at'],
            'updated_at': row.get('reviewed_at') or row['created_at']
        }
    
    @staticmethod
    def fraud_alert_to_db(alert: Dict[str, Any]) -> Dict[str, Any]:
        """Convert unified FraudAlert to existing DB format"""
        
        # Use the most relevant score for existing fraud_score field
        fraud_score = alert.get('final_risk_score', 0.0)
        
        # rules_triggered should be a Python list for PostgreSQL TEXT[] array
        rules = alert.get('rules_triggered', [])
        if isinstance(rules, str):
            import json
            rules = json.loads(rules)
        
        return {
            'alert_id': alert['alert_id'],
            'transaction_id': alert['transaction_id'],
            'customer_id': alert['customer_id'],
            'alert_type': alert.get('detection_method', 'UNIFIED_INTELLIGENCE'),
            'fraud_score': fraud_score,
            'risk_level': alert.get('risk_level', 'MEDIUM'),
            'rules_triggered': rules,  # Pass as list, psycopg2 will handle TEXT[] conversion
            'description': alert.get('llm_reasoning', ''),
            'status': alert.get('status', 'PENDING'),
            'created_at': alert.get('created_at', datetime.now()),
            'reviewed_at': alert.get('updated_at'),
            'reviewed_by': alert.get('assigned_to')
        }


# Export
__all__ = ['SchemaAdapter']
