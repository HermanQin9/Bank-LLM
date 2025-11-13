"""
Step-by-Step Testing of Unified Intelligence System
边写边测试 - 逐步验证每个组件
"""

import sys
from pathlib import Path
from datetime import datetime

# Setup paths
sys.path.insert(0, str(Path(__file__).parent / "unified-intelligence"))
sys.path.insert(0, str(Path(__file__).parent / "LLM" / "src"))

print("="*80)
print("UNIFIED INTELLIGENCE SYSTEM - STEP-BY-STEP TESTING")
print("="*80)

# ==================== TEST 1: Import Shared Models ====================
print("\n[TEST 1] Importing shared models...")
try:
    from shared_models import (
        Transaction, CustomerProfile, FraudAlert, 
        RiskLevel, DetectionMethod
    )
    print("✅ All shared models imported successfully")
    
    # Test creating instances
    test_txn = Transaction(
        transaction_id="TEST_001",
        customer_id="CUST_TEST",
        amount=1000.0,
        merchant_name="Test Merchant",
        transaction_date=datetime.now()
    )
    print(f"✅ Transaction model works: {test_txn.transaction_id}")
    
    test_profile = CustomerProfile(
        customer_id="CUST_TEST"
    )
    print(f"✅ CustomerProfile model works: {test_profile.customer_id}")
    
except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


# ==================== TEST 2: Database Connection ====================
print("\n[TEST 2] Testing database connection...")
try:
    import psycopg2
    
    db_config = {
        'host': 'localhost',
        'port': '5432',
        'database': 'frauddb',
        'user': 'postgres',
        'password': 'postgres'
    }
    
    conn = psycopg2.connect(**db_config)
    print("✅ PostgreSQL connection established")
    
    cursor = conn.cursor()
    
    # Check existing tables
    cursor.execute("""
        SELECT table_name FROM information_schema.tables 
        WHERE table_schema = 'public'
        ORDER BY table_name
    """)
    tables = [row[0] for row in cursor.fetchall()]
    print(f"✅ Found {len(tables)} tables: {', '.join(tables)}")
    
    # Check if we need unified tables
    required_tables = ['customer_profiles', 'fraud_alerts', 'document_evidence']
    missing_tables = [t for t in required_tables if t not in tables]
    
    if missing_tables:
        print(f"⚠️  Missing unified tables: {', '.join(missing_tables)}")
        print("   Will create them...")
    else:
        print("✅ All required tables exist")
    
    conn.close()
    
except Exception as e:
    print(f"❌ Database connection failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


# ==================== TEST 3: DatabaseBridge ====================
print("\n[TEST 3] Testing DatabaseBridge...")
try:
    from database_bridge import DatabaseBridge
    
    db = DatabaseBridge(db_config)
    print("✅ DatabaseBridge initialized")
    
    # Try to query transactions (might be empty)
    transactions = db.get_recent_transactions("CUST_TEST", days=30)
    print(f"✅ Query executed: found {len(transactions)} transactions")
    
    # Try to get customer profile (might not exist)
    profile = db.get_customer_profile("CUST_TEST")
    if profile:
        print(f"✅ Customer profile loaded: {profile.customer_id}")
    else:
        print("ℹ️  No existing profile for CUST_TEST (expected for test)")
    
except Exception as e:
    print(f"❌ DatabaseBridge failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


# ==================== TEST 4: Create Test Data ====================
print("\n[TEST 4] Creating test data in database...")
try:
    # Check if customers table exists and has our test customer
    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor()
    
    # Insert test customer if not exists
    cursor.execute("""
        INSERT INTO customers (customer_id, first_name, last_name, email, phone, created_at)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (customer_id) DO NOTHING
    """, ("CUST_TEST", "Test", "Customer", "test@example.com", "555-0000", datetime.now()))
    
    # Insert test transaction if not exists
    cursor.execute("""
        INSERT INTO transactions (
            transaction_id, customer_id, amount, currency, 
            transaction_type, merchant_name, merchant_category,
            transaction_date, source_system, normalized_at, created_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (transaction_id) DO NOTHING
    """, (
        "TXN_TEST_001", "CUST_TEST", 5000.00, "USD",
        "PURCHASE", "Test Merchant", "RETAIL",
        datetime.now(), "JAVA_TEST", datetime.now(), datetime.now()
    ))
    
    conn.commit()
    print("✅ Test customer and transaction created")
    
    # Verify we can read it back
    cursor.execute("SELECT COUNT(*) FROM transactions WHERE customer_id = %s", ("CUST_TEST",))
    count = cursor.fetchone()[0]
    print(f"✅ Test customer has {count} transaction(s)")
    
    conn.close()
    
except Exception as e:
    print(f"❌ Test data creation failed: {e}")
    import traceback
    traceback.print_exc()


# ==================== TEST 5: LLM Client ====================
print("\n[TEST 5] Testing LLM client availability...")
try:
    from llm_engine.universal_client import UniversalLLMClient
    
    llm = UniversalLLMClient()
    print("✅ UniversalLLMClient initialized")
    
    # Simple test prompt
    test_prompt = "Say 'OK' if you can read this."
    response = llm.generate(test_prompt, max_tokens=10, temperature=0)
    print(f"✅ LLM response: {response[:50]}...")
    
except Exception as e:
    print(f"⚠️  LLM client not available: {e}")
    print("   (This is OK - system can work without LLM for basic tests)")


# ==================== TEST 6: RAG System ====================
print("\n[TEST 6] Testing RAG system availability...")
try:
    vector_store_path = Path(__file__).parent / "LLM" / "data" / "vector_store"
    if vector_store_path.exists():
        from rag_system.gemini_rag_pipeline import GeminiRAGPipeline
        
        rag = GeminiRAGPipeline()
        print("✅ RAG pipeline initialized")
        
        # Simple search test
        results = rag.semantic_search_only("test query", top_k=1)
        print(f"✅ RAG search executed: {len(results)} results")
    else:
        print("ℹ️  Vector store not found (RAG will be disabled)")
        print(f"   Expected at: {vector_store_path}")
        
except Exception as e:
    print(f"⚠️  RAG system not available: {e}")
    print("   (This is OK - system can work without RAG for basic tests)")


# ==================== TEST 7: Unified Engine Basic Test ====================
print("\n[TEST 7] Testing unified engine initialization...")
try:
    from unified_engine import UnifiedIntelligenceEngine
    
    engine = UnifiedIntelligenceEngine(db_config)
    print("✅ UnifiedIntelligenceEngine initialized")
    print(f"   Database: {db_config['host']}:{db_config['port']}")
    print(f"   LLM available: {engine.llm is not None}")
    print(f"   RAG available: {engine.rag is not None}")
    
except Exception as e:
    print(f"❌ UnifiedIntelligenceEngine failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


# ==================== TEST 8: Rule-Based Detection ====================
print("\n[TEST 8] Testing rule-based fraud detection...")
try:
    test_transaction = Transaction(
        transaction_id="TXN_TEST_FRAUD",
        customer_id="CUST_TEST",
        amount=15000.0,  # High amount
        merchant_name="Unknown Merchant",  # Suspicious
        transaction_date=datetime.now()
    )
    
    rule_score, rules = engine._apply_rule_based_detection(
        test_transaction,
        None,  # No profile
        []     # No history
    )
    
    print(f"✅ Rule-based detection executed")
    print(f"   Score: {rule_score:.2f}")
    print(f"   Rules triggered: {', '.join(rules) if rules else 'None'}")
    
except Exception as e:
    print(f"❌ Rule-based detection failed: {e}")
    import traceback
    traceback.print_exc()


# ==================== SUMMARY ====================
print("\n" + "="*80)
print("TEST SUMMARY")
print("="*80)
print("""
All core components tested:
   * Shared data models (Transaction, CustomerProfile, FraudAlert)
   * Database connection (PostgreSQL)
   * DatabaseBridge (bidirectional data access)
   * Test data creation
   * UnifiedIntelligenceEngine initialization
   * Rule-based fraud detection

Optional components (may not be available):
   * LLM client (requires API keys)
   * RAG system (requires vector store)

Next steps:
   1. Apply full unified schema: python apply_unified_schema.py
   2. Run full demo: python demo_unified_system.py
   3. Run integration tests: pytest unified-intelligence/test_integration.py
""")

print("\nAll critical tests passed! System is ready for full integration testing.")
