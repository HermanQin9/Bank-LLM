# Integration Architecture - How The System Really Works

## Problem: Why Two Projects Feel Independent

When financial technology systems are built separately:
- Transaction processing runs in Java/Scala (performance, type safety)
- Document intelligence runs in Python (ML ecosystem, LLM libraries)
- They communicate through REST APIs
- **Result**: Two independent systems with shallow integration

## Solution: Deep Integration Through Shared Data & Business Logic

This project implements **genuine fusion** where both systems:
1. **Share the same database** (PostgreSQL) as single source of truth
2. **Read and write each other's data** (bidirectional, not one-way)
3. **Cannot complete business workflows independently** (collaborative)

---

## 1. Database-Level Integration

### Shared Tables (Both Systems Access)

#### `customer_profiles` Table
**Purpose**: Store business expectations extracted from KYC documents

```sql
CREATE TABLE customer_profiles (
    customer_id VARCHAR(50) PRIMARY KEY,
    business_type VARCHAR(100) NOT NULL,
    expected_monthly_volume DECIMAL(15, 2),
    expected_min_amount DECIMAL(15, 2),
    expected_max_amount DECIMAL(15, 2),
    geographic_scope JSONB,
    risk_indicators JSONB,
    kyc_document_source TEXT,
    confidence_score DECIMAL(3, 2),
    ...
);
```

**Data Flow**:
```
Python LLM → WRITES: Extract structured data from KYC PDFs
Scala Rules → READS: Validate transactions against expected patterns
Java Dashboard → READS: Display customer risk profiles
```

**Real Example**:
```python
# Python: LLM extracts from KYC document
profile = llm.extract_business_profile(kyc_pdf)
db.insert("customer_profiles", profile)
# → Expected monthly volume: $250K, Range: $5K-$50K

# Scala: Rule engine reads this profile
if (transaction.amount > profile.expected_max_amount) {
    trigger_alert("AMOUNT_EXCEEDS_PROFILE")
}
```

#### `transaction_alerts` Table
**Purpose**: Store anomalies detected by unified system

```sql
CREATE TABLE transaction_alerts (
    alert_id VARCHAR(50) PRIMARY KEY,
    transaction_id VARCHAR(50) NOT NULL,
    customer_id VARCHAR(50) NOT NULL,
    alert_type VARCHAR(50),     -- 'AMOUNT_ANOMALY', 'GEO_MISMATCH'
    severity VARCHAR(20),        -- 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    deviation_details JSONB,
    supporting_evidence JSONB,   -- Array of document excerpts
    recommended_action VARCHAR(100),
    ...
);
```

**Data Flow**:
```
Java ETL → WRITES: Statistical anomaly detection
Python ML → WRITES: Add document evidence via RAG search
Java Dashboard → READS: Display alerts to analysts
Python Agents → READS: Generate compliance reports
```

**Real Example**:
```java
// Java: Detect statistical anomaly
if (calculateZScore(transaction) > 3.0) {
    Alert alert = new Alert(transaction, "STATISTICAL_ANOMALY");
    db.insert("transaction_alerts", alert);
}

// Python: Enhance with document evidence
documents = rag_system.search(alert.customer_id, alert.context)
alert.supporting_evidence = documents
db.update("transaction_alerts", alert)

// Java Dashboard: Show to analyst
List<Alert> alerts = db.query("SELECT * FROM transaction_alerts WHERE status='PENDING'");
```

#### `document_evidence` Table
**Purpose**: Link documents to transactions/alerts (RAG results)

```sql
CREATE TABLE document_evidence (
    evidence_id SERIAL PRIMARY KEY,
    alert_id VARCHAR(50),
    transaction_id VARCHAR(50),
    customer_id VARCHAR(50) NOT NULL,
    document_type VARCHAR(50),
    document_path TEXT,
    excerpt TEXT,              -- Relevant excerpt extracted by LLM
    relevance_score DECIMAL(3, 2),
    ...
);
```

**Data Flow**:
```
Python RAG → WRITES: Search results linking docs to alerts
Java Compliance Officer UI → READS: View evidence trail
Python Report Generator → READS: Include in SAR reports
```

#### `compliance_reports` Table
**Purpose**: Store generated regulatory reports (SAR, CTR)

```sql
CREATE TABLE compliance_reports (
    report_id VARCHAR(50) PRIMARY KEY,
    customer_id VARCHAR(50) NOT NULL,
    report_type VARCHAR(20),           -- 'SAR', 'CTR', 'CDD'
    suspicious_transaction_count INTEGER,
    report_content TEXT,               -- LLM-generated narrative
    filed_with_regulator BOOLEAN,
    ...
);
```

**Data Flow**:
```
Python LLM → WRITES: Generate report from DB queries + document analysis
Java Audit System → READS: Track filed reports
Python Dashboard → READS: Display report statistics
```

---

## 2. Business Logic Integration

### Workflow 1: Customer Onboarding

**Without Integration** (Broken):
```
Step 1: Upload KYC document → Store in filesystem
Step 2: Manually enter expected transaction volumes → Database
Step 3: Transaction system has no context about business expectations
```

**With Integration** (Working):
```python
# Python: Process KYC document
kyc_text = extract_pdf("customer_123_kyc.pdf")

# LLM extracts structured data
extraction = llm.complete(f"""
Extract the following from this KYC document:
- Business type
- Expected monthly transaction volume
- Typical transaction amount range
- Geographic scope
- Risk factors

Document: {kyc_text}
""")

# Parse LLM response → structured data
profile = CustomerProfile(
    customer_id="CUST_123",
    business_type="Software Consulting",
    expected_monthly_volume=300000.0,
    expected_min_amount=5000.0,
    expected_max_amount=50000.0,
    geographic_scope=["USA", "Canada"],
    ...
)

# Write to PostgreSQL (shared database)
db.insert("customer_profiles", profile)
```

```scala
// Scala: Transaction validation (reads profile created by Python)
class TransactionValidator(db: Database) {
  def validateTransaction(txn: Transaction): ValidationResult = {
    val profile = db.query(
      "SELECT * FROM customer_profiles WHERE customer_id = ?",
      txn.customerId
    )
    
    // Use LLM-extracted expectations for validation
    if (txn.amount > profile.expectedMaxAmount * 1.5) {
      Alert(
        alertType = "AMOUNT_EXCEEDS_PROFILE",
        severity = "HIGH",
        message = s"Amount ${txn.amount} exceeds expected max ${profile.expectedMaxAmount}"
      )
    }
  }
}
```

**Result**: Java/Scala transaction system **automatically uses** customer expectations extracted by Python LLM. No manual data entry.

---

### Workflow 2: Real-Time Transaction Monitoring

**Without Integration** (Broken):
```
Transaction arrives → Rule-based check → Alert or Approve
(No context about WHY this customer might have unusual patterns)
```

**With Integration** (Working):

```python
# core/unified_financial_intelligence.py

class UnifiedFinancialIntelligence:
    def monitor_transaction(self, transaction_id, customer_id, amount, location):
        # Step 1: Query PostgreSQL for customer profile (written by LLM)
        profile = self.db.query(
            "SELECT * FROM customer_profiles WHERE customer_id = %s",
            (customer_id,)
        )
        
        # Step 2: Query PostgreSQL for transaction history (written by Java ETL)
        stats = self.db.query("""
            SELECT AVG(amount) as avg_amount, STDDEV(amount) as std_amount
            FROM transactions
            WHERE customer_id = %s
        """, (customer_id,))
        
        # Step 3: Detect deviations
        z_score = (amount - stats['avg_amount']) / stats['std_amount']
        
        deviations = {}
        if amount > profile['expected_max_amount']:
            deviations['amount_vs_profile'] = {
                'expected_max': profile['expected_max_amount'],
                'actual': amount,
                'deviation': f"{(amount / profile['expected_max_amount'] - 1) * 100:.1f}%"
            }
        
        if location not in profile['geographic_scope']:
            deviations['geographic_mismatch'] = {
                'expected_regions': profile['geographic_scope'],
                'actual_location': location
            }
        
        # Step 4: If anomalies detected, search documents for context
        if deviations:
            # RAG: Find relevant documents
            documents = self.rag.search(
                query=f"customer {customer_id} business expansion international {location}",
                top_k=3
            )
            
            # LLM: Analyze whether documents explain the anomaly
            analysis = self.llm.complete(f"""
            Transaction Analysis:
            - Customer usually transacts ${profile['expected_max_amount']}
            - This transaction: ${amount} (in {location})
            - Deviations: {deviations}
            
            Relevant documents:
            {documents}
            
            Does customer documentation explain this transaction?
            Is this suspicious or legitimate business expansion?
            """)
            
            # Step 5: Write alert to PostgreSQL
            alert = TransactionAlert(
                alert_id=f"ALERT_{transaction_id}",
                transaction_id=transaction_id,
                customer_id=customer_id,
                alert_type="PROFILE_DEVIATION",
                severity="HIGH",
                deviation_details=deviations,
                supporting_evidence=[
                    {
                        'source': doc['metadata']['source'],
                        'excerpt': doc['content'][:200],
                        'relevance_score': doc['score']
                    }
                    for doc in documents
                ],
                recommended_action="MANUAL_REVIEW"
            )
            
            self.db.insert("transaction_alerts", alert)
            
            return alert
```

**Result**: Transaction monitoring uses:
1. **Java-loaded transaction history** (statistical baseline)
2. **LLM-extracted customer profile** (business expectations)
3. **RAG document search** (contextual evidence)
4. **LLM reasoning** (explain anomaly)

All data stored back in **shared PostgreSQL** for Java dashboard to display.

---

### Workflow 3: Compliance Report Generation

**Without Integration** (Broken):
```
Analyst manually:
1. Queries database for suspicious transactions
2. Opens document repository
3. Reads KYC files, emails, contracts
4. Writes SAR report narrative (hours of work)
```

**With Integration** (Working):

```python
class UnifiedFinancialIntelligence:
    def generate_compliance_report(self, customer_id, start_date, end_date):
        # Step 1: Query PostgreSQL for suspicious transactions (Java-loaded data)
        suspicious_txns = self.db.query("""
            SELECT t.*, ta.severity, ta.alert_type
            FROM transactions t
            JOIN transaction_alerts ta ON t.transaction_id = ta.transaction_id
            WHERE t.customer_id = %s 
              AND t.transaction_date BETWEEN %s AND %s
              AND ta.severity IN ('HIGH', 'CRITICAL')
        """, (customer_id, start_date, end_date))
        
        # Step 2: Get customer profile (LLM-extracted from KYC)
        profile = self.db.query(
            "SELECT * FROM customer_profiles WHERE customer_id = %s",
            (customer_id,)
        )
        
        # Step 3: Get document evidence (RAG-linked documents)
        evidence = self.db.query("""
            SELECT de.*, ta.alert_type
            FROM document_evidence de
            JOIN transaction_alerts ta ON de.alert_id = ta.alert_id
            WHERE de.customer_id = %s
        """, (customer_id,))
        
        # Step 4: LLM generates SAR narrative
        report = self.llm.complete(f"""
        Generate a Suspicious Activity Report (SAR) with the following:
        
        CUSTOMER PROFILE:
        {profile}
        
        SUSPICIOUS TRANSACTIONS ({len(suspicious_txns)} total):
        {suspicious_txns}
        
        SUPPORTING DOCUMENTS:
        {evidence}
        
        Write a professional SAR report including:
        1. Summary of suspicious activity
        2. Pattern analysis
        3. Deviation from expected behavior
        4. Documentary evidence
        5. Regulatory recommendation
        """)
        
        # Step 5: Save report to PostgreSQL
        report_id = f"SAR_{customer_id}_{datetime.now().strftime('%Y%m%d')}"
        self.db.insert("compliance_reports", {
            'report_id': report_id,
            'customer_id': customer_id,
            'report_type': 'SAR',
            'suspicious_transaction_count': len(suspicious_txns),
            'total_suspicious_amount': sum(t['amount'] for t in suspicious_txns),
            'report_content': report,
            'generated_at': datetime.now()
        })
        
        return report
```

**Result**: Compliance report requires:
- **Java ETL data** (transactions)
- **Scala detection** (alerts)
- **Python RAG** (document evidence)
- **LLM reasoning** (narrative generation)

Fully automated, takes seconds instead of hours.

---

## 3. Code-Level Integration

### Key Files

#### `core/unified_financial_intelligence.py`
**Main integration module** (700+ lines)

```python
class UnifiedFinancialIntelligence:
    """
    Orchestrates Java/Scala transaction data with Python/LLM document intelligence.
    
    All methods demonstrate genuine integration:
    - Read from Java-populated tables
    - Write to tables Java reads
    - Combine statistical analysis (Scala) with semantic reasoning (LLM)
    """
    
    def __init__(self, db_config, llm_model):
        self.db = DatabaseConnector(db_config)  # PostgreSQL shared by all systems
        self.llm = UniversalLLMClient(llm_model)
        self.rag = GeminiRAGPipeline()
    
    def onboard_customer(self, customer_id, kyc_document_text, document_source):
        """Documents → LLM → Database (Scala reads)"""
        
    def monitor_transaction(self, transaction_id, customer_id, amount, location):
        """Database (Java) → Python Analysis → RAG Search → Database (Java reads)"""
        
    def generate_compliance_report(self, customer_id, start_date, end_date):
        """Database Queries (Java data) + Document Analysis + LLM → Report"""
```

#### `BankFraudTest/src/main/resources/db/migration/V5__Create_unified_integration_tables.sql`
**Shared database schema** (200+ lines)

```sql
-- Tables that both systems use

CREATE TABLE customer_profiles (
    -- Python LLM writes, Scala rules read
);

CREATE TABLE transaction_alerts (
    -- Python/Scala write, Java dashboard reads
);

CREATE TABLE document_evidence (
    -- Python RAG writes, all systems read
);

CREATE TABLE compliance_reports (
    -- Python LLM writes, Java audit reads
);

-- View combining all systems' data
CREATE VIEW customer_risk_dashboard AS
SELECT 
    cp.*,                              -- LLM-extracted profile
    COUNT(t.transaction_id),           -- Java-loaded transactions
    COUNT(ta.alert_id)                 -- Python-generated alerts
FROM customer_profiles cp
LEFT JOIN transactions t ...           -- Java data
LEFT JOIN transaction_alerts ta ...    -- Python data
GROUP BY cp.customer_id;

-- Function for real-time validation
CREATE FUNCTION check_transaction_deviation(...) AS $$
    -- Uses customer_profiles (LLM data) to validate transactions (Java data)
$$;
```

#### `demo_unified_system.py`
**End-to-end demonstration** (600+ lines)

Shows 3 complete workflows:
1. KYC document → LLM extraction → PostgreSQL → Available for Scala rules
2. Transaction → DB query → RAG search → Alert with evidence → Java UI
3. DB aggregation → Document analysis → LLM report → Stored for audit

**Run the demo**:
```bash
python demo_unified_system.py
```

---

## 4. Why This Is True Integration

### What This Is NOT (Shallow Integration)

```
┌──────────────┐                    ┌──────────────┐
│   System A   │ ←── REST API ───→  │   System B   │
│  (Java)      │                    │  (Python)    │
│              │                    │              │
│  Database A  │                    │  Database B  │
└──────────────┘                    └──────────────┘
```
- Two independent systems
- Communicate via HTTP calls
- Separate databases
- Could function independently

### What This IS (Deep Integration)

```
┌──────────────────────────────────────────────────┐
│           SHARED POSTGRESQL DATABASE              │
│                                                   │
│  ┌────────────────┬──────────────────────────┐   │
│  │ Java Writes    │ Python Writes            │   │
│  │ Python Reads   │ Scala Reads              │   │
│  └────────────────┴──────────────────────────┘   │
└──────────────────────────────────────────────────┘
                       ▲
                       │
        ┌──────────────┼──────────────┐
        │              │               │
┌───────▼──────┐  ┌────▼─────┐  ┌─────▼──────┐
│ Java ETL     │  │ Python ML │  │ Scala Rules│
│ (Producer)   │  │ (Bridge)  │  │ (Consumer) │
└──────────────┘  └──────────┘  └────────────┘
```
- Single source of truth (PostgreSQL)
- Bidirectional data flow
- Cannot function independently
- Business logic requires both systems

### Integration Checklist

| Criterion | Status | Evidence |
|-----------|--------|----------|
| **Shared Database** | Yes | All systems read/write PostgreSQL |
| **Bidirectional Data Flow** | Yes | Java→Python→Scala→Java (circular) |
| **Cross-System Dependencies** | Yes | Scala rules need LLM-extracted profiles |
| **Unified Business Logic** | Yes | Workflows span multiple technologies |
| **Real-Time Integration** | Yes | Transaction validation uses live LLM data |
| **Production Code** | Yes | Working implementations, not just design |

---

## 5. Running The Integrated System

### Setup

```bash
# 1. Database migration (creates shared tables)
cd BankFraudTest
mvn flyway:migrate

# 2. Start all services
docker-compose up

# 3. Run integration demo
python demo_unified_system.py
```

### Services

| Service | Port | Purpose |
|---------|------|---------|
| PostgreSQL | 5432 | Shared database (all systems) |
| Java Transaction Service | 8080 | ETL pipeline, dashboard API |
| Python ML API | 8000 | LLM endpoints, RAG search |
| Streamlit UI | 8501 | Unified dashboard |

### Test Integration

```bash
# Load sample data (Java ETL)
cd BankFraudTest
mvn exec:java -Dexec.mainClass="com.bankfraud.DataLoader"

# Process KYC documents (Python LLM)
cd LLM
python src/main.py --mode onboard --customer CUST_123

# Check database (data from both systems)
psql -U postgres -d bankfraud
\dt  # See all tables
SELECT * FROM customer_profiles;  # LLM data
SELECT * FROM transactions LIMIT 10;  # Java data
SELECT * FROM transaction_alerts;  # Python alerts referencing Java transactions

# Generate compliance report (uses ALL systems' data)
python core/unified_financial_intelligence.py --report CUST_123
```

---

## 6. Future Integration Enhancements

### Planned

1. **Real-Time Stream Processing**
   - Kafka bridge between Java ETL and Python ML
   - Sub-second alert generation
   
2. **Active Learning Loop**
   - Analyst feedback → Update LLM prompts
   - Scala rules → Generate training data for ML models
   
3. **Explainability Dashboard**
   - Show full data lineage: Document → Profile → Transaction → Alert → Report
   
4. **Multi-Tenant Isolation**
   - Row-level security in PostgreSQL
   - Each bank's data isolated while sharing infrastructure

---

## Conclusion

This is **not two projects with API bridges**. It's a **unified system** where:
- Transaction processing **needs** document intelligence (to understand customer context)
- Document intelligence **needs** transaction data (to validate patterns)
- Compliance workflows **need both** (regulatory reports require transactions + documents)

The integration is at the **data level** (shared database), **business logic level** (workflows span systems), and **production level** (actual working code, not just architecture diagrams).

**Try it**: Run `python demo_unified_system.py` to see the three workflows in action.
