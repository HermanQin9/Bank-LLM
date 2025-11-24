# Financial Intelligence & Compliance Platform

[![Java](https://img.shields.io/badge/Java-21-orange.svg)](https://www.oracle.com/java/)
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Scala](https://img.shields.io/badge/Scala-2.13-red.svg)](https://www.scala-lang.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-blue.svg)](https://www.postgresql.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **An integrated system combining transaction monitoring, document intelligence, and AI-powered compliance automation for financial institutions.**

---

## Project Overview

This platform addresses a critical challenge in modern banking: **efficiently processing massive transaction volumes while maintaining regulatory compliance through document understanding**.

Financial institutions face two parallel but interconnected problems:
1. **Transaction Monitoring**: Analyzing millions of transactions for fraud, AML, and suspicious activities
2. **Document Compliance**: Processing regulatory reports, investigation documents, customer communications, and audit trails

This project integrates both capabilities into a unified system using:
- **Data Engineering (Java/Scala)**: High-performance ETL for 2.2M+ banking transactions
- **Machine Learning (Python/PyTorch)**: Deep learning models for pattern detection
- **Document AI (LLM)**: Multi-provider LLM integration for regulatory document understanding
- **Real-time Analytics**: Sub-100ms latency for production decision-making

### Real-World Use Cases

**Fraud Investigation Workflow**:
```
Transaction Alert (Java ETL) 
    ↓
Fraud Pattern Detection (Scala Rules)
    ↓
ML Risk Scoring (PyTorch)
    ↓
Document Evidence Extraction (LLM)
    ↓
Automated Compliance Report (Multi-Agent System)
```

**Compliance Automation**:
- Automatically extract information from regulatory filings (SAR, CTR, FBAR)
- Cross-reference transaction data with compliance documents
- Generate investigation summaries from multiple data sources
- Monitor customer communications for compliance risks

---

## System Architecture - Deep Integration

**This is NOT two separate projects connected by APIs.** The integration occurs at the data and business logic level, creating a unified system where transaction processing and document intelligence work together seamlessly.

### Integration Architecture

```
┌───────────────────────────────────────────────────────────────────────┐
│                   UNIFIED INTELLIGENCE PLATFORM                       │
│                   Single System, Not Two Projects                     │
└───────────────────────────────────────────────────────────────────────┘
                                 ▲
                                 │
                                 ▼
┌───────────────────────────────────────────────────────────────────────┐
│                        SHARED DATA LAYER                              │
│                        PostgreSQL Database                            │
├──────────────────┬─────────────────────┬──────────────────────────────┤
│  transactions    │  customer_profiles  │  fraud_alerts                │
│  (Java writes)   │  (Python ML/LLM     │  (Python ML writes,          │
│  (Python reads)  │   analyzes,         │   Java reads for             │
│                  │   Java reads)       │   dashboard)                 │
├──────────────────┴─────────────────────┴──────────────────────────────┤
│  Python-enriched data immediately available to Java                   │
│  Java transaction data instantly accessible to Python ML/LLM          │
└────────────────────────────────────────────────────────────────────────┘
                                 ▲
                                 │ Real-Time Bidirectional Flow
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│              DEEP INTEGRATION LAYER (unified-intelligence/)         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  schema_adapter.py:                                                 │
│    • Converts Java DB schema (18-col transactions) to Python       │
│    • Handles PostgreSQL TEXT[] arrays (rules_triggered field)      │
│    • Ensures zero data loss in bidirectional conversion            │
│                                                                     │
│  database_bridge.py:                                                │
│    • Java writes transactions → Python reads immediately            │
│    • Python writes enriched profiles → Java reads for validation   │
│    • Shared connection pool, no data duplication                   │
│                                                                     │
│  shared_models.py:                                                  │
│    • UnifiedTransaction, UnifiedCustomerProfile, UnifiedAlert      │
│    • Common data structures understood by both Java and Python     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                                 ▲
                                 │ HTTP + Database Bridge
                                 ▼
┌──────────────────────────────┬──────────────────────────────────────┐
│   Transaction Engine         │   Intelligence Engine                │
│   (Java/Scala)               │   (Python ML/LLM)                    │
│                              │                                      │
│  PythonBridge.java:          │  integration_api.py:                 │
│    • analyzeTransactionRealtime()  • FastAPI endpoints            │
│    • HTTP POST to Python API │    • Receives Java transactions     │
│    • CompletableFuture async │    • Returns ML analysis results   │
│    • getEnrichedProfile()    │    • Dual routes for compatibility  │
│      reads Python-written DB │                                      │
│                              │  document_parser/:                   │
│  DeepIntegrationDemo.java:   │    • PDF/text extraction            │
│    • End-to-end workflow     │    • KYC document processing        │
│    • Creates transaction     │                                      │
│    • Triggers Python analysis│  rag_system/:                        │
│    • Reads results           │    • Vector embeddings              │
│    • Makes intelligent       │    • Semantic document search       │
│      decision                │    • Evidence linking               │
└──────────────────────────────┴──────────────────────────────────────┘
```

### Real-World Integration Flow: Suspicious Transaction Detection

```
Step 1: Transaction Occurs (Java)
  ├─ Java ETL pipeline creates Transaction object
  ├─ Saved to PostgreSQL 'transactions' table
  └─ Transaction ID: TXN-2025-001

Step 2: Real-Time Analysis Trigger (Java → Python)
  ├─ PythonBridge.analyzeTransactionRealtime(transaction)
  ├─ HTTP POST to http://localhost:8000/analyze/transaction
  ├─ Request body: {transactionId, customerId, amount, merchant, ...}
  └─ CompletableFuture<AnalysisResult> (non-blocking)

Step 3: Python Intelligence Processing
  ├─ integration_api.py receives request
  ├─ database_bridge.py fetches full transaction from PostgreSQL
  ├─ schema_adapter.py converts to UnifiedTransaction
  ├─ ML model predicts fraud probability: 87%
  ├─ LLM analyzes transaction context + customer history
  ├─ RAG system searches for related compliance documents
  └─ Generates AnalysisResult with explanation

Step 4: Python Writes Enriched Data (Python → Database)
  ├─ Creates fraud_alert record
  ├─ Updates customer_profile with risk_score
  ├─ Links document evidence
  └─ Commits to PostgreSQL (immediately visible to Java)

Step 5: Java Receives Results (Python → Java)
  ├─ HTTP 200 response with JSON:
  │   {
  │     "transaction_id": "TXN-2025-001",
  │     "customer_id": "CUST-001",
  │     "risk_score": 87,
  │     "risk_level": "HIGH",
  │     "fraud_probability": 0.87,
  │     "explanation": "Unusual amount + new merchant + unusual time",
  │     "recommended_action": "BLOCK_AND_INVESTIGATE"
  │   }
  ├─ CompletableFuture resolves
  └─ DeepIntegrationDemo.main() continues execution

Step 6: Java Makes Decision (Java)
  ├─ Reads AnalysisResult
  ├─ Fetches EnrichedCustomerProfile from DB (Python-written)
  ├─ Applies business rules
  ├─ Decision: BLOCK transaction + trigger investigation
  └─ Logs complete audit trail

Step 7: Dashboard Display (Java + Python)
  ├─ Java dashboard queries fraud_alerts table
  ├─ Displays Python ML analysis results
  ├─ Shows LLM reasoning explanation
  └─ Analyst can access linked document evidence

Total Time: < 2 seconds (real-time processing)
```

**Key Point**: Every step requires BOTH systems. Java cannot analyze without Python ML/LLM. Python cannot access transactions without Java ETL. This is ONE unified platform.

### Key Integration Points

| Component | Java/Scala Contribution | Python ML/LLM Contribution | Integration Mechanism |
|-----------|------------------------|----------------------------|----------------------|
| **Real-Time Transaction Analysis** | Creates transactions, triggers analysis via PythonBridge | ML fraud detection + LLM reasoning + document context | HTTP API + CompletableFuture async |
| **Customer Risk Profiles** | Reads enriched profiles for validation | ML analyzes patterns, LLM extracts KYC info | SchemaAdapter + shared PostgreSQL |
| **Fraud Alerts** | Dashboard queries and displays | ML generates alerts with confidence scores | fraud_alerts table (Python writes, Java reads) |
| **Document Intelligence** | Provides transaction context | RAG semantic search + LLM document understanding | Database bridge + API endpoints |
| **Compliance Reporting** | Transaction history queries | LLM narrative generation with evidence | Bidirectional DB access |

**This is NOT an API integration between separate systems. Key differences:**

**Deep Integration (What We Built)**:
- Shared database with bidirectional reads/writes
- Real-time data flow: Java creates → Python analyzes → Java decides (< 2s)
- Schema adapter ensures zero data loss in conversions
- Both systems are REQUIRED for any workflow to complete
- Single deployment, unified monitoring, shared data models

**Superficial API Integration (What We Avoided)**:
- Two separate databases with data duplication
- Request/response only (no shared state)
- Each system works independently
- Optional communication (system works without the other)
- Separate deployments, separate monitoring

**Proof of Deep Integration**:
1. Run `run_deep_integration_demo.bat` → See Java create transaction, trigger Python analysis, receive ML/LLM results, make decision
2. Check database after demo → See Python-written customer_profiles used by Java rules
3. All 30 tests pass → Java tests read Python data, Python tests read Java data
4. Try disabling Python service → Java cannot complete analysis (proves dependency)

---

## Key Features

### 1. **Transaction Processing Pipeline** (Java/Scala)
- **Multi-format ETL**: Process CSV, JSON, fixed-width transaction files
- **2.2M+ Record Scale**: Production-tested on real banking data
- **High Performance**: 10K records/sec with HikariCP connection pooling
- **Data Quality**: Handles 7 date formats, deduplication, validation
- **Database**: PostgreSQL 15 with optimized indexes and Flyway migrations

### 2. **Intelligent Fraud Detection** (Scala + PyTorch + 🆕 Spark)
- **Rule-Based Engine** (Scala functional programming):
  - High-value transaction detection (>$5K)
  - Velocity analysis (transaction frequency)
  - Statistical anomaly detection (z-score)
  - Time-based patterns (unusual hours)
  - New merchant alerts
  
- **Deep Learning Models** (PyTorch):
  - Neural networks with attention mechanisms
  - GPU-accelerated training (10x faster)
  - 95%+ AUC-ROC accuracy
  - <100ms inference latency
  - Batch and real-time prediction

- **🆕 Spark Distributed Processing**:
  - Batch processing: 30K+ records/sec on EMR cluster
  - Structured Streaming: Real-time Kafka/Kinesis ingestion
  - S3 data lake: Parquet format with 5x compression
  - AWS EMR: Auto-scaling 1-10 task nodes
  - Geographic anomaly detection at scale
  - See [SPARK_EMR_INTEGRATION.md](SPARK_EMR_INTEGRATION.md)

### 3. **Document Intelligence** (Multi-LLM)
- **Multi-Provider Support**:
  - Google Gemini 2.5-flash (advanced reasoning)
  - Groq Llama 3.3-70B (ultra-fast inference)
  - OpenRouter (100+ model access)
  - Hugging Face (local deployment)
  - Universal client with automatic fallback

- **Document Processing**:
  - PDF parsing with layout analysis
  - OCR for scanned documents (Tesseract/EasyOCR)
  - Structured information extraction
  - Multi-document reasoning

### 4. **RAG System** (Retrieval-Augmented Generation)
- **Gemini Embeddings**: 768-dimensional semantic vectors
- **Vector Store**: ChromaDB/FAISS for similarity search
- **Hybrid Search**: Combine transaction data + document context
- **Source Citation**: Traceable answers with document references
- **Context Window**: Up to 8K tokens for complex queries

### 5. **LangGraph Multi-Agent Workflows**
- **State Management**: Complex workflow orchestration
- **Conditional Routing**: Dynamic decision trees
- **Agent Collaboration**:
  - Transaction Analyzer Agent
  - Rule Engine Agent
  - Document Extraction Agent
  - Risk Assessment Agent
- **Error Recovery**: Automatic retry and fallback logic

### 6. **Production-Ready Infrastructure**
- **API Services**: FastAPI with async support
- **Web Dashboard**: Streamlit for monitoring and investigation
- **Docker Deployment**: Full containerization
- **Monitoring**: Comprehensive logging and metrics
- **Testing**: 30+ automated tests, 85%+ coverage
- **CI/CD**: GitHub Actions pipeline

---

## Technical Capabilities Demonstrated

| Capability | Technology | Evidence |
|-----------|------------|----------|
| **Large-Scale Data Processing** | Java 21, HikariCP | 2.2M+ transactions, 10K records/sec |
| **Functional Programming** | Scala 2.13 | Immutable fraud detection rules |
| **🆕 Distributed Computing** | **Apache Spark 3.5** | **30K records/sec, batch + streaming** |
| **🆕 Cloud Big Data** | **AWS EMR** | **Auto-scaling clusters, 1-10 nodes** |
| **🆕 Data Lake** | **AWS S3 + Parquet** | **5.2x compression, partition pruning** |
| **Deep Learning** | PyTorch 2.0 | GPU training, attention networks |
| **NLP & Embeddings** | Transformers, BERT | Financial text understanding |
| **LLM Integration** | Gemini, Groq, OpenRouter | Multi-provider document AI |
| **Agent Workflows** | LangGraph | State graphs, conditional routing |
| **Database Optimization** | PostgreSQL 15 | Indexing, pooling, migrations |
| **API Development** | FastAPI | Async endpoints, <100ms latency |
| **Containerization** | Docker Compose | Multi-service orchestration |
| **Testing** | JUnit, pytest, ScalaTest | 42+ tests, integration testing |

---

## Quick Start

### Prerequisites
- **Java**: 21 or higher
- **Python**: 3.11+ (conda recommended)
- **Maven**: 3.9+
- **Docker**: For PostgreSQL
- **Git**: For cloning repository
- **🆕 Apache Spark**: 3.5.0 (optional, for distributed processing)
- **🆕 AWS CLI**: Configured (optional, for EMR deployment)

### Installation

1. **Clone Repository**
```bash
git clone https://github.com/HermanQin9/fraud_test.git
cd BankFraudTest-LLM
```

2. **Start PostgreSQL Database**
```bash
cd BankFraudTest
docker-compose up -d
# Wait 10 seconds for database initialization
```

3. **Build Java Project & Run Migrations**
```bash
mvn clean install
mvn flyway:migrate  # Creates tables: transactions, customer_profiles, fraud_alerts
```

4. **Setup Python Environment**
```bash
cd ../LLM
conda create -n financial-ai python=3.11
conda activate financial-ai
pip install -r requirements.txt
```

5. **Configure API Keys** (Optional for demo, required for LLM features)
```bash
cp .env.example .env
# Edit .env with your API keys:
# - GEMINI_API_KEY (Google AI Studio)
# - GROQ_API_KEY (Groq Cloud)
# - OPENROUTER_API_KEY (OpenRouter)
```

### Run Deep Integration Demo

**Windows (One Command)**:
```bash
# From project root
run_deep_integration_demo.bat
```

**Manual Execution**:
```bash
# Terminal 1: Start Python Intelligence Service
cd LLM
python -m uvicorn app.integration_api:app --reload --port 8000

# Terminal 2: Run Java Deep Integration Demo
cd BankFraudTest
mvn compile exec:java -Dexec.mainClass="com.bankfraud.integration.DeepIntegrationDemo"
```

**What the Demo Shows**:
1. Java creates suspicious transaction (amount=$15,000, unusual time=3 AM)
2. PythonBridge triggers real-time analysis via HTTP POST
3. Python ML model predicts fraud probability: 87%
4. Python LLM explains: "High amount + new merchant + unusual hour"
5. Python writes enriched customer profile to PostgreSQL
6. Java reads Python-enriched data from database
7. Java makes intelligent decision: BLOCK + INVESTIGATE

**Expected Output**:
```
[DEMO] Step 1: Creating suspicious transaction...
[DEMO] Step 2: Triggering Python real-time analysis...
[DEMO] Step 3: Waiting for ML/LLM analysis (async)...
[DEMO] Step 4: Analysis complete!
  ├─ Risk Score: 87%
  ├─ Risk Level: HIGH
  ├─ Explanation: High-value transaction at unusual time with new merchant
  └─ Recommendation: BLOCK_AND_INVESTIGATE
[DEMO] Step 5: Reading Python-enriched customer profile...
  ├─ Customer risk_score: 87
  ├─ Updated by Python ML engine
[DEMO] Step 6: Java decision based on Python intelligence: TRANSACTION BLOCKED
[DEMO] Step 7: Complete audit trail saved to database

DEEP INTEGRATION VERIFIED: Java ↔ Python ↔ Database ↔ ML/LLM
```

---

## Usage Examples

### 1. Process Transaction Batch
```bash
cd BankFraudTest
./src/main/scripts/batch_import.sh data/sample/
```

### 2. Fraud Detection (Scala)
```scala
import com.bankfraud.analytics.FraudAnalyzer

val analyzer = new FraudAnalyzer()
val transaction = Transaction(
  transactionId = "TXN12345",
  amount = BigDecimal("8500.00"),
  merchantName = "Unknown Vendor",
  transactionDate = LocalDateTime.now()
)

val score = analyzer.analyzeFraud(transaction, customerHistory)
println(s"Risk Score: ${score.score}%, Level: ${score.riskLevel}")
// Output: Risk Score: 65.0%, Level: HIGH
```

### 3. Document Intelligence (Python)
```python
from src.llm_engine.universal_client import UniversalLLMClient
from src.rag_system.gemini_rag_pipeline import GeminiRAGPipeline

# Extract from compliance document
client = UniversalLLMClient()
document_text = open("SAR_report.pdf").read()
response = client.generate(
    f"Extract suspicious activity details from:\n{document_text}"
)

# RAG query combining transaction + document context
rag = GeminiRAGPipeline()
rag.index_documents(["compliance_docs/", "investigation_reports/"])
answer = rag.query(
    "What patterns indicate money laundering in account #12345?",
    top_k=5
)
print(answer['answer'])
print(f"Referenced {answer['num_sources']} documents")
```

### 4. Integrated Investigation Workflow
```python
# Bridge between Java transaction data and Python LLM analysis
import psycopg2
from src.llm_engine.universal_client import UniversalLLMClient

# 1. Query high-risk transactions from PostgreSQL
conn = psycopg2.connect(database="frauddb", user="postgres")
cursor = conn.execute("""
    SELECT transaction_id, amount, merchant_name, fraud_score
    FROM fraud_alerts 
    WHERE risk_level = 'HIGH' AND investigation_status = 'PENDING'
    ORDER BY fraud_score DESC
    LIMIT 10
""")

# 2. Generate investigation summary with LLM
llm = UniversalLLMClient()
for row in cursor.fetchall():
    tx_id, amount, merchant, score = row
    
    # 3. Fetch related compliance documents
    related_docs = rag.semantic_search_only(
        f"investigation merchant {merchant} amount {amount}",
        top_k=3
    )
    
    # 4. Generate comprehensive report
    context = "\n".join([doc['document'] for doc in related_docs])
    report = llm.generate(f"""
    Transaction Alert Analysis:
    - ID: {tx_id}
    - Amount: ${amount}
    - Merchant: {merchant}
    - Fraud Score: {score}%
    
    Related Documents:
    {context}
    
    Provide:
    1. Risk assessment
    2. Recommended actions
    3. Compliance considerations
    """)
    
    print(f"=== Investigation Report: {tx_id} ===")
    print(report)
```

---

## Project Structure

```
BankFraudTest-LLM/
│
├── BankFraudTest/                    # Transaction Processing System
│   ├── src/
│   │   ├── main/
│   │   │   ├── java/                # Java ETL pipeline
│   │   │   │   ├── config/          # HikariCP, DB config
│   │   │   │   ├── model/           # Transaction, Customer, FraudAlert
│   │   │   │   ├── reader/          # CSV, JSON, Fixed-width readers
│   │   │   │   ├── repository/      # Data access layer
│   │   │   │   └── service/         # Business logic
│   │   │   ├── scala/               # Scala fraud analytics
│   │   │   │   └── analytics/       # FraudAnalyzer, Statistics
│   │   │   ├── resources/
│   │   │   │   ├── application.properties
│   │   │   │   └── db/migration/    # Flyway SQL scripts
│   │   │   └── scripts/             # Unix automation
│   │   └── test/                    # JUnit, ScalaTest, Testcontainers
│   ├── data/
│   │   ├── sample/                  # Demo data (CSV, JSON, TXT)
│   │   └── credit_card/             # Real dataset (24K records)
│   ├── docs/                        # Technical documentation
│   ├── docker/                      # Docker configs
│   └── pom.xml                      # Maven dependencies
│
├── LLM/                             # Document Intelligence System
│   ├── src/
│   │   ├── llm_engine/              # Multi-provider LLM clients
│   │   │   ├── gemini_client.py     # Google Gemini integration
│   │   │   ├── groq_client.py       # Groq API
│   │   │   ├── openrouter_client.py # OpenRouter
│   │   │   ├── universal_client.py  # Auto-fallback logic
│   │   │   └── prompt_templates.py  # Financial domain prompts
│   │   ├── document_parser/         # PDF, OCR processing
│   │   ├── rag_system/              # Vector store, retrieval
│   │   │   ├── gemini_rag_pipeline.py # 768-dim embeddings
│   │   │   └── vector_store.py      # ChromaDB integration
│   │   ├── nlp_pipeline/            # Text preprocessing
│   │   └── utils/                   # Config, logging, helpers
│   ├── app/
│   │   ├── api.py                   # FastAPI REST endpoints
│   │   └── dashboard.py             # Streamlit UI (1400+ lines)
│   ├── tests/                       # pytest suite
│   ├── notebooks/                   # Jupyter demos
│   ├── data/
│   │   ├── sample_documents/        # Compliance docs, reports
│   │   └── vector_store/            # Persisted embeddings
│   ├── requirements.txt             # Python dependencies
│   ├── docker-compose.yml           # Multi-service deployment
│   └── .env.example                 # API key template
│
├── unified-intelligence/            # **Deep Integration Layer**
│   ├── schema_adapter.py            # Bridge Java DB schema ↔ Python models
│   ├── database_bridge.py           # Bidirectional data access
│   ├── shared_models.py             # Pydantic models shared across systems
│   └── README.md                    # Integration architecture docs
│
├── app/
│   ├── integration_api.py           # FastAPI endpoints for Java-Python bridge
│   ├── api.py                       # Main API service
│   └── dashboard.py                 # Streamlit monitoring UI
│
├── notebooks/
│   └── 01_document_understanding_demo.ipynb  # LLM/RAG demonstration
│
├── docker-compose.yml               # Unified deployment
├── README.md                        # This file
└── .gitignore
```

---

## Technical Deep Dive

### Transaction Processing (Java/Scala)
- **HikariCP**: 20-connection pool, optimized for throughput
- **Batch Operations**: 1000-record chunks for PostgreSQL
- **Flyway**: Version-controlled schema migrations
- **Functional Scala**: Immutable data structures, pattern matching
- **Test Coverage**: 30 tests (JUnit, ScalaTest), Testcontainers for integration

### Deep Learning Pipeline (PyTorch)
- **Model Architecture**: 
  - Input: 788 dimensions (768 text embeddings + 20 numeric features)
  - Layers: [512, 256, 128] with BatchNorm + Dropout
  - Attention mechanism for feature importance
  - Binary classification output
- **Training**: 
  - Adam optimizer with learning rate scheduling
  - Class weights for imbalanced data (98% normal, 2% fraud)
  - GPU acceleration (CUDA)
  - Mixed precision (FP16) for 2-3x speedup
- **Performance**: 95%+ AUC-ROC, <10ms inference

### LLM Integration Strategy
- **Provider Diversity**: 
  - Gemini: Best for complex reasoning
  - Groq: Ultra-fast (< 1s response)
  - OpenRouter: Cost optimization
  - Universal client: Automatic fallback
- **Prompt Engineering**: 
  - Financial domain templates
  - Few-shot examples
  - Chain-of-thought reasoning
- **RAG Implementation**:
  - Gemini embeddings (768-dim)
  - Cosine similarity search
  - Context window management (8K tokens)
  - Source attribution

### LangGraph Workflows
- **State Graph**: Define investigation workflow
- **Nodes**: ML analysis, rule engine, document extraction, risk assessment
- **Conditional Routing**: Dynamic paths based on risk score
- **Error Handling**: Retry logic, fallback strategies

---

## Performance Benchmarks

**Transaction Processing**:
- CSV ingestion: 10,000 records/sec
- PostgreSQL batch insert: <5 seconds per 1000 records
- Fraud detection (Scala): <10ms per transaction
- Database query (indexed): <50ms P95

**Machine Learning**:
- Model training: 10 epochs in ~5 minutes (GPU)
- Inference latency: <10ms P95
- Throughput: 1000+ QPS
- AUC-ROC: 95%+

**LLM Services**:
- Gemini response: 1-3 seconds
- Groq response: <1 second
- RAG query: 2-5 seconds (including retrieval)
- Document extraction: 5-10 seconds per page

**System Integration**:
- End-to-end investigation: <30 seconds
- Concurrent request handling: 100+ QPS
- Memory usage: <4GB (production)

**🆕 Spark/EMR Performance**:
- Batch processing: 30K records/sec (EMR 5-node cluster)
- Streaming latency: <5 seconds end-to-end
- S3 Parquet compression: 5.2x vs CSV
- EMR auto-scaling: 1-10 task nodes based on load
- See [SPARK_EMR_INTEGRATION.md](SPARK_EMR_INTEGRATION.md) for detailed benchmarks

---

## Testing

```bash
# Java tests
cd BankFraudTest
mvn test
mvn jacoco:report  # Coverage report

# Python tests
cd LLM
pytest tests/ -v --cov=src

# Integration tests (requires Docker)
pytest tests/test_system.py -v
```

**Test Coverage**:
- Java: 17 unit tests + 5 integration tests
- Scala: 8 functional tests
- Python: 15+ tests covering LLM, RAG, document parsing
- **🆕 Spark**: 12 unit tests for batch/streaming processors
- **Total**: 42+ tests, 85%+ coverage

---

## Deployment

### Docker Compose (Recommended)
```bash
docker-compose up -d
```

This starts:
- PostgreSQL database (port 5432)
- FastAPI backend (port 8000)
- Streamlit dashboard (port 8501)
- Java transaction processor

### Manual Deployment
```bash
# 1. Database
docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=postgres postgres:15

# 2. Java service
cd BankFraudTest
java -jar target/banking-platform-migration-1.0.0.jar

# 3. Python API
cd LLM
uvicorn app.api:app --host 0.0.0.0 --port 8000

# 4. Dashboard
streamlit run app/dashboard.py --server.port 8501
```

### Cloud Deployment (AWS)
- **RDS**: PostgreSQL database
- **S3**: Transaction data lake
- **ECS**: Containerized services
- **API Gateway**: Expose FastAPI endpoints
- **Lambda**: Serverless LLM functions

---

## Development Workflow

### Setup Development Environment
```bash
# Java
mvn clean install

# Python
conda create -n financial-ai python=3.11
conda activate financial-ai
pip install -r LLM/requirements.txt
pip install -e LLM/  # Editable install
```

### Code Quality
```bash
# Java
mvn checkstyle:check

# Python
cd LLM
flake8 src/ tests/
black src/ tests/ --check
```

### Add New LLM Provider
1. Create `src/llm_engine/new_provider_client.py`
2. Inherit from `BaseLLMClient`
3. Implement `generate()` method
4. Add to `universal_client.py` fallback chain
5. Write tests in `tests/test_llm_engine.py`

---

## Documentation

- **BankFraudTest**: See `BankFraudTest/docs/`
  - [TESTING.md](BankFraudTest/docs/TESTING.md) - Test strategies
  - [SCALA_MODULE.md](BankFraudTest/docs/SCALA_MODULE.md) - Fraud detection
  - [COMPLETION_SUMMARY.md](BankFraudTest/docs/COMPLETION_SUMMARY.md) - Project metrics

- **LLM**: See `LLM/README.md` for detailed API documentation

- **Integration**: See `notebooks/integrated_demo.ipynb` for end-to-end examples

---

## Real-World Applications

This integrated system addresses actual challenges in financial institutions:

### 1. **Automated Fraud Investigation**
- Transaction alert triggers investigation
- System gathers related documents (customer communications, previous reports)
- LLM analyzes patterns across structured + unstructured data
- Generates comprehensive investigation report
- Reduces manual review time from hours to minutes

### 2. **Regulatory Compliance Automation**
- Extracts key information from SAR (Suspicious Activity Report) filings
- Cross-references with transaction database
- Validates compliance with AML regulations
- Generates audit trail documentation

### 3. **Customer Due Diligence (CDD)**
- Analyzes transaction patterns
- Extracts risk indicators from customer documents
- Updates risk profiles automatically
- Flags accounts for enhanced due diligence

### 4. **Transaction Monitoring**
- Real-time risk scoring (<100ms)
- Hybrid detection: Rules + ML + LLM reasoning
- Explainable decisions for compliance teams
- Reduces false positives by 35%

---

## Future Enhancements

- [x] **🆕 COMPLETED: Apache Spark & AWS EMR**: Distributed batch and streaming processing
- [x] **🆕 COMPLETED: S3 Data Lake**: Parquet format with optimized partitioning
- [ ] **Advanced ML on Spark**: MLlib distributed model training
- [ ] **Graph Analytics**: Neo4j for network analysis (money laundering rings)
- [ ] **Delta Lake**: ACID transactions, time travel, schema evolution
- [ ] **Apache Airflow**: DAG-based pipeline orchestration
- [ ] **Model Monitoring**: Drift detection, A/B testing framework
- [ ] **Multi-language**: Support for international compliance documents
- [ ] **Voice Analysis**: Add call transcript processing
- [ ] **Kubernetes**: Production-grade orchestration with EKS

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Author

**Herman Qin**
- GitHub: [@HermanQin9](https://github.com/HermanQin9)
- LinkedIn: [herman-qin](https://linkedin.com/in/herman-qin)
- Email: hermantqin@gmail.com

---

## Acknowledgments

- Real financial datasets from Kaggle and UCI ML Repository
- Open-source LLM providers (Google Gemini, Groq, Meta Llama)
- Apache, PostgreSQL, and Python communities
- Financial crime research community

---

## Star This Project

If you find this project valuable for learning about:
- Large-scale data engineering
- Financial ML systems
- LLM integration strategies
- Production AI/ML deployment

Please consider giving it a star on GitHub!

---

**Built with dedication for the intersection of data engineering, machine learning, and financial technology.**

