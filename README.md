# Financial Intelligence & Compliance Platform

[![Java](https://img.shields.io/badge/Java-21-orange.svg)](https://www.oracle.com/java/)
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Scala](https://img.shields.io/badge/Scala-2.13-red.svg)](https://www.scala-lang.org/)
[![Apache Spark](https://img.shields.io/badge/Spark-3.5-E25A1C.svg)](https://spark.apache.org/)

Integrated system for transaction monitoring, fraud detection, and document intelligence using Java, Scala, Python, and Apache Spark.

## Architecture

**Deep Integration Pattern**: Single PostgreSQL database as shared state, bidirectional data flow between Java/Scala transaction engine and Python ML/LLM intelligence layer.

```
Transaction Engine (Java/Scala) → PostgreSQL ← Intelligence Engine (Python ML/LLM)
                                       ↓
                                   S3 Data Lake
                                       ↓
                            Spark/EMR (Distributed Processing)
```

## Technology Stack

**Backend**
- Java 21: Transaction processing, ETL pipeline
- Scala 2.13: Functional fraud detection, Spark jobs
- Python 3.11: ML models, LLM integration
- Apache Spark 3.5: Distributed batch and streaming processing

**Data Layer**
- PostgreSQL 15: Shared database (2.2M+ transactions)
- AWS S3: Data lake with Parquet format
- AWS EMR: Managed Spark clusters with auto-scaling

**ML/AI**
- PyTorch 2.0: Deep learning models (95%+ AUC-ROC)
- Transformers: NLP and embeddings
- Google Gemini: Advanced reasoning
- Groq: Ultra-fast inference
- ChromaDB: Vector storage for RAG

**Infrastructure**
- Docker Compose: Local development
- Kubernetes: Production deployment
- HikariCP: Connection pooling
- Flyway: Database migrations

## Features

### Transaction Processing
- Multi-format ETL (CSV, JSON, fixed-width)
- 10K records/sec throughput
- Real-time fraud detection (<100ms)
- Batch processing: 30K records/sec on Spark

### Fraud Detection
- Rule-based engine (Scala functional programming)
- Deep learning models (PyTorch with GPU)
- Distributed analysis (Spark on EMR)
- Real-time streaming (Kafka/Kinesis)

### Document Intelligence
- Multi-LLM integration (Gemini, Groq, OpenRouter)
- PDF/document parsing
- RAG system for semantic search
- Compliance document extraction

### Big Data Processing
- Spark batch processing (100M+ transactions)
- Structured Streaming (<5s latency)
- S3 data lake (Parquet, 5.2x compression)
- EMR auto-scaling (1-10 task nodes)

## Quick Start

### Prerequisites
```bash
java --version     # Java 21
python --version   # Python 3.11
mvn --version      # Maven 3.9+
docker --version   # Docker
```

### Setup

1. Clone repository
```bash
git clone https://github.com/HermanQin9/Bank-LLM.git
cd BankFraudTest-LLM
```

2. Start PostgreSQL
```bash
cd BankFraudTest
docker-compose up -d
```

3. Build Java project
```bash
mvn clean install
mvn flyway:migrate
```

4. Setup Python environment
```bash
cd ../LLM
pip install -r requirements.txt
```

5. Run integration demo
```bash
# Start Python API
python -m uvicorn app.integration_api:app --port 8000

# In another terminal, run Java demo
cd ../BankFraudTest
mvn exec:java -Dexec.mainClass="com.bankfraud.integration.DeepIntegrationDemo"
```

## Testing

### Java Tests
```bash
cd BankFraudTest
mvn test
mvn jacoco:report
```

### Python Tests
```bash
cd LLM
pytest tests/ --cov=src --cov-report=html -v
```

### Scala/Spark Tests
```bash
cd BankFraudTest
mvn test -Dtest=SparkTransactionProcessorTest
```

## Performance

| Component | Throughput | Latency |
|-----------|------------|---------|
| Java ETL | 10K records/sec | N/A |
| Fraud detection (Scala) | N/A | <10ms |
| ML inference (PyTorch) | 1000+ QPS | <10ms |
| Spark batch (EMR) | 30K records/sec | N/A |
| Spark streaming | 10K events/sec | <5s |

## Spark/EMR Integration

### Batch Processing
```bash
cd BankFraudTest/src/main/scripts
./create-emr-cluster.sh
./submit-spark-job.sh <cluster-id> batch \
  s3://bank-fraud-data/transactions/ \
  s3://bank-fraud-data/fraud-alerts/
```

### Configuration
- Master: 1x m5.xlarge
- Core: 2x m5.xlarge
- Task: 1-10x m5.xlarge (auto-scaling, spot instances)
- EMR 6.15.0 with Spark, Hadoop, Hive

## Project Structure

```
BankFraudTest/              # Java/Scala transaction engine
├── src/main/java/          # Java services and ETL
├── src/main/scala/         # Scala fraud detection + Spark jobs
├── src/main/scripts/       # EMR deployment scripts
├── src/test/               # Java/Scala tests
└── pom.xml                 # Maven configuration

LLM/                        # Python ML/LLM engine
├── src/                    # Source code
│   ├── llm_engine/         # Multi-LLM clients
│   ├── rag_system/         # Vector store and RAG
│   ├── document_parser/    # PDF/document processing
│   └── utils/              # Utilities
├── app/                    # FastAPI services
├── tests/                  # Python tests
└── requirements.txt        # Dependencies

unified-intelligence/       # Integration layer
├── schema_adapter.py       # Java ↔ Python schema conversion
├── database_bridge.py      # Shared database access
└── shared_models.py        # Common data models
```

## CI/CD

GitHub Actions automatically runs tests on every push:
- Java: Maven tests with JaCoCo coverage
- Python: Pytest with coverage reporting
- Scala: Spark unit tests
- Code linting: flake8, black

See `.github/workflows/ci.yml` for configuration.

## Documentation

- `DEEP_INTEGRATION.md`: Architecture and integration patterns
- Javadoc: `mvn javadoc:javadoc` → `target/site/apidocs/`
- Python docs: `pytest --doctest-modules`

## Database Schema

```sql
transactions          # Java writes, Python reads
customer_profiles     # Python writes, Java reads
fraud_alerts         # Python writes, Java reads
transaction_alerts   # Both write
document_evidence    # Python LLM system
```

## Cost Optimization

**EMR Cluster**
- On-demand: $1.15/hour
- With spot instances: $0.75/hour (35% savings)
- Auto-scaling: Averages $0.60/hour

**S3 Storage**
- Parquet compression: 5.2x vs CSV
- Partition pruning reduces read costs

## License

MIT License

## Author

Herman Qin
- GitHub: [@HermanQin9](https://github.com/HermanQin9)
- Email: hermantqin@gmail.com

