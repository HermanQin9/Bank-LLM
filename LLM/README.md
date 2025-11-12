# Multi-LLM Document Intelligence System

A production-ready document processing system with unified access to multiple LLM providers (Gemini, Groq, OpenRouter, Hugging Face) for intelligent document understanding, information extraction, and retrieval-augmented generation.

## Overview

This system provides advanced AI-powered document processing capabilities:

- **Multi-Provider LLM Integration**: Seamless access to Gemini, Groq, OpenRouter, and Hugging Face models
- **Universal Client**: Automatic fallback between providers for maximum reliability
- **LangGraph Agents**: Sophisticated workflow orchestration with state management
- **GPU Training Pipeline**: Production-grade deep learning with multi-GPU support
- **Document Processing**: PDF parsing and OCR capabilities
- **RAG System**: Retrieval-Augmented Generation with vector-based semantic search
- **Scalable Architecture**: Distributed data processing for enterprise scale
- **Production Ready**: Comprehensive monitoring, error handling, and logging
- **Web Applications**: Interactive Streamlit dashboard and FastAPI REST API

## Key Features

### 🤖 Advanced GenAI Capabilities
- **LangGraph Agent Workflows**: Multi-step document processing with conditional routing
  - Automatic document classification and routing
  - Iterative extraction with validation loops
  - Multi-agent collaboration patterns
  - State management and error recovery
- **GPU-Accelerated Training**: Production PyTorch pipelines
  - Multi-GPU support with DataParallel/DistributedDataParallel
  - Mixed precision training (FP16) for 2-3x speedup
  - Model quantization and ONNX export
  - Gradient accumulation for large batch training
- **Scalable Data Processing**: Enterprise-grade pipelines
  - Distributed processing with Dask (1M+ documents/hour)
  - Multi-process parallelization across CPU cores
  - Streaming support for unlimited dataset sizes
  - Data quality validation and monitoring

### Multi-Provider LLM Support
- **Google Gemini**: Advanced reasoning with gemini-2.5-flash model (REST API)
- **Groq**: Ultra-fast inference with llama-3.3-70b-versatile
- **OpenRouter**: Unified access to 100+ models (qwen/qwen3-coder:free)
- **Hugging Face**: Local and hosted model support
- **Universal Client**: Automatic provider selection and fallback mechanism

### Document Understanding
- Multi-format document parsing (PDF, images, text)
- OCR for scanned documents (Tesseract/EasyOCR)
- Layout analysis and preprocessing
- Structured information extraction

### RAG System
- **Gemini Embeddings**: Production-grade semantic embeddings (768 dimensions)
- Vector store with persistent storage
- Semantic document search with cosine similarity
- Context-aware question answering with source citation
- Multi-document reasoning
- Reranking support for improved relevance

## Quick Start

### Prerequisites

- Python 3.8+ (tested on Python 3.8.5) **OR** Docker
- API keys for desired LLM providers
- 4GB+ RAM recommended

### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/HermanQin9/LLM.git
cd LLM

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Start with Docker Compose
docker-compose up -d

# Access services
# - API: http://localhost:8000/docs
# - Dashboard: http://localhost:8501
```

### Option 2: Local Installation

```bash
# Clone the repository
git clone https://github.com/HermanQin9/LLM.git
cd LLM

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
# For Python 3.8 compatibility:
# pip install -r requirements_py38.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

## Configuration

### Environment Setup

1. Copy the example configuration:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your API keys:
   ```env
   # Google Gemini
   GOOGLE_API_KEY=your_gemini_api_key
   GEMINI_MODEL=models/gemini-2.5-flash
   GEMINI_MAX_TOKENS=8192

   # Groq
   GROQ_API_KEY=your_groq_api_key
   GROQ_MODEL=llama-3.3-70b-versatile

   # OpenRouter
   OPENROUTER_API_KEY=your_openrouter_api_key
   OPENROUTER_MODEL=qwen/qwen3-coder:free

   # Hugging Face
   HUGGINGFACE_API_KEY=your_hf_api_key
   HUGGINGFACE_MODEL=mistralai/Mistral-7B-Instruct-v0.2
   ```

**Note:** `.env` is git-ignored and will not be committed. Never commit API keys to version control.

### Get API Keys

- **Gemini**: https://makersuite.google.com/app/apikey
- **Groq**: https://console.groq.com/keys (14,000 free requests/day)
- **OpenRouter**: https://openrouter.ai/keys
- **Hugging Face**: https://huggingface.co/settings/tokens

## Usage

### Run Applications

**1. Interactive Streamlit Dashboard**
```bash
streamlit run app/dashboard.py
```
Open http://localhost:8501 in your browser

**2. REST API Server**
```bash
python app/api.py
# or
uvicorn app.api:app --reload --port 8000
```
API docs: http://localhost:8000/docs

**3. Command Line Interface**
```bash
python src/main.py --input data/sample_documents/ --output results/
```

**4. Jupyter Notebook Demo**
```bash
jupyter notebook notebooks/01_document_understanding_demo.ipynb
```

## Project Structure

```
LLM/
├── src/                          # Source code
│   ├── llm_engine/               # LLM provider clients
│   │   ├── base_llm_client.py    # Abstract base class
│   │   ├── gemini_client.py      # Gemini SDK client
│   │   ├── gemini_client_v2.py   # Gemini REST API client (Python 3.8 compatible)
│   │   ├── groq_client.py        # Groq API client
│   │   ├── openrouter_client.py  # OpenRouter API client
│   │   ├── huggingface_client.py # Hugging Face client
│   │   ├── local_llm_client.py   # Local model support
│   │   ├── universal_client.py   # Multi-provider with fallback
│   │   ├── prompt_templates.py   # Prompt engineering templates
│   │   └── prompt_optimizer.py   # Token optimization
│   │
│   ├── document_parser/          # Document processing
│   │   ├── pdf_parser.py         # PDF extraction
│   │   ├── ocr_engine.py         # OCR integration
│   │   └── preprocessor.py       # Text preprocessing
│   │
│   ├── rag_system/               # RAG implementation
│   │   ├── vector_store.py       # Vector database wrapper
│   │   ├── retriever.py          # Semantic search
│   │   ├── generator.py          # Answer generation
│   │   └── rag_pipeline.py       # End-to-end pipeline
│   │
│   ├── utils/                    # Utilities
│   │   ├── config.py             # Configuration management
│   │   ├── logger.py             # Logging setup
│   │   └── helpers.py            # Helper functions
│   │
│   └── main.py                   # CLI entry point
│
├── app/                          # Web applications
│   ├── dashboard.py              # Streamlit UI
│   ├── api.py                    # FastAPI endpoints
│   └── README.md                 # App documentation
│
├── tests/                        # Unit tests
│   ├── test_llm_engine.py        # LLM client tests
│   ├── test_document_parser.py   # Parser tests
│   ├── test_system.py            # System integration tests
│   └── test_universal_llm.py     # Universal client tests
│
├── notebooks/                    # Jupyter notebooks
│   └── 01_document_understanding_demo.ipynb
│
├── data/                         # Data directory
│   ├── raw/                      # Raw documents
│   ├── processed/                # Processed data
│   └── sample_documents/         # Demo samples
│
├── results/                      # Output results
│   ├── extracted_data/
│   └── evaluations/
│
├── configs/                      # Configuration files
│   └── README.md
│
├── logs/                         # Application logs
│
├── .env                          # Environment variables (not in git)
├── .env.example                  # Environment template
├── requirements.txt              # Python dependencies
├── requirements_py38.txt         # Python 3.8 specific deps
└── README.md                     # This file
```

## Technical Stack

### Core Technologies
- **Python 3.8+**: Primary programming language
- **LLM Providers**: Google Gemini, Groq, OpenRouter, Hugging Face
- **ML Frameworks**: PyTorch, TensorFlow, Scikit-learn
- **NLP**: spaCy, Transformers, NLTK, Sentence-Transformers

### Document Processing
- **PyPDF2/pdfplumber**: PDF parsing
- **Tesseract/EasyOCR**: OCR capabilities
- **Pillow**: Image processing

### RAG & Vector Databases
- **LangChain**: RAG framework
- **ChromaDB/FAISS**: Vector storage and similarity search
- **Sentence-Transformers**: Text embeddings

### Web Applications
- **Streamlit**: Interactive dashboard
- **FastAPI**: REST API framework
- **Uvicorn**: ASGI server
- **Pydantic**: Data validation

### DevOps & Deployment
- **Docker**: Containerization
- **Docker Compose**: Multi-container orchestration
- **GitHub Actions**: CI/CD pipeline
- **pytest**: Automated testing
- **flake8/black**: Code quality

### Development Tools
- **python-dotenv**: Environment management
- **loguru**: Advanced logging
- **Makefile**: Task automation
- **Git**: Version control

## LLM Provider Details

### Gemini (Google)
- **Model**: gemini-2.5-flash
- **Implementation**: REST API (Python 3.8 compatible)
- **Features**: Advanced reasoning, multimodal support
- **Rate Limits**: Generous free tier
- **Best For**: Complex document understanding, multilingual tasks

### Groq
- **Model**: llama-3.3-70b-versatile
- **Speed**: Ultra-fast inference (< 1s typical)
- **Rate Limits**: 14,400 requests/day free tier
- **Best For**: High-throughput applications, real-time processing

### OpenRouter
- **Model**: qwen/qwen3-coder:free (configurable to 100+ models)
- **Features**: Unified API for multiple providers
- **Best For**: Model comparison, cost optimization

### Hugging Face
- **Support**: Both API and local inference
- **Models**: 100,000+ open-source models
- **Best For**: Custom models, offline deployment

## Usage Examples

### Using Universal Client (Recommended)

```python
from src.llm_engine.universal_client import UniversalLLMClient

# Automatic provider selection with fallback
client = UniversalLLMClient()
response = client.generate("Explain machine learning in one sentence")
print(response)
```

### Using Specific Providers

```python
from src.llm_engine.gemini_client_v2 import GeminiClientV2
from src.llm_engine.groq_client import GroqClient
from src.llm_engine.openrouter_client import OpenRouterClient

# Gemini
gemini = GeminiClientV2()
response = gemini.generate("Analyze this document...", max_tokens=2000)

# Groq (fastest)
groq = GroqClient()
response = groq.generate("Summarize...", temperature=0.7)

# OpenRouter (most models)
openrouter = OpenRouterClient()
response = openrouter.generate("Extract information...")
```

### RAG System

```python
from src.rag_system.gemini_rag_pipeline import GeminiRAGPipeline

# Initialize RAG with Gemini embeddings
rag = GeminiRAGPipeline(llm_provider="gemini")

# Add documents (automatically chunks and embeds with Gemini)
rag.index_documents([
    "path/to/document1.pdf",
    "path/to/document2.pdf"
])

# Query with semantic search
answer = rag.query("What are the key findings?", top_k=5)
print(answer['answer'])
print(f"Sources used: {answer['num_sources']}")

# Get search results only (no generation)
results = rag.semantic_search_only("key findings", top_k=3)
for result in results:
    print(f"Score: {result['score']:.3f} - {result['document'][:100]}")
```

### Document Processing

```python
from src.document_parser.pdf_parser import PDFParser
from src.document_parser.ocr_engine import OCREngine

# Parse PDF
parser = PDFParser()
text = parser.extract_text("document.pdf")

# OCR for scanned documents
ocr = OCREngine()
text = ocr.extract_text("scanned_document.pdf")
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test suite
pytest tests/test_llm_engine.py -v

# Run with coverage
pytest --cov=src tests/

# Test specific provider
python -m pytest tests/test_llm_engine.py::test_gemini_client -v
```

### CI/CD Pipeline

This project uses **GitHub Actions** for automated testing and deployment:

- **Continuous Integration**: Runs on every push/PR
  - Multi-version Python testing (3.8, 3.9)
  - Code linting (flake8)
  - Code formatting (black)
  - Security scanning (safety)
  - Docker build verification

- **Continuous Deployment**: Runs on main branch
  - Automated Docker image builds
  - GitHub releases for tagged versions

See [CI_CD_GUIDE.md](CI_CD_GUIDE.md) for detailed CI/CD documentation.

## Configuration

The system uses environment variables for configuration. Key settings:

```env
# LLM Provider Settings
GEMINI_MAX_TOKENS=8192
GROQ_MAX_TOKENS=4096
OPENROUTER_MAX_TOKENS=4096

# RAG Settings
VECTOR_STORE_TYPE=chromadb  # or faiss
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=512
CHUNK_OVERLAP=50

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/app.log
```

## Troubleshooting

### Python 3.8 Compatibility

If using Python 3.8, use `requirements_py38.txt` which includes:
- google-generativeai compatible version
- Gemini REST API client (gemini_client_v2.py) instead of SDK

### Common Issues

**API Key Errors**
```bash
# Verify .env file exists and contains valid keys
cat .env
```

**Import Errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

**Gemini MAX_TOKENS Error**
- Increase `GEMINI_MAX_TOKENS` in .env (recommended: 8192)
- Use gemini_client_v2.py (REST API) instead of SDK client

## Deployment

### Docker Deployment

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Manual Deployment

```bash
# Start API server
uvicorn app.api:app --host 0.0.0.0 --port 8000

# Start Dashboard (separate terminal)
streamlit run app/dashboard.py --server.port 8501
```

### Using Makefile

```bash
# View all available commands
make help

# Install dependencies
make install

# Run tests
make test

# Start with Docker
make docker-up

# Stop Docker services
make docker-down
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure CI/CD checks pass
6. Submit a pull request

**Development workflow**:
```bash
# Install dev dependencies
pip install pytest pytest-cov flake8 black

# Format code
make format

# Run linting
make lint

# Run tests
make test
```

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Author

**Herman Qin**
- GitHub: [@HermanQin9](https://github.com/HermanQin9)
- Repository: [LLM](https://github.com/HermanQin9/LLM)

## Acknowledgments

- Built with support from Google Gemini, Groq, OpenRouter, and Hugging Face
- Inspired by enterprise document processing challenges
- Designed for production reliability and multi-provider flexibility

---

**Note**: This project demonstrates production-ready AI/ML capabilities with emphasis on:
- Multi-provider reliability and fallback mechanisms
- Python 3.8+ compatibility
- Comprehensive error handling and logging
- Real-world document intelligence applications
