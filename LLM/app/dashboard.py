"""Streamlit dashboard for Document Intelligence System."""

import streamlit as st
import sys
import os
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.document_parser import PDFParser, TextPreprocessor
from src.llm_engine import GeminiClientV2, PromptTemplates, UniversalLLMClient
from src.utils import Config

# Lazy import RAG to avoid TensorFlow DLL errors at startup
RAGPipeline = None

# Page configuration
st.set_page_config(
    page_title="Multi-LLM Document Intelligence",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'client' not in st.session_state:
    try:
        # Use Universal Client for multi-provider support
        st.session_state.client = UniversalLLMClient(provider='auto', enable_fallback=True)
        
        # Use Gemini embeddings RAG system for production-grade semantic search
        from src.rag_system.gemini_rag_pipeline import GeminiRAGPipeline
        st.session_state.rag = GeminiRAGPipeline()
        st.session_state.rag_available = True
        
        st.session_state.parser = PDFParser()
        st.session_state.preprocessor = TextPreprocessor()
        st.session_state.initialized = True
        st.session_state.provider_info = {
            'primary': st.session_state.client.provider,
            'fallbacks': len(st.session_state.client.fallback_clients)
        }
    except Exception as e:
        st.session_state.initialized = False
        st.session_state.error = str(e)

if 'documents_indexed' not in st.session_state:
    st.session_state.documents_indexed = 0

if 'processing_history' not in st.session_state:
    st.session_state.processing_history = []

if 'model_performance' not in st.session_state:
    st.session_state.model_performance = {
        'gemini': {'requests': 0, 'success': 0, 'avg_latency': 0},
        'groq': {'requests': 0, 'success': 0, 'avg_latency': 0},
        'openrouter': {'requests': 0, 'success': 0, 'avg_latency': 0}
    }


def main():
    """Main dashboard function."""
    
    # Header
    st.markdown('<div class="main-header">Multi-LLM Document Intelligence System</div>', 
                unsafe_allow_html=True)
    
    # Check initialization
    if not st.session_state.initialized:
        st.error(f"Failed to initialize system: {st.session_state.get('error', 'Unknown error')}")
        st.info("Please check your .env file and ensure at least one LLM provider API key is set.")
        st.markdown("""
        **Supported Providers:**
        - `GOOGLE_API_KEY` for Gemini
        - `GROQ_API_KEY` for Groq (fastest)
        - `OPENROUTER_API_KEY` for OpenRouter (100+ models)
        - `HUGGINGFACE_API_KEY` for Hugging Face
        """)
        return
    
    # Success banner
    provider_info = st.session_state.get('provider_info', {})
    st.success(f"✓ System Ready | Primary: {provider_info.get('primary', 'N/A').upper()} | Fallbacks: {provider_info.get('fallbacks', 0)}")
    
    # Sidebar
    with st.sidebar:
        st.header("🧭 Navigation")
        page = st.radio(
            "Select Feature",
            ["📊 Dashboard Overview", "📄 Document Classification", "🔍 Information Extraction", 
             "🏷️ Entity Recognition", "💬 RAG Q&A System", "🔧 Prompt Engineering", 
             "📈 Analytics & Performance", "⚙️ Model Comparison"]
        )
        
        st.markdown("---")
        st.subheader("📊 System Status")
        st.metric("Documents Indexed", st.session_state.documents_indexed)
        st.metric("Operations", len(st.session_state.processing_history))
        st.metric("Active Provider", provider_info.get('primary', 'N/A').upper())
        
        st.markdown("---")
        st.subheader("⚙️ Configuration")
        current_provider = st.session_state.get('provider_info', {}).get('primary', 'auto')
        st.text(f"Provider: {current_provider}")
        st.text(f"Fallback: {'Enabled' if provider_info.get('fallbacks', 0) > 0 else 'Disabled'}")
    
    # Main content area
    if page == "📊 Dashboard Overview":
        dashboard_overview_page()
    elif page == "📄 Document Classification":
        document_classification_page()
    elif page == "🔍 Information Extraction":
        information_extraction_page()
    elif page == "🏷️ Entity Recognition":
        entity_recognition_page()
    elif page == "💬 RAG Q&A System":
        rag_qa_page()
    elif page == "🔧 Prompt Engineering":
        prompt_engineering_page()
    elif page == "📈 Analytics & Performance":
        analytics_page()
    elif page == "⚙️ Model Comparison":
        model_comparison_page()


def dashboard_overview_page():
    """Dashboard overview with key metrics."""
    st.header("📊 System Overview")
    
    # Key metrics in cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Operations",
            value=len(st.session_state.processing_history),
            delta="+1" if st.session_state.processing_history else None
        )
    
    with col2:
        total_tokens = sum(item.get('tokens', 0) for item in st.session_state.processing_history)
        st.metric(
            label="Total Tokens Used",
            value=f"{total_tokens:,}",
            delta=None
        )
    
    with col3:
        st.metric(
            label="Documents Indexed",
            value=st.session_state.documents_indexed,
            delta=None
        )
    
    with col4:
        provider = st.session_state.get('provider_info', {}).get('primary', 'N/A')
        st.metric(
            label="Active Provider",
            value=provider.upper(),
            delta=None
        )
    
    st.markdown("---")
    
    # System capabilities
    st.subheader("🚀 System Capabilities")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📄 Document Processing**")
        st.markdown("- PDF Parsing")
        st.markdown("- OCR Recognition")
        st.markdown("- Text Preprocessing")
        st.markdown("- Multi-format Support")
    
    with col2:
        st.markdown("**🤖 LLM Features**")
        st.markdown("- Multi-provider Support")
        st.markdown("- Auto Fallback")
        st.markdown("- Prompt Optimization")
        st.markdown("- Token Management")
    
    with col3:
        st.markdown("**🔍 AI Capabilities**")
        st.markdown("- Document Classification")
        st.markdown("- Information Extraction")
        st.markdown("- Entity Recognition")
        st.markdown("- RAG Q&A System")
    
    st.markdown("---")
    
    # Provider status
    st.subheader("🌐 LLM Provider Status")
    
    providers_data = []
    for provider, stats in st.session_state.model_performance.items():
        success_rate = (stats['success'] / stats['requests'] * 100) if stats['requests'] > 0 else 0
        providers_data.append({
            'Provider': provider.capitalize(),
            'Requests': stats['requests'],
            'Success': stats['success'],
            'Success Rate': f"{success_rate:.1f}%",
            'Avg Latency (s)': f"{stats['avg_latency']:.2f}"
        })
    
    df_providers = pd.DataFrame(providers_data)
    st.dataframe(df_providers, use_container_width=True)
    
    # Quick actions
    st.markdown("---")
    st.subheader("⚡ Quick Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📄 Classify Document", use_container_width=True):
            st.session_state.current_page = "Document Classification"
            st.rerun()
    
    with col2:
        if st.button("🔍 Extract Info", use_container_width=True):
            st.session_state.current_page = "Information Extraction"
            st.rerun()
    
    with col3:
        if st.button("💬 Ask Question", use_container_width=True):
            st.session_state.current_page = "RAG Q&A System"
            st.rerun()
    
    with col4:
        if st.button("📈 View Analytics", use_container_width=True):
            st.session_state.current_page = "Analytics"
            st.rerun()


def document_classification_page():
    """Document classification interface."""
    st.header("Document Classification")
    st.markdown("Classify documents into predefined categories using Gemini.")
    
    # Initialize session state for document text
    if 'classification_document_text' not in st.session_state:
        st.session_state.classification_document_text = ""
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Input Document")
        document_text = st.text_area(
            "Paste document text here",
            value=st.session_state.classification_document_text,
            height=200,
            placeholder="Enter or paste your document text..."
        )
        
        # Update session state
        st.session_state.classification_document_text = document_text
        
        # Predefined categories
        default_categories = ["Contract", "Invoice", "Report", "Form", "Letter", "Email", "Other"]
        categories_input = st.text_input(
            "Categories (comma-separated)",
            value=", ".join(default_categories)
        )
        categories = [cat.strip() for cat in categories_input.split(",") if cat.strip()]
        
        classify_button = st.button("Classify Document", type="primary")
    
    with col2:
        st.subheader("Quick Examples")
        if st.button("Load Contract Example"):
            st.session_state.classification_document_text = "This Agreement is made on January 1, 2024, between Party A (ABC Corporation) and Party B (XYZ Limited). Both parties hereby agree to the following terms and conditions. The contract duration shall be 12 months starting from the effective date. Party A agrees to provide services as outlined in Appendix A, while Party B agrees to compensate Party A according to the payment schedule in Section 3."
            st.rerun()
        if st.button("Load Invoice Example"):
            st.session_state.classification_document_text = "INVOICE #INV-2024-12345\nDate: January 15, 2024\nDue Date: February 15, 2024\n\nBill To: ABC Corporation\n123 Business Street\nNew York, NY 10001\n\nItems:\n1. Consulting Services (40 hours) - $4,000.00\n2. Software License - $1,000.00\n\nSubtotal: $5,000.00\nTax (8%): $400.00\nTotal Amount Due: $5,400.00\n\nPayment Terms: Net 30 days"
            st.rerun()
        if st.button("Load Report Example"):
            st.session_state.classification_document_text = "Q4 2023 Financial Performance Report\n\nExecutive Summary:\nOur company achieved strong financial results in Q4 2023. Revenue increased by 15% compared to Q4 2022, reaching $12.5 million. Net profit margin improved to 18%, up from 15% in the previous year. Key growth drivers included expansion into new markets and successful product launches. Operating expenses were well-controlled at 45% of revenue. Looking ahead to 2024, we project continued growth with expected revenue increase of 12-15%."
            st.rerun()
    
    # Input validation
    if classify_button:
        if not document_text or len(document_text.strip()) < 10:
            st.warning("⚠️ Please enter a document with at least 10 characters.")
            return
        
        if not categories or len(categories) == 0:
            st.error("❌ Please provide at least one category.")
            return
    
    if classify_button and document_text:
        with st.spinner("Classifying document..."):
            try:
                import time
                start_time = time.time()
                
                prompt = PromptTemplates.document_classification(document_text, categories)
                result = st.session_state.client.generate_with_metadata(prompt)
                
                latency = time.time() - start_time
                
                # Clean up the classification result
                classification = result['text'].strip()
                
                # Validate classification result
                if classification not in categories:
                    # Try to find a close match (case-insensitive)
                    classification_lower = classification.lower()
                    matched = False
                    for cat in categories:
                        if cat.lower() == classification_lower:
                            classification = cat
                            matched = True
                            break
                    
                    if not matched:
                        st.warning(f"⚠️ Model returned '{classification}' which is not in the category list. This might indicate an issue.")
                
                st.success("✅ Classification Complete!")
                
                # Display results in a more prominent way
                st.markdown("### 📋 Classification Result")
                st.markdown(f"<div style='background-color: #e8f4f8; padding: 20px; border-radius: 10px; text-align: center;'><h2 style='color: #1f77b4; margin: 0;'>{classification}</h2></div>", unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Token usage metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📥 Prompt Tokens", result.get('prompt_tokens', 0))
                with col2:
                    st.metric("📤 Response Tokens", result.get('response_tokens', 0))
                with col3:
                    st.metric("⚡ Total Tokens", result.get('prompt_tokens', 0) + result.get('response_tokens', 0))
                with col4:
                    st.metric("⏱️ Latency", f"{latency:.2f}s")
                
                # Show document preview
                with st.expander("📄 View Document Preview"):
                    preview_length = min(500, len(document_text))
                    st.text(document_text[:preview_length] + ("..." if len(document_text) > preview_length else ""))
                
                # Store in history
                st.session_state.processing_history.append({
                    'timestamp': datetime.now(),
                    'operation': 'Classification',
                    'tokens': result.get('prompt_tokens', 0) + result.get('response_tokens', 0),
                    'latency': latency,
                    'result': classification
                })
                
                # Update provider performance
                provider = st.session_state.get('provider_info', {}).get('primary', 'unknown')
                if provider in st.session_state.model_performance:
                    perf = st.session_state.model_performance[provider]
                    perf['requests'] += 1
                    perf['success'] += 1
                    perf['avg_latency'] = (perf['avg_latency'] * (perf['requests'] - 1) + latency) / perf['requests']
                
            except Exception as e:
                st.error(f"❌ Classification failed: {str(e)}")
                
                # Show helpful error message
                st.info("💡 **Troubleshooting Tips:**\n"
                       "- Check your API keys in the `.env` file\n"
                       "- Verify network connection\n"
                       "- Try reducing document length\n"
                       "- The system will automatically try fallback providers")
                
                # Track failure
                provider = st.session_state.get('provider_info', {}).get('primary', 'unknown')
                if provider in st.session_state.model_performance:
                    st.session_state.model_performance[provider]['requests'] += 1


def information_extraction_page():
    """Information extraction interface."""
    st.header("Information Extraction")
    st.markdown("Extract structured information from documents.")
    
    # Initialize session state
    if 'extraction_document_text' not in st.session_state:
        st.session_state.extraction_document_text = ""
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        document_text = st.text_area(
            "Document Text",
            value=st.session_state.extraction_document_text,
            height=200,
            placeholder="Paste your document here..."
        )
        st.session_state.extraction_document_text = document_text
        
        fields_input = st.text_input(
            "Fields to Extract (comma-separated)",
            value="Date, Amount, Parties, Terms"
        )
        fields = [field.strip() for field in fields_input.split(",") if field.strip()]
        
        extract_button = st.button("Extract Information", type="primary")
    
    with col2:
        st.subheader("Common Fields")
        st.markdown("""
        - Date
        - Amount/Value
        - Parties Involved
        - Contract Terms
        - Payment Terms
        - Expiration Date
        - Contact Information
        """)
    
    # Input validation
    if extract_button:
        if not document_text or len(document_text.strip()) < 10:
            st.warning("⚠️ Please enter a document with at least 10 characters.")
            return
        
        if not fields or len(fields) == 0:
            st.error("❌ Please provide at least one field to extract.")
            return
    
    if extract_button and document_text:
        with st.spinner("Extracting information..."):
            try:
                import time
                start_time = time.time()
                
                prompt = PromptTemplates.information_extraction(document_text, fields)
                result = st.session_state.client.generate_with_metadata(prompt)
                
                latency = time.time() - start_time
                
                st.success("✅ Extraction Complete!")
                
                # Parse and display extracted information in a structured way
                st.markdown("### 📋 Extracted Information")
                
                extracted_text = result['text'].strip()
                
                # Try to parse the extracted information into a table
                try:
                    lines = extracted_text.split('\n')
                    data = []
                    for line in lines:
                        line = line.strip()
                        if ':' in line:
                            field, value = line.split(':', 1)
                            data.append({
                                'Field': field.strip(),
                                'Value': value.strip()
                            })
                    
                    if data:
                        # Display as a nice table
                        import pandas as pd
                        df = pd.DataFrame(data)
                        st.dataframe(df, use_container_width=True, hide_index=True)
                    else:
                        # Fallback to raw text
                        st.markdown(extracted_text)
                except:
                    # If parsing fails, show raw text
                    st.code(extracted_text, language=None)
                
                st.markdown("---")
                
                # Token usage
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📥 Prompt Tokens", result.get('prompt_tokens', 0))
                with col2:
                    st.metric("📤 Response Tokens", result.get('response_tokens', 0))
                with col3:
                    st.metric("⚡ Total Tokens", result.get('prompt_tokens', 0) + result.get('response_tokens', 0))
                with col4:
                    st.metric("⏱️ Latency", f"{latency:.2f}s")
                
                st.session_state.processing_history.append({
                    'timestamp': datetime.now(),
                    'operation': 'Extraction',
                    'tokens': result.get('prompt_tokens', 0) + result.get('response_tokens', 0),
                    'latency': latency
                })
                
            except Exception as e:
                st.error(f"❌ Extraction failed: {str(e)}")
                st.info("💡 Try reducing document length or checking API connection.")


def entity_recognition_page():
    """Named entity recognition interface."""
    st.header("Named Entity Recognition")
    st.markdown("Extract named entities from text documents.")
    
    # Initialize session state
    if 'entity_document_text' not in st.session_state:
        st.session_state.entity_document_text = ""
    
    document_text = st.text_area(
        "Document Text",
        value=st.session_state.entity_document_text,
        height=200,
        placeholder="Paste your document here..."
    )
    st.session_state.entity_document_text = document_text
    
    col1, col2 = st.columns(2)
    with col1:
        entity_types_input = st.text_input(
            "Entity Types (comma-separated)",
            value="Organizations, People, Locations, Dates, Monetary Amounts"
        )
        entity_types = [et.strip() for et in entity_types_input.split(",") if et.strip()]
    
    with col2:
        st.markdown("**Common Entity Types:**")
        st.markdown("Organizations, People, Locations, Dates, Amounts, Products")
    
    extract_entities_button = st.button("Extract Entities", type="primary")
    
    # Input validation
    if extract_entities_button:
        if not document_text or len(document_text.strip()) < 10:
            st.warning("⚠️ Please enter a document with at least 10 characters.")
            return
    
    if extract_entities_button and document_text:
        with st.spinner("Extracting entities..."):
            try:
                import time
                start_time = time.time()
                
                prompt = PromptTemplates.entity_extraction(document_text, entity_types)
                result = st.session_state.client.generate_with_metadata(prompt)
                
                latency = time.time() - start_time
                
                st.success("✅ Entity Extraction Complete!")
                
                st.markdown("### 🏷️ Extracted Entities")
                st.markdown(result['text'])
                
                st.markdown("---")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📥 Prompt Tokens", result.get('prompt_tokens', 0))
                with col2:
                    st.metric("📤 Response Tokens", result.get('response_tokens', 0))
                with col3:
                    st.metric("⚡ Total Tokens", result.get('prompt_tokens', 0) + result.get('response_tokens', 0))
                with col4:
                    st.metric("⏱️ Latency", f"{latency:.2f}s")
                
                st.session_state.processing_history.append({
                    'timestamp': datetime.now(),
                    'operation': 'Entity Recognition',
                    'tokens': result.get('prompt_tokens', 0) + result.get('response_tokens', 0),
                    'latency': latency
                })
                
            except Exception as e:
                st.error(f"❌ Entity extraction failed: {str(e)}")
                st.info("💡 Try reducing document length or simplifying entity types.")


def rag_qa_page():
    """RAG Q&A system interface."""
    st.header("💬 RAG Q&A System")
    st.markdown("Retrieval-Augmented Generation with **Google Gemini embeddings** for semantic search.")
    st.info("📌 Using Gemini embeddings (768-dim) for production-grade semantic understanding")
    
    # Check if RAG is initialized
    if 'rag' not in st.session_state or st.session_state.rag is None:
        st.warning("⚠️ RAG system is not initialized. Attempting to initialize...")
        try:
            from src.rag_system.gemini_rag_pipeline import GeminiRAGPipeline
            st.session_state.rag = GeminiRAGPipeline()
            st.session_state.rag_available = True
            st.success("✅ RAG system initialized with Gemini embeddings!")
        except Exception as e:
            st.error(f"❌ RAG initialization failed: {str(e)}")
            st.info("💡 The system will continue to work with other features.")
            return
    
    # Document indexing section
    st.subheader("1. Index Documents")
    
    documents_text = st.text_area(
        "Enter documents (one per line or separated by blank lines)",
        height=150,
        placeholder="Enter multiple documents to index..."
    )
    
    col1, col2 = st.columns([1, 3])
    with col1:
        index_button = st.button("Index Documents", type="primary")
    with col2:
        if st.session_state.documents_indexed > 0:
            st.info(f"Currently indexed: {st.session_state.documents_indexed} documents")
    
    if index_button and documents_text:
        with st.spinner("Indexing documents..."):
            try:
                # Split documents by double newline or single newline
                documents = [doc.strip() for doc in documents_text.split("\n\n") if doc.strip()]
                if not documents:
                    documents = [doc.strip() for doc in documents_text.split("\n") if doc.strip()]
                
                num_indexed = st.session_state.rag.index_documents(documents)
                st.session_state.documents_indexed = num_indexed
                
                st.success(f"✅ Successfully indexed {num_indexed} document chunks!")
                
            except Exception as e:
                st.error(f"Indexing failed: {str(e)}")
    
    st.markdown("---")
    
    # Q&A section
    st.subheader("2. Ask Questions")
    
    question = st.text_input(
        "Enter your question",
        placeholder="What would you like to know about the documents?"
    )
    
    col1, col2 = st.columns([1, 1])
    with col1:
        top_k = st.slider("Number of documents to retrieve", 1, 10, 3)
    with col2:
        show_sources = st.checkbox("Show source documents", value=True)
    
    ask_button = st.button("Get Answer", type="primary")
    
    if ask_button and question:
        if st.session_state.documents_indexed == 0:
            st.warning("⚠️ Please index some documents first!")
        else:
            with st.spinner("🔍 Searching and generating answer..."):
                try:
                    import time
                    start_time = time.time()
                    
                    result = st.session_state.rag.query(
                        question,
                        top_k=top_k,
                        include_sources=show_sources
                    )
                    
                    latency = time.time() - start_time
                    
                    st.success("✅ Answer Generated!")
                    
                    st.markdown("### 💡 Answer")
                    st.markdown(result['answer'])
                    
                    st.markdown("---")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("📚 Sources Used", result['num_sources'])
                    with col2:
                        st.metric("⏱️ Latency", f"{latency:.2f}s")
                    
                    if show_sources and 'sources' in result:
                        st.markdown("### 📄 Source Documents")
                        for i, source in enumerate(result['sources'], 1):
                            with st.expander(f"📑 Source {i} (Score: {source.get('score', 0):.3f})"):
                                st.text(source['document'])
                                if 'metadata' in source:
                                    st.caption(f"Metadata: {source['metadata']}")
                    
                    st.session_state.processing_history.append({
                        'timestamp': datetime.now(),
                        'operation': 'RAG Q&A',
                        'tokens': 0,  # SimpleRAG doesn't track tokens separately
                        'latency': latency
                    })
                    
                except Exception as e:
                    st.error(f"❌ Q&A failed: {str(e)}")
                    st.info("💡 Try rephrasing your question or indexing more relevant documents.")


def prompt_engineering_page():
    """Prompt engineering interface."""
    st.header("Prompt Engineering")
    st.markdown("Test and optimize prompts with advanced techniques.")
    
    tab1, tab2, tab3 = st.tabs(["Custom Prompt", "Chain-of-Thought", "Risk Analysis"])
    
    with tab1:
        st.subheader("Custom Prompt Testing")
        
        custom_prompt = st.text_area(
            "Enter your custom prompt",
            height=150,
            placeholder="Write your custom prompt here..."
        )
        
        col1, col2 = st.columns(2)
        with col1:
            temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
        with col2:
            max_tokens = st.slider("Max Tokens", 100, 4096, 1000, 100)
        
        if st.button("Generate Response", type="primary"):
            if custom_prompt:
                with st.spinner("Generating..."):
                    try:
                        result = st.session_state.client.generate_with_metadata(
                            custom_prompt,
                            temperature=temperature,
                            max_tokens=max_tokens
                        )
                        
                        st.success("Response Generated!")
                        st.markdown(result['text'])
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Prompt Tokens", result.get('prompt_tokens', 0))
                        with col2:
                            st.metric("Response Tokens", result.get('response_tokens', 0))
                        
                    except Exception as e:
                        st.error(f"Generation failed: {str(e)}")
    
    with tab2:
        st.subheader("Chain-of-Thought Reasoning")
        st.info("💡 Chain-of-Thought helps LLMs break down complex problems into steps.")
        
        cot_task = st.text_area(
            "Enter task for chain-of-thought reasoning",
            height=100,
            placeholder="Example: Calculate the total cost if I buy 3 items at $15 each and 2 items at $25 each, with 10% tax..."
        )
        
        if st.button("Apply Chain-of-Thought", type="primary"):
            if not cot_task or len(cot_task.strip()) < 10:
                st.warning("⚠️ Please enter a task with at least 10 characters.")
            else:
                with st.spinner("Processing with chain-of-thought..."):
                    try:
                        # Simple CoT prompt without importing PromptOptimizer
                        cot_prompt = f"""Let's solve this step by step:

Task: {cot_task}

Please think through this problem carefully:
1. First, identify what information we have
2. Then, break down what we need to find
3. Next, work through the calculations or reasoning step by step
4. Finally, provide the answer

Solution:"""
                        
                        import time
                        start_time = time.time()
                        
                        result = st.session_state.client.generate_with_metadata(cot_prompt)
                        
                        latency = time.time() - start_time
                        
                        st.success("✅ Chain-of-Thought Response!")
                        st.markdown("### 🧠 Step-by-Step Reasoning")
                        st.markdown(result['text'])
                        
                        st.markdown("---")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("📥 Prompt Tokens", result.get('prompt_tokens', 0))
                        with col2:
                            st.metric("📤 Response Tokens", result.get('response_tokens', 0))
                        with col3:
                            st.metric("⏱️ Latency", f"{latency:.2f}s")
                        
                    except Exception as e:
                        st.error(f"❌ Processing failed: {str(e)}")
                        st.info("💡 Try simplifying the task or checking your API connection.")
    
    with tab3:
        st.subheader("Risk Analysis")
        st.info("🔍 Analyze documents for potential risks, concerns, and red flags.")
        
        risk_document = st.text_area(
            "Document for risk analysis",
            height=150,
            placeholder="Example: This contract requires a non-refundable deposit of $50,000 due within 24 hours..."
        )
        
        if st.button("Analyze Risks", type="primary"):
            if not risk_document or len(risk_document.strip()) < 20:
                st.warning("⚠️ Please enter a document with at least 20 characters.")
            else:
                with st.spinner("Analyzing risks..."):
                    try:
                        import time
                        start_time = time.time()
                        
                        prompt = PromptTemplates.risk_analysis(risk_document)
                        result = st.session_state.client.generate_with_metadata(prompt)
                        
                        latency = time.time() - start_time
                        
                        st.success("✅ Risk Analysis Complete!")
                        st.markdown("### ⚠️ Risk Assessment")
                        st.markdown(result['text'])
                        
                        st.markdown("---")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("📥 Prompt Tokens", result.get('prompt_tokens', 0))
                        with col2:
                            st.metric("📤 Response Tokens", result.get('response_tokens', 0))
                        with col3:
                            st.metric("⏱️ Latency", f"{latency:.2f}s")
                        
                    except Exception as e:
                        st.error(f"❌ Analysis failed: {str(e)}")
                        st.info("💡 Try shortening the document or checking your connection.")


def analytics_page():
    """Analytics and metrics dashboard."""
    st.header("📈 Analytics & Performance Dashboard")
    st.markdown("Comprehensive system metrics and performance analysis.")
    
    if not st.session_state.processing_history:
        st.info("No operations performed yet. Use other features to generate analytics data.")
        
        # Demo mode
        st.markdown("---")
        st.subheader("📊 Demo Metrics")
        st.markdown("Start using the system to see real-time analytics here.")
        
        # Show sample structure
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Operations", 0, delta=None)
        with col2:
            st.metric("Total Tokens", 0)
        with col3:
            st.metric("Avg Latency", "0.00s")
        with col4:
            st.metric("Success Rate", "0%")
        
        return
    
    # Convert history to dataframe
    df = pd.DataFrame(st.session_state.processing_history)
    
    # Overview metrics (Power BI style cards)
    st.markdown("### 📊 Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_ops = len(df)
        st.metric("Total Operations", total_ops, delta=f"+{total_ops}")
    
    with col2:
        total_tokens = df['tokens'].sum()
        st.metric("Total Tokens", f"{total_tokens:,}", delta=f"{total_tokens}")
    
    with col3:
        if 'latency' in df.columns:
            avg_latency = df['latency'].mean()
            st.metric("Avg Latency", f"{avg_latency:.2f}s", delta=f"{avg_latency:.2f}s")
        else:
            st.metric("Avg Latency", "N/A")
    
    with col4:
        st.metric("Documents Indexed", st.session_state.documents_indexed)
    
    st.markdown("---")
    
    # Advanced visualizations (Power BI inspired)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Operations Distribution")
        op_counts = df['operation'].value_counts()
        fig = px.pie(
            values=op_counts.values,
            names=op_counts.index,
            title="Operation Types",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("⏱️ Response Time Analysis")
        if 'latency' in df.columns:
            fig = px.box(
                df,
                y='latency',
                x='operation',
                title="Latency Distribution by Operation",
                color='operation',
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Latency data not available")
    
    st.markdown("---")
    
    # Timeline analysis
    st.subheader("📈 Performance Over Time")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.line(
            df,
            x='timestamp',
            y='tokens',
            title="Token Usage Timeline",
            markers=True,
            color_discrete_sequence=['#1f77b4']
        )
        fig.update_layout(xaxis_title="Time", yaxis_title="Tokens")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'latency' in df.columns:
            fig = px.scatter(
                df,
                x='timestamp',
                y='latency',
                color='operation',
                title="Latency Scatter Plot",
                size='tokens',
                hover_data=['operation', 'tokens']
            )
            fig.update_layout(xaxis_title="Time", yaxis_title="Latency (s)")
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Token analysis by operation
    st.subheader("💎 Token Consumption Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        token_by_op = df.groupby('operation')['tokens'].sum().reset_index()
        fig = px.bar(
            token_by_op,
            x='operation',
            y='tokens',
            title="Total Tokens per Operation Type",
            color='tokens',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        token_stats = df.groupby('operation')['tokens'].agg(['mean', 'min', 'max']).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Average',
            x=token_stats['operation'],
            y=token_stats['mean'],
            marker_color='lightblue'
        ))
        fig.add_trace(go.Bar(
            name='Max',
            x=token_stats['operation'],
            y=token_stats['max'],
            marker_color='darkblue'
        ))
        fig.update_layout(
            title="Token Usage Statistics (Avg vs Max)",
            xaxis_title="Operation",
            yaxis_title="Tokens",
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Provider performance comparison
    st.subheader("🌐 Provider Performance Metrics")
    
    provider_data = []
    for provider, stats in st.session_state.model_performance.items():
        if stats['requests'] > 0:
            success_rate = (stats['success'] / stats['requests'] * 100)
            provider_data.append({
                'Provider': provider.capitalize(),
                'Total Requests': stats['requests'],
                'Successful': stats['success'],
                'Failed': stats['requests'] - stats['success'],
                'Success Rate (%)': success_rate,
                'Avg Latency (s)': stats['avg_latency']
            })
    
    if provider_data:
        df_providers = pd.DataFrame(provider_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(
                df_providers.style.background_gradient(subset=['Success Rate (%)'], cmap='RdYlGn'),
                use_container_width=True
            )
        
        with col2:
            fig = px.bar(
                df_providers,
                x='Provider',
                y=['Successful', 'Failed'],
                title="Request Status by Provider",
                barmode='stack',
                color_discrete_map={'Successful': 'green', 'Failed': 'red'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Recent operations table
    st.subheader("🕐 Recent Operations Log")
    recent_df = df.tail(20).copy()
    recent_df['timestamp'] = recent_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Reorder columns
    cols = ['timestamp', 'operation', 'tokens']
    if 'latency' in recent_df.columns:
        recent_df['latency'] = recent_df['latency'].round(2)
        cols.append('latency')
    
    st.dataframe(
        recent_df[cols].sort_values('timestamp', ascending=False),
        use_container_width=True,
        height=400
    )


def model_comparison_page():
    """Compare different LLM providers."""
    st.header("⚙️ Model Comparison")
    st.markdown("Compare performance across different LLM providers.")
    
    # Test prompt input
    st.subheader("Test Prompt")
    test_prompt = st.text_area(
        "Enter a test prompt to compare models",
        value="Explain what machine learning is in one sentence.",
        height=100
    )
    
    col1, col2 = st.columns(2)
    with col1:
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1, key="compare_temp")
    with col2:
        max_tokens = st.slider("Max Tokens", 100, 2000, 500, 100, key="compare_tokens")
    
    if st.button("🚀 Run Comparison", type="primary"):
        if test_prompt:
            st.markdown("---")
            st.subheader("Results")
            
            results = {}
            providers = ['gemini', 'groq', 'openrouter']
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, provider in enumerate(providers):
                status_text.text(f"Testing {provider.capitalize()}...")
                
                try:
                    import time
                    start_time = time.time()
                    
                    # Create client for this provider
                    test_client = UniversalLLMClient(
                        provider=provider,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        enable_fallback=False
                    )
                    
                    response = test_client.generate(test_prompt)
                    latency = time.time() - start_time
                    
                    results[provider] = {
                        'response': response,
                        'latency': latency,
                        'success': True,
                        'tokens': len(response.split())
                    }
                    
                except Exception as e:
                    results[provider] = {
                        'response': f"Error: {str(e)}",
                        'latency': 0,
                        'success': False,
                        'tokens': 0
                    }
                
                progress_bar.progress((idx + 1) / len(providers))
            
            status_text.text("Comparison complete!")
            
            # Display results
            st.markdown("---")
            
            # Metrics comparison
            col1, col2, col3 = st.columns(3)
            
            for idx, (provider, result) in enumerate(results.items()):
                col = [col1, col2, col3][idx]
                with col:
                    st.markdown(f"**{provider.capitalize()}**")
                    status = "✅" if result['success'] else "❌"
                    st.markdown(f"Status: {status}")
                    st.metric("Latency", f"{result['latency']:.2f}s")
                    st.metric("Response Length", f"{result['tokens']} tokens")
            
            # Detailed responses
            st.markdown("---")
            st.subheader("Detailed Responses")
            
            for provider, result in results.items():
                with st.expander(f"📝 {provider.capitalize()} Response"):
                    if result['success']:
                        st.success("Success")
                        st.markdown(result['response'])
                    else:
                        st.error("Failed")
                        st.code(result['response'])
            
            # Performance chart
            st.markdown("---")
            st.subheader("Performance Visualization")
            
            perf_data = []
            for provider, result in results.items():
                if result['success']:
                    perf_data.append({
                        'Provider': provider.capitalize(),
                        'Latency (s)': result['latency'],
                        'Response Length': result['tokens']
                    })
            
            if perf_data:
                df_perf = pd.DataFrame(perf_data)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.bar(
                        df_perf,
                        x='Provider',
                        y='Latency (s)',
                        title='Response Latency Comparison',
                        color='Provider'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.bar(
                        df_perf,
                        x='Provider',
                        y='Response Length',
                        title='Response Length Comparison',
                        color='Provider'
                    )
                    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
