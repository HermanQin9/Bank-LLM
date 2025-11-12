"""Main entry point for document intelligence system."""

import argparse
from pathlib import Path
from typing import List
from src.document_parser import PDFParser, OCREngine
from src.llm_engine import GeminiClient, PromptTemplates
from src.rag_system import RAGPipeline
from src.utils import Config, logger, ensure_dir


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Document Intelligence System with Gemini"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory containing documents"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["parse", "classify", "extract", "rag"],
        default="parse",
        help="Processing mode"
    )
    
    parser.add_argument(
        "--query",
        type=str,
        help="Query for RAG mode"
    )
    
    return parser.parse_args()


def process_documents(input_dir: str, output_dir: str, mode: str, query: str = None):
    """
    Process documents based on mode.
    
    Args:
        input_dir: Input directory
        output_dir: Output directory
        mode: Processing mode
        query: Query for RAG mode
    """
    ensure_dir(output_dir)
    
    input_path = Path(input_dir)
    if not input_path.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return
    
    # Get all PDF files
    pdf_files = list(input_path.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files")
    
    if not pdf_files:
        logger.warning("No PDF files found")
        return
    
    if mode == "parse":
        # Parse documents
        parser = PDFParser()
        for pdf_file in pdf_files:
            logger.info(f"Parsing {pdf_file.name}")
            result = parser.parse(str(pdf_file))
            logger.info(f"Extracted {len(result['text'])} characters")
    
    elif mode == "classify":
        # Classify documents
        parser = PDFParser()
        client = GeminiClient()
        
        categories = ["Contract", "Invoice", "Report", "Form", "Letter", "Other"]
        
        for pdf_file in pdf_files:
            logger.info(f"Classifying {pdf_file.name}")
            result = parser.parse(str(pdf_file))
            
            if result['text']:
                prompt = PromptTemplates.document_classification(
                    result['text'][:2000],  # Use first 2000 chars
                    categories
                )
                classification = client.generate(prompt)
                logger.info(f"Classification: {classification}")
    
    elif mode == "extract":
        # Extract information
        parser = PDFParser()
        client = GeminiClient()
        
        fields = ["Date", "Amount", "Parties", "Purpose"]
        
        for pdf_file in pdf_files:
            logger.info(f"Extracting from {pdf_file.name}")
            result = parser.parse(str(pdf_file))
            
            if result['text']:
                prompt = PromptTemplates.information_extraction(
                    result['text'][:3000],
                    fields
                )
                extraction = client.generate(prompt)
                logger.info(f"Extraction:\n{extraction}")
    
    elif mode == "rag":
        # RAG system
        if not query:
            logger.error("Query required for RAG mode (use --query)")
            return
        
        logger.info("Initializing RAG pipeline")
        rag = RAGPipeline()
        
        # Index documents
        parser = PDFParser()
        documents = []
        metadatas = []
        
        for pdf_file in pdf_files:
            logger.info(f"Indexing {pdf_file.name}")
            result = parser.parse(str(pdf_file))
            if result['text']:
                documents.append(result['text'])
                metadatas.append({
                    'filename': pdf_file.name,
                    'source': str(pdf_file)
                })
        
        if documents:
            rag.index_documents(documents, metadatas)
            
            # Query
            logger.info(f"Querying: {query}")
            answer = rag.query(query)
            
            logger.info(f"\nAnswer: {answer['answer']}")
            logger.info(f"Sources: {answer['num_sources']}")


def main():
    """Main function."""
    args = parse_arguments()
    
    logger.info("Starting Document Intelligence System")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    
    process_documents(
        args.input,
        args.output,
        args.mode,
        args.query
    )
    
    logger.info("Processing complete")


if __name__ == "__main__":
    main()
