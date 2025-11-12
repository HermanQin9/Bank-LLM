"""PDF document parser."""

from typing import Optional, Dict, List
import PyPDF2
import pdfplumber
from pathlib import Path
from src.utils import logger, clean_text


class PDFParser:
    """Parser for PDF documents."""
    
    def __init__(self):
        """Initialize PDF parser."""
        self.logger = logger
        
    def extract_text_pypdf2(self, filepath: str) -> str:
        """Extract text using PyPDF2."""
        try:
            with open(filepath, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return clean_text(text)
        except Exception as e:
            self.logger.error(f"PyPDF2 extraction failed: {e}")
            return ""
    
    def extract_text_pdfplumber(self, filepath: str) -> str:
        """Extract text using pdfplumber (better for complex layouts)."""
        try:
            with pdfplumber.open(filepath) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                return clean_text(text)
        except Exception as e:
            self.logger.error(f"pdfplumber extraction failed: {e}")
            return ""
    
    def extract_tables(self, filepath: str) -> List[List[List[str]]]:
        """Extract tables from PDF."""
        try:
            tables = []
            with pdfplumber.open(filepath) as pdf:
                for page in pdf.pages:
                    page_tables = page.extract_tables()
                    if page_tables:
                        tables.extend(page_tables)
            return tables
        except Exception as e:
            self.logger.error(f"Table extraction failed: {e}")
            return []
    
    def get_metadata(self, filepath: str) -> Dict[str, any]:
        """Extract PDF metadata."""
        try:
            with open(filepath, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                metadata = pdf_reader.metadata
                return {
                    'title': metadata.get('/Title', ''),
                    'author': metadata.get('/Author', ''),
                    'subject': metadata.get('/Subject', ''),
                    'creator': metadata.get('/Creator', ''),
                    'num_pages': len(pdf_reader.pages)
                }
        except Exception as e:
            self.logger.error(f"Metadata extraction failed: {e}")
            return {}
    
    def parse(self, filepath: str, method: str = "pdfplumber") -> Dict[str, any]:
        """
        Parse PDF document and extract all information.
        
        Args:
            filepath: Path to PDF file
            method: Extraction method ('pdfplumber' or 'pypdf2')
            
        Returns:
            Dictionary containing text, tables, and metadata
        """
        self.logger.info(f"Parsing PDF: {filepath}")
        
        if not Path(filepath).exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        result = {
            'filepath': filepath,
            'filename': Path(filepath).name,
            'text': '',
            'tables': [],
            'metadata': {}
        }
        
        # Extract text
        if method == "pdfplumber":
            result['text'] = self.extract_text_pdfplumber(filepath)
        else:
            result['text'] = self.extract_text_pypdf2(filepath)
        
        # Extract tables
        result['tables'] = self.extract_tables(filepath)
        
        # Extract metadata
        result['metadata'] = self.get_metadata(filepath)
        
        self.logger.info(f"Extracted {len(result['text'])} characters, {len(result['tables'])} tables")
        
        return result
