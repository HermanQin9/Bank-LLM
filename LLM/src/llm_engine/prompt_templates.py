"""Prompt templates for document understanding tasks."""

from typing import Dict, List, Optional


class PromptTemplates:
    """Collection of optimized prompts for various document tasks."""
    
    @staticmethod
    def document_classification(document_text: str, categories: List[str]) -> str:
        """
        Prompt for document classification.
        
        Args:
            document_text: Text content of the document
            categories: List of possible categories
            
        Returns:
            Formatted prompt
        """
        categories_str = ", ".join(categories)
        
        prompt = f"""Classify the following document into one of these categories: {categories_str}

Document:
{document_text}

Instructions:
- Read the document carefully and identify key characteristics
- Select the SINGLE most appropriate category from the list
- Output ONLY the category name, nothing else
- Do not provide explanations or additional text

Category:"""
        
        return prompt
    
    @staticmethod
    def information_extraction(document_text: str, fields: List[str]) -> str:
        """
        Prompt for structured information extraction.
        
        Args:
            document_text: Text content of the document
            fields: List of fields to extract
            
        Returns:
            Formatted prompt
        """
        fields_str = "\n".join([f"- {field}" for field in fields])
        
        prompt = f"""Extract the following information from the document. For each field, provide the extracted value or "Not found" if the information is not present in the document.

Fields to extract:
{fields_str}

Document:
{document_text}

Instructions:
- Extract ONLY the requested information
- If a field is not present in the document, write "Not found"
- Be precise and concise - extract the exact values, not descriptions
- Output ONLY the field name and value in this format: "FieldName: value"
- Each field should be on a new line

Output format example:
Date: 2024-01-15
Amount: $5,000.00
Parties: Not found

Extracted Information:
"""
        
        return prompt
    
    @staticmethod
    def document_summary(document_text: str, max_sentences: int = 3) -> str:
        """
        Prompt for document summarization.
        
        Args:
            document_text: Text content of the document
            max_sentences: Maximum number of sentences in summary
            
        Returns:
            Formatted prompt
        """
        prompt = f"""Provide a concise summary of the following document in {max_sentences} sentences or less.

Document:
{document_text}

Requirements:
- Capture the main points
- Be clear and concise
- Focus on key information
- Use professional language

Summary:"""
        
        return prompt
    
    @staticmethod
    def entity_extraction(document_text: str, entity_types: Optional[List[str]] = None) -> str:
        """
        Prompt for named entity extraction.
        
        Args:
            document_text: Text content of the document
            entity_types: List of entity types to extract (optional)
            
        Returns:
            Formatted prompt
        """
        if entity_types:
            types_str = ", ".join(entity_types)
            type_instruction = f"Focus on these entity types: {types_str}"
        else:
            type_instruction = "Extract all important entities including names, organizations, locations, dates, and amounts."
        
        prompt = f"""Extract named entities from the following document.

{type_instruction}

Document:
{document_text}

Instructions:
1. Identify all relevant entities
2. Categorize each entity by type
3. List entities in a structured format
4. Include context if helpful

Entities:
"""
        
        return prompt
    
    @staticmethod
    def question_answering(context: str, question: str) -> str:
        """
        Prompt for question answering based on document context.
        
        Args:
            context: Document context
            question: Question to answer
            
        Returns:
            Formatted prompt
        """
        prompt = f"""Answer the following question based on the provided context.

Context:
{context}

Question: {question}

Instructions:
- Base your answer strictly on the provided context
- If the answer is not in the context, say "I cannot answer based on the provided context"
- Be specific and cite relevant parts of the context
- Keep the answer concise

Answer:"""
        
        return prompt
    
    @staticmethod
    def risk_analysis(document_text: str) -> str:
        """
        Prompt for risk analysis in documents.
        
        Args:
            document_text: Text content of the document
            
        Returns:
            Formatted prompt
        """
        prompt = f"""Analyze the following document for potential risks, concerns, or red flags.

Document:
{document_text}

Analysis Requirements:
1. Identify any risks, concerns, or problematic clauses
2. Rate the severity of each risk (Low, Medium, High)
3. Provide brief explanations
4. Suggest mitigations if applicable

Risk Analysis:
"""
        
        return prompt
    
    @staticmethod
    def key_terms_extraction(document_text: str, max_terms: int = 10) -> str:
        """
        Prompt for extracting key terms and phrases.
        
        Args:
            document_text: Text content of the document
            max_terms: Maximum number of terms to extract
            
        Returns:
            Formatted prompt
        """
        prompt = f"""Extract the {max_terms} most important key terms or phrases from the document.

Document:
{document_text}

Instructions:
1. Identify the most significant terms
2. Focus on domain-specific terminology
3. Include both single words and phrases
4. Order by importance

Key Terms:
"""
        
        return prompt
    
    @staticmethod
    def custom_extraction(document_text: str, instruction: str) -> str:
        """
        Generic prompt for custom extraction tasks.
        
        Args:
            document_text: Text content of the document
            instruction: Custom extraction instruction
            
        Returns:
            Formatted prompt
        """
        prompt = f"""Task: {instruction}

Document:
{document_text}

Please complete the task following the given instruction.

Result:
"""
        
        return prompt
