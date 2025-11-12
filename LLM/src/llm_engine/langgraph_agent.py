"""LangGraph-based multi-agent system for complex document workflows."""

from typing import TypedDict, List, Dict, Any
try:
    from typing import Annotated
except ImportError:
    from typing_extensions import Annotated

try:
    from langgraph.graph import StateGraph, END
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    END = None
    BaseMessage = HumanMessage = AIMessage = None

from src.llm_engine.universal_client import UniversalLLMClient
from src.utils import logger


class AgentState(TypedDict):
    """State for the agent workflow."""
    messages: List[BaseMessage]
    document_text: str
    current_task: str
    extracted_data: Dict[str, Any]
    analysis_results: Dict[str, Any]
    next_action: str


class DocumentProcessingAgent:
    """
    LangGraph-based agent for complex document processing workflows.
    
    Demonstrates agentic workflows with:
    - Multi-step document analysis
    - Decision-making based on document type
    - Iterative refinement of extraction results
    - Error handling and retry mechanisms
    """
    
    def __init__(self, llm_provider: str = "auto"):
        """
        Initialize the document processing agent.
        
        Args:
            llm_provider: LLM provider to use
        """
        self.llm_client = UniversalLLMClient(provider=llm_provider)
        self.logger = logger
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(AgentState)
        
        # Add nodes for each step
        workflow.add_node("classify_document", self.classify_document)
        workflow.add_node("extract_structured_data", self.extract_structured_data)
        workflow.add_node("analyze_content", self.analyze_content)
        workflow.add_node("validate_results", self.validate_results)
        workflow.add_node("refine_extraction", self.refine_extraction)
        
        # Define edges (workflow paths)
        workflow.set_entry_point("classify_document")
        
        workflow.add_conditional_edges(
            "classify_document",
            self.route_by_document_type,
            {
                "contract": "extract_structured_data",
                "invoice": "extract_structured_data",
                "report": "analyze_content",
                "unknown": END
            }
        )
        
        workflow.add_edge("extract_structured_data", "validate_results")
        workflow.add_edge("analyze_content", "validate_results")
        
        workflow.add_conditional_edges(
            "validate_results",
            self.check_validation,
            {
                "valid": END,
                "needs_refinement": "refine_extraction",
                "failed": END
            }
        )
        
        workflow.add_edge("refine_extraction", "validate_results")
        
        return workflow.compile()
    
    def classify_document(self, state: AgentState) -> AgentState:
        """Classify the document type."""
        self.logger.info("Classifying document...")
        
        prompt = f"""Classify the following document into ONE of these categories:
- contract
- invoice
- report
- unknown

Document:
{state['document_text'][:1000]}

Classification (one word only):"""
        
        classification = self.llm_client.generate(prompt, temperature=0.1).strip().lower()
        
        state["current_task"] = "classification"
        state["analysis_results"]["document_type"] = classification
        state["messages"].append(AIMessage(content=f"Document classified as: {classification}"))
        
        return state
    
    def route_by_document_type(self, state: AgentState) -> str:
        """Route to appropriate processing based on document type."""
        doc_type = state["analysis_results"].get("document_type", "unknown")
        self.logger.info(f"Routing document type: {doc_type}")
        return doc_type
    
    def extract_structured_data(self, state: AgentState) -> AgentState:
        """Extract structured data from document."""
        self.logger.info("Extracting structured data...")
        
        doc_type = state["analysis_results"]["document_type"]
        
        if doc_type == "contract":
            fields = ["Parties", "Effective Date", "Termination Date", "Key Terms", "Payment Terms"]
        elif doc_type == "invoice":
            fields = ["Invoice Number", "Date", "Due Date", "Amount", "Items", "Tax"]
        else:
            fields = ["Key Information"]
        
        prompt = f"""Extract the following information from the document:

Fields to extract: {', '.join(fields)}

Document:
{state['document_text'][:2000]}

Provide the extracted information in JSON format."""
        
        extraction = self.llm_client.generate(prompt, temperature=0.2)
        
        state["extracted_data"] = {"raw_extraction": extraction, "fields": fields}
        state["current_task"] = "extraction"
        state["messages"].append(AIMessage(content=f"Extracted data for {len(fields)} fields"))
        
        return state
    
    def analyze_content(self, state: AgentState) -> AgentState:
        """Analyze document content for insights."""
        self.logger.info("Analyzing document content...")
        
        prompt = f"""Analyze the following report and provide:
1. Key findings (3-5 bullet points)
2. Main conclusions
3. Risk factors (if any)
4. Recommendations

Document:
{state['document_text'][:2000]}

Analysis:"""
        
        analysis = self.llm_client.generate(prompt, temperature=0.3)
        
        state["analysis_results"]["content_analysis"] = analysis
        state["current_task"] = "analysis"
        state["messages"].append(AIMessage(content="Content analysis completed"))
        
        return state
    
    def validate_results(self, state: AgentState) -> AgentState:
        """Validate extracted results for completeness and accuracy."""
        self.logger.info("Validating results...")
        
        # Simple validation logic
        if state["current_task"] == "extraction":
            expected_fields = len(state["extracted_data"].get("fields", []))
            # Check if extraction contains expected number of fields
            extraction_text = state["extracted_data"].get("raw_extraction", "")
            
            # Count how many fields appear in the extraction
            found_fields = sum(1 for field in state["extracted_data"].get("fields", []) 
                             if field.lower() in extraction_text.lower())
            
            completeness = found_fields / expected_fields if expected_fields > 0 else 0
            
            state["analysis_results"]["validation"] = {
                "complete": completeness >= 0.7,
                "completeness_score": completeness,
                "needs_refinement": completeness < 0.7
            }
        else:
            # For analysis tasks, assume valid
            state["analysis_results"]["validation"] = {
                "complete": True,
                "completeness_score": 1.0,
                "needs_refinement": False
            }
        
        state["messages"].append(AIMessage(
            content=f"Validation score: {state['analysis_results']['validation']['completeness_score']:.2f}"
        ))
        
        return state
    
    def check_validation(self, state: AgentState) -> str:
        """Check validation results and decide next step."""
        validation = state["analysis_results"].get("validation", {})
        
        if validation.get("complete", False):
            return "valid"
        elif validation.get("needs_refinement", False):
            return "needs_refinement"
        else:
            return "failed"
    
    def refine_extraction(self, state: AgentState) -> AgentState:
        """Refine extraction with more specific prompts."""
        self.logger.info("Refining extraction...")
        
        missing_fields = []
        extraction_text = state["extracted_data"].get("raw_extraction", "")
        
        for field in state["extracted_data"].get("fields", []):
            if field.lower() not in extraction_text.lower():
                missing_fields.append(field)
        
        if missing_fields:
            prompt = f"""Previous extraction was incomplete. Please extract specifically:

Missing fields: {', '.join(missing_fields)}

Document:
{state['document_text'][:2000]}

Provide ONLY the missing information in JSON format."""
            
            refinement = self.llm_client.generate(prompt, temperature=0.1)
            
            # Append refinement to original extraction
            state["extracted_data"]["raw_extraction"] += "\n\nRefinement:\n" + refinement
            state["messages"].append(AIMessage(content=f"Refined extraction for {len(missing_fields)} missing fields"))
        
        return state
    
    def process_document(self, document_text: str) -> Dict[str, Any]:
        """
        Process a document through the agent workflow.
        
        Args:
            document_text: Document text to process
            
        Returns:
            Dictionary with processing results
        """
        self.logger.info("Starting document processing workflow...")
        
        # Initialize state
        initial_state: AgentState = {
            "messages": [HumanMessage(content="Process this document")],
            "document_text": document_text,
            "current_task": "",
            "extracted_data": {},
            "analysis_results": {},
            "next_action": ""
        }
        
        # Run the workflow
        final_state = self.workflow.invoke(initial_state)
        
        # Prepare results
        results = {
            "document_type": final_state["analysis_results"].get("document_type", "unknown"),
            "extracted_data": final_state.get("extracted_data", {}),
            "analysis": final_state["analysis_results"].get("content_analysis", ""),
            "validation": final_state["analysis_results"].get("validation", {}),
            "workflow_messages": [msg.content for msg in final_state["messages"]],
            "status": "success" if final_state["analysis_results"].get("validation", {}).get("complete") else "incomplete"
        }
        
        self.logger.info(f"Workflow completed with status: {results['status']}")
        
        return results


class MultiAgentCollaboration:
    """
    Multi-agent system where specialized agents collaborate on complex tasks.
    
    Demonstrates:
    - Agent specialization (extraction, analysis, validation)
    - Inter-agent communication
    - Consensus building
    """
    
    def __init__(self, llm_provider: str = "auto"):
        """Initialize multi-agent system."""
        self.llm_client = UniversalLLMClient(provider=llm_provider)
        self.logger = logger
    
    def extraction_agent(self, document: str, fields: List[str]) -> str:
        """Specialized agent for data extraction."""
        prompt = f"""You are an extraction specialist. Extract: {', '.join(fields)}

Document:
{document[:1500]}

Extraction (JSON format):"""
        
        return self.llm_client.generate(prompt, temperature=0.1)
    
    def analysis_agent(self, document: str) -> str:
        """Specialized agent for document analysis."""
        prompt = f"""You are an analysis specialist. Analyze this document for:
- Key themes
- Sentiment
- Important entities
- Risk factors

Document:
{document[:1500]}

Analysis:"""
        
        return self.llm_client.generate(prompt, temperature=0.3)
    
    def validation_agent(self, extraction: str, analysis: str, document: str) -> Dict[str, Any]:
        """Specialized agent for validation."""
        prompt = f"""You are a validation specialist. Review the extraction and analysis:

Extraction:
{extraction[:500]}

Analysis:
{analysis[:500]}

Original Document:
{document[:1000]}

Validation (JSON with "accuracy", "completeness", "issues"):"""
        
        validation = self.llm_client.generate(prompt, temperature=0.2)
        
        return {"validation_report": validation}
    
    def collaborative_processing(self, document: str, fields: List[str]) -> Dict[str, Any]:
        """
        Process document with multiple specialized agents.
        
        Args:
            document: Document text
            fields: Fields to extract
            
        Returns:
            Combined results from all agents
        """
        self.logger.info("Starting multi-agent collaborative processing...")
        
        # Agent 1: Extraction
        extraction = self.extraction_agent(document, fields)
        
        # Agent 2: Analysis
        analysis = self.analysis_agent(document)
        
        # Agent 3: Validation
        validation = self.validation_agent(extraction, analysis, document)
        
        return {
            "extraction": extraction,
            "analysis": analysis,
            "validation": validation,
            "agents_used": ["extraction", "analysis", "validation"]
        }
