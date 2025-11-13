package com.bankfraud;

import com.bankfraud.integration.LLMServiceClient;
import com.bankfraud.model.FraudAlert;
import com.bankfraud.model.Transaction;
import com.bankfraud.repository.TransactionRepository;
import com.bankfraud.service.EnhancedFraudDetectionService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

/**
 * Demo: Real Integration Between Java and Python Systems
 * 
 * Shows actual data flow:
 * 1. Java loads transaction → PostgreSQL
 * 2. Java fraud detector calls Python LLM API
 * 3. Python reads from PostgreSQL + searches documents
 * 4. Python LLM analyzes + returns risk score
 * 5. Java combines scores → generates alert
 * 6. Alert stored in PostgreSQL (Python can read it)
 */
public class IntegrationDemo {
    private static final Logger logger = LoggerFactory.getLogger(IntegrationDemo.class);
    
    public static void main(String[] args) {
        logger.info("=".repeat(80));
        logger.info("REAL INTEGRATION DEMO: Java <-> Python");
        logger.info("=".repeat(80));
        
        // Initialize services
        String pythonApiUrl = System.getenv("PYTHON_API_URL");
        if (pythonApiUrl == null) {
            pythonApiUrl = "http://localhost:8000";
        }
        
        logger.info("\n[STEP 1] Initializing services...");
        logger.info("  Java: Transaction processing + rule-based detection");
        logger.info("  Python API: {} (LLM + RAG)", pythonApiUrl);
        
        TransactionRepository transactionRepo = new TransactionRepository();
        EnhancedFraudDetectionService fraudService = 
            new EnhancedFraudDetectionService(transactionRepo, pythonApiUrl);
        LLMServiceClient llmClient = new LLMServiceClient(pythonApiUrl);
        
        // Check if Python service is available
        boolean pythonAvailable = llmClient.isServiceHealthy();
        logger.info("  Python service status: {}", pythonAvailable ? "ONLINE" : "OFFLINE");
        
        if (!pythonAvailable) {
            logger.warn("\n[WARNING] Python service not available!");
            logger.warn("To run full integration:");
            logger.warn("  Terminal 1: cd LLM && python app/integration_api.py");
            logger.warn("  Terminal 2: Run this Java program");
            logger.warn("\nContinuing with rule-based detection only...\n");
        }
        
        // Demo Scenario 1: Analyze suspicious transaction
        logger.info("\n" + "=".repeat(80));
        logger.info("[SCENARIO 1] Suspicious Transaction Analysis");
        logger.info("=".repeat(80));
        
        Transaction suspiciousTx = new Transaction();
        suspiciousTx.setTransactionId("TXN_2025_DEMO_001");
        suspiciousTx.setCustomerId("CUST_12345");
        suspiciousTx.setAmount(new BigDecimal("15000.00")); // Unusually high
        suspiciousTx.setMerchantName("Unknown Overseas Vendor");
        suspiciousTx.setTransactionDate(LocalDateTime.now());
        
        logger.info("\nTransaction Details:");
        logger.info("  ID: {}", suspiciousTx.getTransactionId());
        logger.info("  Customer: {}", suspiciousTx.getCustomerId());
        logger.info("  Amount: ${}", suspiciousTx.getAmount());
        logger.info("  Merchant: {}", suspiciousTx.getMerchantName());
        
        logger.info("\n[Java] Running enhanced fraud detection...");
        FraudAlert alert = fraudService.analyzeTransaction(suspiciousTx);
        
        logger.info("\n[RESULT] Fraud Alert Generated:");
        logger.info("  Fraud Score: {}", alert.getFraudScore());
        logger.info("  Risk Level: {}", alert.getRiskLevel());
        logger.info("  Alert Type: {}", alert.getAlertType());
        logger.info("  Description: {}", alert.getDescription());
        
        if (alert.getRulesTriggered() != null && !alert.getRulesTriggered().isEmpty()) {
            logger.info("  Rules Triggered: {} items", alert.getRulesTriggered().size());
            logger.info("  Rules: {}", alert.getRulesTriggered());
        }
        
        // Demo Scenario 2: Search customer documents
        if (pythonAvailable) {
            logger.info("\n" + "=".repeat(80));
            logger.info("[SCENARIO 2] Document Evidence Search (RAG)");
            logger.info("=".repeat(80));
            
            String searchQuery = "previous suspicious transactions for customer CUST_12345";
            logger.info("\n[Java] Requesting document search from Python...");
            logger.info("  Query: {}", searchQuery);
            
            List<Map<String, Object>> documents = fraudService.searchCustomerDocuments(
                "CUST_12345", 
                searchQuery
            );
            
            logger.info("\n[Python -> Java] Document search results:");
            if (documents.isEmpty()) {
                logger.info("  No documents found (RAG may not be initialized)");
            } else {
                for (int i = 0; i < documents.size(); i++) {
                    Map<String, Object> doc = documents.get(i);
                    logger.info("  [{}] Source: {}", i+1, doc.get("source"));
                    logger.info("      Relevance: {}", doc.get("relevance_score"));
                    String content = (String) doc.get("content");
                    logger.info("      Excerpt: {}...", 
                               content.substring(0, Math.min(100, content.length())));
                }
            }
        }
        
        // Demo Scenario 3: Generate compliance report
        if (pythonAvailable) {
            logger.info("\n" + "=".repeat(80));
            logger.info("[SCENARIO 3] Compliance Report Generation");
            logger.info("=".repeat(80));
            
            logger.info("\n[Java] Requesting SAR report generation from Python...");
            logger.info("  Customer: CUST_12345");
            logger.info("  Report Type: SAR (Suspicious Activity Report)");
            
            String report = fraudService.generateComplianceReport("CUST_12345", "SAR");
            
            logger.info("\n[Python -> Java] Generated report:");
            logger.info("-".repeat(80));
            logger.info(report);
            logger.info("-".repeat(80));
        }
        
        // Summary
        logger.info("\n" + "=".repeat(80));
        logger.info("INTEGRATION SUMMARY");
        logger.info("=".repeat(80));
        
        logger.info("\n[DATA FLOW]");
        logger.info("  1. Java ETL -> PostgreSQL (transactions table)");
        logger.info("  2. Java rules -> Identifies suspicious pattern");
        logger.info("  3. Java HTTP -> Python API (analyzeTransaction)");
        logger.info("  4. Python -> PostgreSQL (reads transactions + customer_profiles)");
        logger.info("  5. Python RAG -> Searches document vector store");
        logger.info("  6. Python LLM -> Analyzes transaction + document context");
        logger.info("  7. Python HTTP -> Java (returns risk score + evidence)");
        logger.info("  8. Java -> PostgreSQL (stores alert in transaction_alerts)");
        logger.info("  9. Python -> Can read alerts for compliance reports");
        
        logger.info("\n[INTEGRATION POINTS]");
        logger.info("  {} Java LLMServiceClient: HTTP client calling Python API", 
                   pythonAvailable ? "[ACTIVE]" : "[INACTIVE]");
        logger.info("  {} Python integration_api.py: FastAPI endpoints", 
                   pythonAvailable ? "[ACTIVE]" : "[INACTIVE]");
        logger.info("  [SHARED] PostgreSQL database (both systems read/write)");
        logger.info("  [SHARED] transaction_alerts table (Java writes, Python reads)");
        logger.info("  [SHARED] customer_profiles table (Python writes, Java reads)");
        
        logger.info("\n[TECHNOLOGIES WORKING TOGETHER]");
        logger.info("  - Java 21: Transaction ETL, rule-based detection");
        logger.info("  - Scala 2.13: Statistical fraud analysis");
        logger.info("  - Python 3.11: LLM integration, RAG document search");
        logger.info("  - PostgreSQL 15: Shared data layer");
        logger.info("  - FastAPI: REST API for Java <-> Python communication");
        logger.info("  - Gemini/Groq LLMs: Transaction reasoning");
        logger.info("  - ChromaDB: Vector store for document embeddings");
        
        logger.info("\n" + "=".repeat(80));
        logger.info("This is REAL integration - not just architecture diagrams!");
        logger.info("=".repeat(80));
    }
}
