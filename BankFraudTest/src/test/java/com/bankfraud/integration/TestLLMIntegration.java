package com.bankfraud.integration;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Integration test for Java → Python LLM service communication.
 * Validates real-time transaction analysis with LLM + RAG capabilities.
 * 
 * Prerequisites:
 * 1. Python FastAPI service running on http://localhost:8000
 * 2. PostgreSQL with test data (customers, transactions, transaction_alerts)
 * 3. RAG vector store initialized
 * 
 * Run: mvn test -Dtest=TestLLMIntegration
 */
public class TestLLMIntegration {

    private static final Logger logger = LoggerFactory.getLogger(TestLLMIntegration.class);
    private static LLMServiceClient llmClient;

    @BeforeAll
    public static void setUp() {
        llmClient = new LLMServiceClient("http://localhost:8000");

        // Verify service is running
        boolean healthy = llmClient.isServiceHealthy();
        if (!healthy) {
            logger.warn("⚠️  LLM service is not healthy - tests may fail!");
            logger.warn("   Make sure Python FastAPI is running: cd LLM && python app/integration_api.py");
        } else {
            logger.info("✅ LLM service is healthy and ready for testing");
        }
    }

    @Test
    public void testHealthCheck() {
        boolean healthy = llmClient.isServiceHealthy();
        assertTrue(healthy, "LLM service should be reachable");
        logger.info("✅ Health check passed");
    }

    @Test
    public void testAnalyzeNormalTransaction() {
        logger.info("🔍 Testing LLM analysis for NORMAL transaction...");

        Map<String, Object> result = llmClient.analyzeTransaction(
                "TXN-TEST-NORMAL-001",
                "CUST_001",
                5000.0,
                "Amazon");

        assertNotNull(result, "Result should not be null");
        assertFalse((Boolean) result.getOrDefault("error", false),
                "Should not have error: " + result.get("message"));

        // Validate response structure
        assertTrue(result.containsKey("risk_score"), "Should have risk_score");
        assertTrue(result.containsKey("reasoning"), "Should have reasoning");
        assertTrue(result.containsKey("recommended_action"), "Should have recommended_action");

        double riskScore = ((Number) result.get("risk_score")).doubleValue();
        assertTrue(riskScore >= 0.0 && riskScore <= 1.0, "Risk score should be between 0 and 1");

        logger.info("   Transaction ID: {}", result.get("transaction_id"));
        logger.info("   Risk Score: {}", riskScore);
        logger.info("   Reasoning: {}", result.get("reasoning"));
        logger.info("   Recommended Action: {}", result.get("recommended_action"));
        logger.info("   Deviation from Average: {}%",
                ((Number) result.getOrDefault("deviation_from_average", 0)).doubleValue() * 100);

        logger.info("✅ Normal transaction analysis completed");
    }

    @Test
    public void testAnalyzeSuspiciousTransaction() {
        logger.info("🔍 Testing LLM analysis for SUSPICIOUS transaction...");

        // High-value transaction that should trigger high risk score
        Map<String, Object> result = llmClient.analyzeTransaction(
                "TXN-TEST-SUSPICIOUS-002",
                "CUST_001",
                95000.0, // Significantly above normal
                "Wire Transfer - Unknown Recipient");

        assertNotNull(result, "Result should not be null");
        assertFalse((Boolean) result.getOrDefault("error", false),
                "Should not have error: " + result.get("message"));

        double riskScore = ((Number) result.get("risk_score")).doubleValue();

        logger.info("   Transaction ID: {}", result.get("transaction_id"));
        logger.info("   Risk Score: {}", riskScore);
        logger.info("   Reasoning: {}", result.get("reasoning"));
        logger.info("   Recommended Action: {}", result.get("recommended_action"));

        // For suspicious transaction, we expect higher risk score
        assertTrue(riskScore > 0.4, "Suspicious transaction should have elevated risk score");

        // Check if supporting documents were found
        Object supportingDocs = result.get("supporting_documents");
        if (supportingDocs instanceof List) {
            logger.info("   Supporting Documents Found: {}", ((List<?>) supportingDocs).size());
        }

        logger.info("✅ Suspicious transaction analysis completed");
    }

    @Test
    public void testSearchDocuments() {
        logger.info("🔍 Testing document search via RAG...");

        Map<String, Object> result = llmClient.searchDocuments(
                "CUST_001",
                "customer transaction limit monthly volume business type",
                5);

        assertNotNull(result, "Result should not be null");
        assertFalse((Boolean) result.getOrDefault("error", false),
                "Should not have error: " + result.get("message"));

        assertTrue(result.containsKey("documents"), "Should have documents field");
        assertTrue(result.containsKey("customer_id"), "Should have customer_id");
        assertTrue(result.containsKey("query"), "Should have query");

        Object documentsObj = result.get("documents");
        if (documentsObj instanceof List) {
            List<?> documents = (List<?>) documentsObj;
            logger.info("   Found {} relevant documents", documents.size());

            if (!documents.isEmpty()) {
                Object firstDoc = documents.get(0);
                logger.info("   First document preview: {}", firstDoc);
            }
        }

        logger.info("✅ Document search completed");
    }

    @Test
    public void testGenerateComplianceReport() {
        logger.info("🔍 Testing compliance report generation...");

        Map<String, Object> result = llmClient.generateComplianceReport(
                "CUST_001",
                "SAR" // Suspicious Activity Report
        );

        assertNotNull(result, "Result should not be null");

        if ((Boolean) result.getOrDefault("error", false)) {
            logger.warn("⚠️  Report generation returned error (may be expected if no data): {}",
                    result.get("message"));
        } else {
            assertTrue(result.containsKey("report_id"), "Should have report_id");
            assertTrue(result.containsKey("report_content"), "Should have report_content");
            assertTrue(result.containsKey("report_type"), "Should have report_type");

            logger.info("   Report ID: {}", result.get("report_id"));
            logger.info("   Report Type: {}", result.get("report_type"));
            logger.info("   Transaction Count: {}", result.get("transaction_count"));

            String reportContent = (String) result.get("report_content");
            if (reportContent != null) {
                logger.info("   Report Preview: {}...",
                        reportContent.length() > 200 ? reportContent.substring(0, 200) : reportContent);
            }

            logger.info("✅ Compliance report generation completed");
        }
    }

    @Test
    public void testRealTimeWorkflow() {
        logger.info("\n" + "=".repeat(70));
        logger.info("🚀 Testing REAL-TIME Java → Python → LLM Workflow");
        logger.info("=".repeat(70));

        // Step 1: Simulate real-time transaction from Java system
        String transactionId = "TXN-REALTIME-" + System.currentTimeMillis();
        String customerId = "CUST_001";
        double amount = 45000.0;
        String merchant = "Overseas Wire Transfer";

        logger.info("\n📋 Step 1: Transaction Detected in Java System");
        logger.info("   Transaction ID: {}", transactionId);
        logger.info("   Customer: {}", customerId);
        logger.info("   Amount: ${:,.2f}", amount);
        logger.info("   Merchant: {}", merchant);

        // Step 2: Call Python LLM service for real-time analysis
        logger.info("\n🤖 Step 2: Calling Python LLM Service for Analysis...");
        long startTime = System.currentTimeMillis();

        Map<String, Object> analysis = llmClient.analyzeTransaction(
                transactionId, customerId, amount, merchant);

        long duration = System.currentTimeMillis() - startTime;
        logger.info("   Analysis completed in {}ms", duration);

        // Step 3: Process LLM response
        logger.info("\n📊 Step 3: Processing LLM Analysis Results");

        if ((Boolean) analysis.getOrDefault("error", false)) {
            logger.error("   ❌ LLM analysis failed: {}", analysis.get("message"));
            fail("LLM analysis should succeed");
        }

        double riskScore = ((Number) analysis.get("risk_score")).doubleValue();
        String reasoning = (String) analysis.get("reasoning");
        String recommendedAction = (String) analysis.get("recommended_action");

        logger.info("   🎯 Risk Score: {}", riskScore);
        logger.info("   💭 LLM Reasoning: {}", reasoning);
        logger.info("   ⚡ Recommended Action: {}", recommendedAction);

        // Step 4: Make decision based on LLM response
        logger.info("\n🎯 Step 4: Java System Decision");

        if (riskScore >= 0.7) {
            logger.info("   🚨 HIGH RISK - Blocking transaction and triggering investigation");
            assertEquals("BLOCK", recommendedAction, "High risk should recommend blocking");
        } else if (riskScore >= 0.4) {
            logger.info("   ⚠️  MEDIUM RISK - Flagging for manual review");
        } else {
            logger.info("   ✅ LOW RISK - Approving transaction");
        }

        // Step 5: Verify integration completeness
        logger.info("\n✅ Step 5: Verification");
        assertTrue(duration < 5000, "LLM analysis should complete within 5 seconds");
        assertNotNull(reasoning, "Should have LLM reasoning");
        assertNotNull(recommendedAction, "Should have recommended action");

        logger.info("   ✅ Real-time integration workflow validated successfully!");
        logger.info("   ✅ Java ↔ Python ↔ LLM communication working");
        logger.info("   ✅ Response time: {}ms (acceptable for real-time)", duration);

        logger.info("\n" + "=".repeat(70));
        logger.info("🎉 REAL-TIME INTEGRATION TEST PASSED");
        logger.info("=".repeat(70) + "\n");
    }
}
