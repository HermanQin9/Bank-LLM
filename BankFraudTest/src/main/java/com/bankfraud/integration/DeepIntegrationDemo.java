package com.bankfraud.integration;

import com.bankfraud.model.Transaction;
import lombok.extern.slf4j.Slf4j;

import java.sql.Connection;
import java.sql.DriverManager;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * End-to-End Deep Integration Demo
 * 
 * This demonstrates the REAL integration between Java and Python/LLM:
 * 1. Java receives transaction
 * 2. Automatically triggers Python/LLM analysis
 * 3. Python writes enriched data to shared database
 * 4. Java reads enriched data for decision making
 * 
 * NO duplication, NO separate systems - ONE unified intelligence platform!
 * 
 * @author Banking Platform Team
 * @version 2.0
 */
@Slf4j
public class DeepIntegrationDemo {

    private static final String DB_URL = "jdbc:postgresql://localhost:5432/frauddb";
    private static final String DB_USER = "postgres";
    private static final String DB_PASSWORD = "postgres";
    private static final String PYTHON_SERVICE_URL = "http://localhost:8000";

    public static void main(String[] args) throws Exception {
        log.info("=".repeat(70));
        log.info("🚀 DEEP INTEGRATION DEMO - Java ↔ Python ↔ LLM");
        log.info("=".repeat(70));

        // Initialize connections
        Connection dbConnection = DriverManager.getConnection(DB_URL, DB_USER, DB_PASSWORD);
        PythonBridge pythonBridge = new PythonBridge(PYTHON_SERVICE_URL, dbConnection);
        ExecutorService executor = Executors.newFixedThreadPool(4);

        // Step 1: Check if Python service is healthy
        log.info("\n📡 Step 1: Checking Python/LLM Service Health...");
        boolean healthy = pythonBridge.isHealthy();
        if (healthy) {
            log.info("   ✅ Python service is healthy and ready");
        } else {
            log.warn("   ⚠️  Python service not available - demo will show database integration only");
        }

        // Step 2: Create a suspicious transaction in Java
        log.info("\n💳 Step 2: Java System Detects New Transaction...");
        Transaction suspiciousTransaction = createSuspiciousTransaction();
        log.info("   Transaction ID: {}", suspiciousTransaction.getTransactionId());
        log.info("   Customer: {}", suspiciousTransaction.getCustomerId());
        log.info("   Amount: ${:,.2f}", suspiciousTransaction.getAmount());
        log.info("   Merchant: {}", suspiciousTransaction.getMerchantName());

        // Step 3: Trigger Python analysis in real-time (async)
        log.info("\n🤖 Step 3: Automatically Triggering Python/LLM Analysis...");
        if (healthy) {
            pythonBridge.analyzeTransactionRealtime(suspiciousTransaction)
                    .thenAccept(result -> {
                        if (result != null) {
                            log.info("\n📊 Step 4: Python Analysis Results Received");
                            log.info("   🎯 Risk Score: {}", result.getRiskScore());
                            log.info("   ⚠️  Risk Level: {}", result.getRiskLevel());
                            log.info("   💭 LLM Reasoning: {}",
                                    result.getReasoning().substring(0,
                                            Math.min(100, result.getReasoning().length())) + "...");
                            log.info("   ⚡ Recommended Action: {}", result.getRecommendedAction());

                            // Step 5: Java makes decision based on Python's analysis
                            log.info("\n🎯 Step 5: Java System Makes Decision");
                            if (result.getRiskScore() > 0.7) {
                                log.info("   🚫 BLOCKING transaction - High risk detected by unified intelligence");
                            } else if (result.getRiskScore() > 0.5) {
                                log.info("   ⚠️  FLAGGING for manual review - Medium risk");
                            } else {
                                log.info("   ✅ APPROVING transaction - Low risk");
                            }
                        }
                    })
                    .exceptionally(ex -> {
                        log.error("   ❌ Analysis failed: {}", ex.getMessage());
                        return null;
                    });
        } else {
            log.info("   ⏭️  Skipping Python analysis - service not available");
        }

        // Step 6: Demonstrate bidirectional data flow
        log.info("\n📥 Step 6: Java Reads Python-Enriched Customer Profile...");
        Thread.sleep(2000); // Give Python time to process

        PythonBridge.EnrichedCustomerProfile profile = pythonBridge
                .getEnrichedProfile(suspiciousTransaction.getCustomerId());

        if (profile != null) {
            log.info("   ✅ Enriched profile loaded from shared database");
            log.info("   👤 Business Type: {}", profile.getBusinessType());
            log.info("   💰 Expected Monthly Volume: ${:,.2f}",
                    profile.getExpectedMonthlyVolume());
            log.info("   📈 Confidence Score: {}", profile.getConfidenceScore());
            log.info("   📄 KYC Source: {}", profile.getKycDocumentSource());

            // Step 7: Java uses Python's insights for rule-based decisions
            log.info("\n🧠 Step 7: Java Rules Engine Uses Python's ML Insights");
            if (profile.getConfidenceScore() < 0.5) {
                log.info("   ⚠️  Low confidence profile - Additional verification required");
            } else if (suspiciousTransaction.getAmount().doubleValue() > profile.getExpectedMonthlyVolume() * 2) {
                log.info("   🚨 Transaction exceeds 2x expected monthly volume!");
            } else {
                log.info("   ✅ Transaction within expected parameters");
            }
        } else {
            log.info("   ℹ️  No enriched profile available yet - will be created by Python");
        }

        // Summary
        log.info("\n" + "=".repeat(70));
        log.info("✅ DEEP INTEGRATION DEMONSTRATION COMPLETE");
        log.info("=".repeat(70));
        log.info("\n🎯 Key Integration Points Demonstrated:");
        log.info("   1. ✅ Java → Python: Real-time transaction analysis trigger");
        log.info("   2. ✅ Python → Java: ML/LLM insights flow back automatically");
        log.info("   3. ✅ Shared Database: Single source of truth (PostgreSQL)");
        log.info("   4. ✅ Unified Models: Same data structures across systems");
        log.info("   5. ✅ Async Processing: Non-blocking for real-time performance");
        log.info("\n💡 This is NOT two separate projects - it's ONE unified platform!");
        log.info("=".repeat(70));

        // Cleanup
        Thread.sleep(3000); // Wait for async operations
        executor.shutdown();
        dbConnection.close();
    }

    private static Transaction createSuspiciousTransaction() {
        Transaction txn = new Transaction();
        txn.setTransactionId("TXN-DEMO-" + System.currentTimeMillis());
        txn.setCustomerId("CUST_VIP_001");
        txn.setAmount(new java.math.BigDecimal("75000.00"));
        txn.setMerchantName("Overseas Wire Transfer");
        txn.setMerchantCategory("WIRE_TRANSFER");
        txn.setTransactionType("WIRE");
        txn.setTransactionDate(java.time.LocalDateTime.now());
        txn.setCurrency("USD");
        return txn;
    }
}
