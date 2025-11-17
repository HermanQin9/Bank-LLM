package com.bankfraud.integration;

import com.bankfraud.model.Transaction;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import lombok.extern.slf4j.Slf4j;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.time.Duration;
import java.util.concurrent.CompletableFuture;

/**
 * Real-time Bridge between Java and Python/LLM systems
 * 
 * This is NOT just an HTTP client - it's a bidirectional integration bridge
 * that:
 * 1. Pushes Java data to Python for ML/LLM processing
 * 2. Reads Python-generated insights back into Java
 * 3. Uses shared PostgreSQL as the integration layer
 * 
 * Design Philosophy:
 * - Database as Single Source of Truth
 * - Async/non-blocking for real-time performance
 * - Unified data models (matches Python's unified-intelligence)
 * 
 * @author Banking Platform Team
 * @version 2.0 - Deep Integration Edition
 */
@Slf4j
public class PythonBridge {

    private final HttpClient httpClient;
    private final ObjectMapper objectMapper;
    private final String pythonServiceUrl;
    private final Connection dbConnection;

    public PythonBridge(String pythonServiceUrl, Connection dbConnection) {
        this.pythonServiceUrl = pythonServiceUrl;
        this.dbConnection = dbConnection;
        this.httpClient = HttpClient.newBuilder()
                .version(HttpClient.Version.HTTP_2)
                .connectTimeout(Duration.ofSeconds(10))
                .build();
        this.objectMapper = new ObjectMapper();
    }

    /**
     * Real-time transaction analysis hook
     * Called automatically when a transaction is created in Java
     * 
     * Flow:
     * 1. Java creates transaction → saves to DB
     * 2. This method triggers Python analysis
     * 3. Python reads from DB, runs ML+LLM
     * 4. Python writes results back to DB
     * 5. Java can query enriched results
     */
    public CompletableFuture<AnalysisResult> analyzeTransactionRealtime(Transaction transaction) {
        log.info("🔗 Triggering real-time analysis for transaction {}", transaction.getTransactionId());

        return CompletableFuture.supplyAsync(() -> {
            try {
                // Prepare request payload
                ObjectNode payload = objectMapper.createObjectNode();
                payload.put("transaction_id", transaction.getTransactionId());
                payload.put("customer_id", transaction.getCustomerId());
                payload.put("amount", transaction.getAmount().doubleValue());
                payload.put("merchant_name", transaction.getMerchantName());

                // Call Python unified intelligence engine
                HttpRequest request = HttpRequest.newBuilder()
                        .uri(URI.create(pythonServiceUrl + "/analyze/transaction"))
                        .header("Content-Type", "application/json")
                        .POST(HttpRequest.BodyPublishers.ofString(payload.toString()))
                        .timeout(Duration.ofSeconds(30))
                        .build();

                HttpResponse<String> response = httpClient.send(request,
                        HttpResponse.BodyHandlers.ofString());

                if (response.statusCode() == 200) {
                    // Parse Python's response
                    AnalysisResult result = objectMapper.readValue(
                            response.body(), AnalysisResult.class);

                    log.info("✅ Analysis completed: risk_score={}, action={}",
                            result.getRiskScore(), result.getRecommendedAction());

                    // Optionally: Write result to shared database
                    savePythonAnalysisToDb(transaction.getTransactionId(), result);

                    return result;
                } else {
                    log.error("❌ Python service returned error: {}", response.statusCode());
                    return null;
                }

            } catch (Exception e) {
                log.error("❌ Real-time analysis failed", e);
                return null;
            }
        });
    }

    /**
     * Read Python-enriched customer profile from shared database
     * 
     * This demonstrates bidirectional data flow:
     * - Python writes enriched profiles (ML clusters, LLM insights)
     * - Java reads them for rule-based decision making
     */
    public EnrichedCustomerProfile getEnrichedProfile(String customerId) {
        log.info("📊 Reading Python-enriched profile for customer {}", customerId);

        try (PreparedStatement stmt = dbConnection.prepareStatement(
                "SELECT customer_id, business_type, expected_monthly_volume, " +
                        "       expected_max_amount, confidence_score, kyc_document_source " +
                        "FROM customer_profiles WHERE customer_id = ?")) {

            stmt.setString(1, customerId);
            ResultSet rs = stmt.executeQuery();

            if (rs.next()) {
                EnrichedCustomerProfile profile = new EnrichedCustomerProfile();
                profile.setCustomerId(rs.getString("customer_id"));
                profile.setBusinessType(rs.getString("business_type"));
                profile.setExpectedMonthlyVolume(rs.getDouble("expected_monthly_volume"));
                profile.setExpectedMaxAmount(rs.getDouble("expected_max_amount"));
                profile.setConfidenceScore(rs.getDouble("confidence_score"));
                profile.setKycDocumentSource(rs.getString("kyc_document_source"));

                log.info("✅ Loaded enriched profile: type={}, confidence={}",
                        profile.getBusinessType(), profile.getConfidenceScore());

                return profile;
            } else {
                log.warn("⚠️  No enriched profile found for customer {}", customerId);
                return null;
            }

        } catch (Exception e) {
            log.error("❌ Failed to read enriched profile", e);
            return null;
        }
    }

    /**
     * Check if Python/LLM service is healthy and responsive
     */
    public boolean isHealthy() {
        try {
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(pythonServiceUrl + "/health"))
                    .GET()
                    .timeout(Duration.ofSeconds(5))
                    .build();

            HttpResponse<String> response = httpClient.send(request,
                    HttpResponse.BodyHandlers.ofString());

            return response.statusCode() == 200;
        } catch (Exception e) {
            log.error("❌ Python service health check failed", e);
            return false;
        }
    }

    /**
     * Save Python's analysis result to shared database
     * This enables Java to query analysis results without calling Python again
     */
    private void savePythonAnalysisToDb(String transactionId, AnalysisResult result) {
        try (PreparedStatement stmt = dbConnection.prepareStatement(
                "INSERT INTO fraud_alerts (transaction_id, customer_id, alert_type, " +
                        "fraud_score, risk_level, rules_triggered, description, status, created_at) " +
                        "VALUES (?, ?, ?, ?, ?, ?::text[], ?, ?, NOW()) " +
                        "ON CONFLICT (transaction_id) DO UPDATE SET " +
                        "fraud_score = EXCLUDED.fraud_score, description = EXCLUDED.description")) {

            stmt.setString(1, transactionId);
            stmt.setString(2, result.getCustomerId());
            stmt.setString(3, "UNIFIED_INTELLIGENCE");
            stmt.setDouble(4, result.getRiskScore());
            stmt.setString(5, result.getRiskLevel());

            // Convert rules to PostgreSQL array format
            String[] rules = result.getKeyRiskFactors() != null ? result.getKeyRiskFactors().toArray(new String[0])
                    : new String[0];
            java.sql.Array sqlArray = dbConnection.createArrayOf("text", rules);
            stmt.setArray(6, sqlArray);

            stmt.setString(7, result.getReasoning());
            stmt.setString(8, "PENDING");

            stmt.executeUpdate();

            log.info("💾 Saved Python analysis to shared database");

        } catch (Exception e) {
            log.error("❌ Failed to save Python analysis to DB", e);
        }
    }

    /**
     * Data class for Python's analysis result
     * Matches the unified_intelligence.shared_models.FraudAlert structure
     */
    public static class AnalysisResult {
        private String customerId;
        private double riskScore;
        private String riskLevel;
        private String reasoning;
        private String recommendedAction;
        private java.util.List<String> keyRiskFactors;

        // Getters and setters
        public String getCustomerId() {
            return customerId;
        }

        public void setCustomerId(String customerId) {
            this.customerId = customerId;
        }

        public double getRiskScore() {
            return riskScore;
        }

        public void setRiskScore(double riskScore) {
            this.riskScore = riskScore;
        }

        public String getRiskLevel() {
            return riskLevel;
        }

        public void setRiskLevel(String riskLevel) {
            this.riskLevel = riskLevel;
        }

        public String getReasoning() {
            return reasoning;
        }

        public void setReasoning(String reasoning) {
            this.reasoning = reasoning;
        }

        public String getRecommendedAction() {
            return recommendedAction;
        }

        public void setRecommendedAction(String action) {
            this.recommendedAction = action;
        }

        public java.util.List<String> getKeyRiskFactors() {
            return keyRiskFactors;
        }

        public void setKeyRiskFactors(java.util.List<String> factors) {
            this.keyRiskFactors = factors;
        }
    }

    /**
     * Enriched customer profile from Python ML/LLM processing
     */
    public static class EnrichedCustomerProfile {
        private String customerId;
        private String businessType;
        private double expectedMonthlyVolume;
        private double expectedMaxAmount;
        private double confidenceScore;
        private String kycDocumentSource;

        // Getters and setters
        public String getCustomerId() {
            return customerId;
        }

        public void setCustomerId(String id) {
            this.customerId = id;
        }

        public String getBusinessType() {
            return businessType;
        }

        public void setBusinessType(String type) {
            this.businessType = type;
        }

        public double getExpectedMonthlyVolume() {
            return expectedMonthlyVolume;
        }

        public void setExpectedMonthlyVolume(double volume) {
            this.expectedMonthlyVolume = volume;
        }

        public double getExpectedMaxAmount() {
            return expectedMaxAmount;
        }

        public void setExpectedMaxAmount(double amount) {
            this.expectedMaxAmount = amount;
        }

        public double getConfidenceScore() {
            return confidenceScore;
        }

        public void setConfidenceScore(double score) {
            this.confidenceScore = score;
        }

        public String getKycDocumentSource() {
            return kycDocumentSource;
        }

        public void setKycDocumentSource(String source) {
            this.kycDocumentSource = source;
        }
    }
}
