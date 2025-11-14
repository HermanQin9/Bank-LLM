package com.bankfraud.integration;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.HashMap;
import java.util.Map;

/**
 * Client for integrating with Python LLM services
 * Enables Java transaction processing to leverage document intelligence
 * 
 * Integration Points:
 * 1. Analyze suspicious transaction patterns with LLM reasoning
 * 2. Search documents for transaction evidence via RAG
 * 3. Generate compliance reports combining DB + document analysis
 */
public class LLMServiceClient {
    private static final Logger logger = LoggerFactory.getLogger(LLMServiceClient.class);

    private final HttpClient httpClient;
    private final String pythonApiUrl;
    private final ObjectMapper objectMapper;

    public LLMServiceClient(String pythonApiUrl) {
        this.pythonApiUrl = pythonApiUrl != null ? pythonApiUrl : "http://localhost:8000";
        this.httpClient = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(10))
                .version(HttpClient.Version.HTTP_1_1)
                .build();
        this.objectMapper = new ObjectMapper();
        // Don't set naming strategy - we're using Map with snake_case keys directly
    }

    /**
     * Analyze transaction using LLM + document context
     * 
     * @param transactionId Transaction to analyze
     * @param customerId    Customer ID
     * @param amount        Transaction amount
     * @param merchantName  Merchant name
     * @return Analysis result with risk score and evidence
     */
    public Map<String, Object> analyzeTransaction(
            String transactionId,
            String customerId,
            double amount,
            String merchantName) {

        try {
            Map<String, Object> requestBody = new HashMap<>();
            requestBody.put("transaction_id", transactionId);
            requestBody.put("customer_id", customerId);
            requestBody.put("amount", amount);
            requestBody.put("merchant_name", merchantName);

            String jsonBody = objectMapper.writeValueAsString(requestBody);

            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(pythonApiUrl + "/api/analyze-transaction"))
                    .header("Content-Type", "application/json; charset=UTF-8")
                    .timeout(Duration.ofSeconds(30))
                    .POST(HttpRequest.BodyPublishers.ofString(jsonBody, java.nio.charset.StandardCharsets.UTF_8))
                    .build();

            HttpResponse<String> response = httpClient.send(
                    request,
                    HttpResponse.BodyHandlers.ofString());

            if (response.statusCode() == 200) {
                Map<String, Object> result = objectMapper.readValue(
                        response.body(),
                        Map.class);
                logger.info("LLM analysis completed for transaction {}: risk_score={}",
                        transactionId, result.get("risk_score"));
                return result;
            } else {
                logger.error("LLM service returned error: {} - {}",
                        response.statusCode(), response.body());
                return createErrorResponse("LLM service error: " + response.statusCode());
            }

        } catch (Exception e) {
            logger.error("Failed to call LLM service for transaction " + transactionId, e);
            return createErrorResponse("Exception: " + e.getMessage());
        }
    }

    /**
     * Search documents for evidence related to customer/transaction
     * Uses RAG (Retrieval-Augmented Generation) system
     * 
     * @param customerId Customer ID
     * @param query      Search query
     * @param topK       Number of documents to retrieve
     * @return List of relevant document excerpts
     */
    public Map<String, Object> searchDocuments(String customerId, String query, int topK) {
        try {
            Map<String, Object> requestBody = new HashMap<>();
            requestBody.put("customer_id", customerId);
            requestBody.put("query", query);
            requestBody.put("top_k", topK);

            String jsonBody = objectMapper.writeValueAsString(requestBody);

            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(pythonApiUrl + "/api/search-documents"))
                    .header("Content-Type", "application/json; charset=UTF-8")
                    .timeout(Duration.ofSeconds(20))
                    .POST(HttpRequest.BodyPublishers.ofString(jsonBody, java.nio.charset.StandardCharsets.UTF_8))
                    .build();

            HttpResponse<String> response = httpClient.send(
                    request,
                    HttpResponse.BodyHandlers.ofString());

            if (response.statusCode() == 200) {
                return objectMapper.readValue(response.body(), Map.class);
            } else {
                logger.error("Document search failed: {} - {}",
                        response.statusCode(), response.body());
                return createErrorResponse("Search failed: " + response.statusCode());
            }

        } catch (Exception e) {
            logger.error("Failed to search documents for customer " + customerId, e);
            return createErrorResponse("Exception: " + e.getMessage());
        }
    }

    /**
     * Generate compliance report using LLM
     * Combines transaction data (from DB) with document analysis
     * 
     * @param customerId Customer ID
     * @param reportType Report type (SAR, CTR, CDD)
     * @return Generated report content
     */
    public Map<String, Object> generateComplianceReport(String customerId, String reportType) {
        try {
            Map<String, Object> requestBody = new HashMap<>();
            requestBody.put("customer_id", customerId);
            requestBody.put("report_type", reportType);

            String jsonBody = objectMapper.writeValueAsString(requestBody);

            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(pythonApiUrl + "/api/generate-report"))
                    .header("Content-Type", "application/json; charset=UTF-8")
                    .timeout(Duration.ofSeconds(60))
                    .POST(HttpRequest.BodyPublishers.ofString(jsonBody, java.nio.charset.StandardCharsets.UTF_8))
                    .build();

            HttpResponse<String> response = httpClient.send(
                    request,
                    HttpResponse.BodyHandlers.ofString());

            if (response.statusCode() == 200) {
                logger.info("Compliance report generated for customer {}", customerId);
                return objectMapper.readValue(response.body(), Map.class);
            } else {
                logger.error("Report generation failed: {} - {}",
                        response.statusCode(), response.body());
                return createErrorResponse("Report generation failed: " + response.statusCode());
            }

        } catch (Exception e) {
            logger.error("Failed to generate report for customer " + customerId, e);
            return createErrorResponse("Exception: " + e.getMessage());
        }
    }

    /**
     * Health check - verify Python service is available
     */
    public boolean isServiceHealthy() {
        try {
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(pythonApiUrl + "/health"))
                    .timeout(Duration.ofSeconds(5))
                    .GET()
                    .build();

            HttpResponse<String> response = httpClient.send(
                    request,
                    HttpResponse.BodyHandlers.ofString());

            return response.statusCode() == 200;
        } catch (Exception e) {
            logger.warn("LLM service health check failed: {}", e.getMessage());
            return false;
        }
    }

    private Map<String, Object> createErrorResponse(String message) {
        Map<String, Object> error = new HashMap<>();
        error.put("error", true);
        error.put("message", message);
        error.put("risk_score", 0.0);
        error.put("documents", new Object[0]);
        return error;
    }
}
