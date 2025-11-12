package com.bankfraud.service;

import com.bankfraud.integration.LLMServiceClient;
import com.bankfraud.model.FraudAlert;
import com.bankfraud.model.Transaction;
import com.bankfraud.repository.TransactionRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Map;

/**
 * Enhanced fraud detection combining rule-based + ML + LLM analysis
 * 
 * Integration workflow:
 * 1. Rule-based detection (Java/Scala) identifies suspicious patterns
 * 2. LLM service analyzes transaction with document context
 * 3. Combined score determines alert level
 */
public class EnhancedFraudDetectionService {
    private static final Logger logger = LoggerFactory.getLogger(EnhancedFraudDetectionService.class);
    
    private final TransactionRepository transactionRepository;
    private final LLMServiceClient llmClient;
    private final boolean llmEnabled;
    
    public EnhancedFraudDetectionService(
            TransactionRepository transactionRepository,
            String pythonApiUrl) {
        this.transactionRepository = transactionRepository;
        this.llmClient = new LLMServiceClient(pythonApiUrl);
        this.llmEnabled = llmClient.isServiceHealthy();
        
        if (llmEnabled) {
            logger.info("LLM integration enabled - Python service available at {}", pythonApiUrl);
        } else {
            logger.warn("LLM integration disabled - Python service not available. " +
                       "Falling back to rule-based detection only.");
        }
    }
    
    /**
     * Analyze transaction with enhanced detection
     * Combines rule-based fraud score with LLM analysis
     */
    public FraudAlert analyzeTransaction(Transaction transaction) {
        // Step 1: Basic rule-based fraud score (existing logic)
        double ruleBasedScore = calculateRuleBasedScore(transaction);
        
        // Step 2: If suspicious and LLM available, get enhanced analysis
        double finalScore = ruleBasedScore;
        String analysisDetails = "Rule-based detection";
        List<Map<String, Object>> documentEvidence = null;
        
        if (ruleBasedScore > 0.5 && llmEnabled) {
            try {
                logger.info("Transaction {} flagged by rules (score={}), requesting LLM analysis", 
                           transaction.getTransactionId(), ruleBasedScore);
                
                // Call Python LLM service for enhanced analysis
                Map<String, Object> llmAnalysis = llmClient.analyzeTransaction(
                    transaction.getTransactionId(),
                    transaction.getCustomerId(),
                    transaction.getAmount().doubleValue(),
                    transaction.getMerchantName()
                );
                
                if (!llmAnalysis.containsKey("error")) {
                    // Combine rule-based score with LLM risk score
                    double llmScore = ((Number) llmAnalysis.get("risk_score")).doubleValue();
                    finalScore = (ruleBasedScore * 0.6) + (llmScore * 0.4); // Weighted average
                    
                    analysisDetails = String.format(
                        "Enhanced detection (Rule: %.2f, LLM: %.2f, Combined: %.2f). Reasoning: %s",
                        ruleBasedScore, llmScore, finalScore, 
                        llmAnalysis.get("reasoning")
                    );
                    
                    // Get document evidence if available
                    if (llmAnalysis.containsKey("supporting_documents")) {
                        documentEvidence = (List<Map<String, Object>>) llmAnalysis.get("supporting_documents");
                        logger.info("Found {} supporting documents for transaction {}", 
                                   documentEvidence.size(), transaction.getTransactionId());
                    }
                } else {
                    logger.warn("LLM analysis returned error for {}: {}", 
                               transaction.getTransactionId(), llmAnalysis.get("message"));
                }
                
            } catch (Exception e) {
                logger.error("LLM analysis failed for transaction " + 
                            transaction.getTransactionId() + ", using rule-based score only", e);
            }
        }
        
        // Step 3: Create alert with combined analysis
        FraudAlert alert = new FraudAlert();
        alert.setTransactionId(transaction.getTransactionId());
        alert.setCustomerId(transaction.getCustomerId());
        alert.setRiskScore(finalScore);
        alert.setRiskLevel(determineRiskLevel(finalScore));
        alert.setAnalysisDetails(analysisDetails);
        alert.setDetectionMethod(llmEnabled && finalScore != ruleBasedScore ? 
                                "HYBRID_RULE_LLM" : "RULE_BASED");
        
        if (documentEvidence != null && !documentEvidence.isEmpty()) {
            alert.setDocumentEvidenceCount(documentEvidence.size());
            // Store evidence references in alert
            StringBuilder evidenceSummary = new StringBuilder("Supporting documents: ");
            for (Map<String, Object> doc : documentEvidence) {
                evidenceSummary.append(doc.get("source")).append("; ");
            }
            alert.setEvidenceSummary(evidenceSummary.toString());
        }
        
        logger.info("Fraud analysis complete for {}: score={}, level={}, method={}", 
                   transaction.getTransactionId(), finalScore, alert.getRiskLevel(), 
                   alert.getDetectionMethod());
        
        return alert;
    }
    
    /**
     * Generate compliance report for customer
     * Calls Python service to combine transaction data + document analysis
     */
    public String generateComplianceReport(String customerId, String reportType) {
        if (!llmEnabled) {
            logger.warn("Cannot generate compliance report - LLM service not available");
            return "ERROR: LLM service not available for report generation";
        }
        
        try {
            logger.info("Generating {} report for customer {}", reportType, customerId);
            
            Map<String, Object> reportResult = llmClient.generateComplianceReport(
                customerId, 
                reportType
            );
            
            if (!reportResult.containsKey("error")) {
                String reportContent = (String) reportResult.get("report_content");
                logger.info("Successfully generated {} report for customer {} ({} chars)", 
                           reportType, customerId, reportContent.length());
                return reportContent;
            } else {
                logger.error("Report generation failed: {}", reportResult.get("message"));
                return "ERROR: " + reportResult.get("message");
            }
            
        } catch (Exception e) {
            logger.error("Failed to generate compliance report for customer " + customerId, e);
            return "ERROR: Exception during report generation: " + e.getMessage();
        }
    }
    
    /**
     * Search customer documents via Python RAG system
     */
    public List<Map<String, Object>> searchCustomerDocuments(String customerId, String query) {
        if (!llmEnabled) {
            logger.warn("Cannot search documents - LLM service not available");
            return List.of();
        }
        
        try {
            Map<String, Object> searchResult = llmClient.searchDocuments(customerId, query, 5);
            
            if (!searchResult.containsKey("error")) {
                List<Map<String, Object>> documents = 
                    (List<Map<String, Object>>) searchResult.get("documents");
                logger.info("Document search for customer {} query '{}' returned {} results", 
                           customerId, query, documents.size());
                return documents;
            } else {
                logger.error("Document search failed: {}", searchResult.get("message"));
                return List.of();
            }
            
        } catch (Exception e) {
            logger.error("Failed to search documents for customer " + customerId, e);
            return List.of();
        }
    }
    
    /**
     * Calculate rule-based fraud score (existing logic)
     * This is the baseline that gets enhanced by LLM
     */
    private double calculateRuleBasedScore(Transaction transaction) {
        double score = 0.0;
        
        // High amount threshold
        if (transaction.getAmount().doubleValue() > 5000) {
            score += 0.3;
        }
        
        // Unusual merchant
        if (isUnusualMerchant(transaction.getMerchantName())) {
            score += 0.2;
        }
        
        // Velocity check - frequent transactions
        long recentTransactionCount = transactionRepository.countRecentTransactions(
            transaction.getCustomerId(),
            24 // last 24 hours
        );
        if (recentTransactionCount > 10) {
            score += 0.3;
        }
        
        // Time-based patterns (e.g., unusual hours)
        if (isUnusualTime(transaction.getTransactionDate())) {
            score += 0.2;
        }
        
        return Math.min(score, 1.0); // Cap at 1.0
    }
    
    private String determineRiskLevel(double score) {
        if (score >= 0.8) return "CRITICAL";
        if (score >= 0.6) return "HIGH";
        if (score >= 0.4) return "MEDIUM";
        return "LOW";
    }
    
    private boolean isUnusualMerchant(String merchantName) {
        // Simplified logic - in production, check against known merchants
        return merchantName == null || 
               merchantName.toLowerCase().contains("unknown") ||
               merchantName.toLowerCase().contains("test");
    }
    
    private boolean isUnusualTime(java.time.LocalDateTime transactionDate) {
        // Simplified logic - flag transactions between 2 AM and 5 AM
        int hour = transactionDate.getHour();
        return hour >= 2 && hour < 5;
    }
}
