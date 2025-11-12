package com.bankfraud.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.List;

/**
 * Fraud alert domain model representing a detected suspicious activity.
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class FraudAlert {

    private Long alertId;
    private String transactionId;
    private String customerId;
    private String alertType;
    private BigDecimal fraudScore;
    private RiskLevel riskLevel;
    private List<String> rulesTriggered;
    private String description;
    private AlertStatus status;
    private LocalDateTime createdAt;
    private LocalDateTime reviewedAt;
    private String reviewedBy;

    /**
     * Check if alert is critical (fraud score > 75)
     */
    public boolean isCritical() {
        return fraudScore != null &&
                fraudScore.compareTo(new BigDecimal("75")) > 0;
    }

    /**
     * Check if alert is pending review
     */
    public boolean isPending() {
        return status == AlertStatus.PENDING;
    }

    /**
     * Mark alert as reviewed
     */
    public void markAsReviewed(String reviewer, AlertStatus newStatus) {
        this.reviewedBy = reviewer;
        this.reviewedAt = LocalDateTime.now();
        this.status = newStatus;
    }

    /**
     * Risk level enum
     */
    public enum RiskLevel {
        LOW, MEDIUM, HIGH, CRITICAL
    }

    /**
     * Alert status enum
     */
    public enum AlertStatus {
        PENDING,
        CONFIRMED,
        FALSE_POSITIVE,
        UNDER_REVIEW
    }
}
