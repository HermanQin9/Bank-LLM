package com.bankfraud.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.math.BigDecimal;
import java.time.LocalDateTime;

/**
 * Transaction domain model representing a normalized financial transaction.
 * Supports data from multiple banking systems.
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class Transaction {

    private String transactionId;
    private String customerId;
    private LocalDateTime transactionDate;
    private BigDecimal amount;
    private String currency;
    private String merchantName;
    private String merchantCategory;
    private String transactionType;
    private String cardLastFour;
    private String locationCountry;
    private String locationCity;
    private String location; // Combined location string (city, state, zip)
    private String ipAddress;
    private String deviceFingerprint;
    private boolean isOnline;
    private boolean fraudFlag; // Indicates if transaction is fraudulent
    private String status; // PENDING, APPROVED, REJECTED, INVESTIGATING
    private String sourceSystem;
    private String rawData; // JSON string
    private LocalDateTime normalizedAt;
    private LocalDateTime createdAt;

    /**
     * Check if transaction is high-value (over $10,000)
     */
    public boolean isHighValue() {
        return amount != null && amount.compareTo(new BigDecimal("10000")) > 0;
    }

    /**
     * Check if transaction is international
     */
    public boolean isInternational(String homeCountry) {
        return locationCountry != null && !locationCountry.equalsIgnoreCase(homeCountry);
    }

    /**
     * Check if transaction occurred during unusual hours (2 AM - 5 AM)
     */
    public boolean isDuringUnusualHours() {
        if (transactionDate == null)
            return false;
        int hour = transactionDate.getHour();
        return hour >= 2 && hour < 5;
    }
}
