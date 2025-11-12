package com.bankfraud.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.math.BigDecimal;
import java.time.LocalDate;
import java.time.LocalDateTime;

/**
 * Customer domain model with risk profiling.
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class Customer {

    private String customerId;
    private String firstName;
    private String lastName;
    private String email;
    private String phone;
    private LocalDate accountCreatedDate;
    private RiskLevel riskLevel;
    private int totalTransactions;
    private BigDecimal lifetimeValue;
    private LocalDateTime lastTransactionDate;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    /**
     * Get full name
     */
    public String getFullName() {
        return firstName + " " + lastName;
    }

    /**
     * Check if customer is new (account created within 30 days)
     */
    public boolean isNewCustomer() {
        if (accountCreatedDate == null)
            return false;
        return accountCreatedDate.isAfter(LocalDate.now().minusDays(30));
    }

    /**
     * Check if customer is high-value (lifetime value > $50,000)
     */
    public boolean isHighValueCustomer() {
        return lifetimeValue != null &&
                lifetimeValue.compareTo(new BigDecimal("50000")) > 0;
    }

    /**
     * Risk level enum
     */
    public enum RiskLevel {
        LOW, MEDIUM, HIGH, CRITICAL
    }
}
