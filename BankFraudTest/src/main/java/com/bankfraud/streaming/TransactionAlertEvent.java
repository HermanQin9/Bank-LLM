package com.bankfraud.streaming;

import com.bankfraud.model.FraudAlert;
import com.bankfraud.model.Transaction;
import lombok.Builder;
import lombok.Value;

import java.math.BigDecimal;
import java.time.OffsetDateTime;
import java.time.ZoneOffset;
import java.util.List;
import java.util.UUID;

/**
 * Immutable payload published to Kafka whenever a fraud alert is generated.
 * Keeps Java, Python, and Scala services in sync via the shared event bus.
 */
@Value
@Builder
public class TransactionAlertEvent {

    String eventId;
    String alertId;
    String transactionId;
    String customerId;
    BigDecimal ruleBasedScore;
    BigDecimal finalScore;
    String riskLevel;
    String detectionMethod;
    List<String> rulesTriggered;
    String merchantName;
    BigDecimal amount;
    String currency;
    String summary;
    OffsetDateTime createdAt;

    public static TransactionAlertEvent from(Transaction transaction, FraudAlert alert, double ruleScore,
            double finalScore) {
        return TransactionAlertEvent.builder()
                .eventId(UUID.randomUUID().toString())
                .alertId(alert.getAlertId() != null
                        ? alert.getAlertId().toString()
                        : "ALERT-" + transaction.getTransactionId())
                .transactionId(transaction.getTransactionId())
                .customerId(transaction.getCustomerId())
                .ruleBasedScore(BigDecimal.valueOf(ruleScore * 100.0))
                .finalScore(BigDecimal.valueOf(finalScore * 100.0))
                .riskLevel(alert.getRiskLevel() != null ? alert.getRiskLevel().name() : "UNKNOWN")
                .detectionMethod(alert.getAlertType())
                .rulesTriggered(alert.getRulesTriggered())
                .merchantName(transaction.getMerchantName())
                .amount(transaction.getAmount())
                .currency(transaction.getCurrency())
                .summary(alert.getDescription())
                .createdAt(OffsetDateTime.now(ZoneOffset.UTC))
                .build();
    }
}
