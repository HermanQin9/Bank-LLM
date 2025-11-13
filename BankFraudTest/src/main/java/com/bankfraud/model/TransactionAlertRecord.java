package com.bankfraud.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

/**
 * Projection of the transaction_alerts table including JSON payloads.
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class TransactionAlertRecord {

    private String alertId;
    private String transactionId;
    private String customerId;
    private String alertType;
    private String severity;
    private String recommendedAction;
    private String status;
    private LocalDateTime createdAt;
    private LocalDateTime resolvedAt;
    private Map<String, Object> deviationDetails;
    private List<String> supportingEvidence;
    private List<DocumentEvidenceRecord> documentEvidence;
}
