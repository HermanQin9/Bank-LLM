package com.bankfraud.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

/**
 * Lightweight view of evidence snippets persisted by Python/LLM services.
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class DocumentEvidenceRecord {

    private Long evidenceId;
    private String alertId;
    private String documentType;
    private String documentPath;
    private String excerpt;
    private Double relevanceScore;
    private LocalDateTime extractedAt;
}
