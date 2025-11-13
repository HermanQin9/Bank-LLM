package com.bankfraud.repository;

import com.bankfraud.config.DatabaseConfig;
import com.bankfraud.model.DocumentEvidenceRecord;
import com.bankfraud.model.TransactionAlertRecord;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Timestamp;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * DAO that exposes the transaction_alerts + document_evidence tables for Java
 * dashboards.
 */
public class TransactionAlertRepository {

    private static final Logger logger = LoggerFactory.getLogger(TransactionAlertRepository.class);

    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper()
            .registerModule(new JavaTimeModule());

    private static final TypeReference<Map<String, Object>> MAP_TYPE = new TypeReference<>() {
    };
    private static final TypeReference<List<String>> LIST_TYPE = new TypeReference<>() {
    };

    private static final String BASE_SELECT = "SELECT alert_id, transaction_id, customer_id, alert_type, severity, "
            + "recommended_action, status, deviation_details, supporting_evidence, created_at, resolved_at "
            + "FROM transaction_alerts";

    private static final String SELECT_LATEST = BASE_SELECT + " ORDER BY created_at DESC LIMIT ?";
    private static final String SELECT_BY_ID = BASE_SELECT + " WHERE alert_id = ?";
    private static final String SELECT_EVIDENCE = "SELECT evidence_id, alert_id, document_type, document_path, "
            + "excerpt, relevance_score, extracted_at FROM document_evidence WHERE alert_id = ? ORDER BY extracted_at DESC";

    public List<TransactionAlertRecord> findRecent(int limit) {
        List<TransactionAlertRecord> alerts = new ArrayList<>();
        try (Connection connection = DatabaseConfig.getDataSource().getConnection();
                PreparedStatement ps = connection.prepareStatement(SELECT_LATEST)) {
            ps.setInt(1, Math.max(1, Math.min(limit, 200)));
            try (ResultSet rs = ps.executeQuery()) {
                while (rs.next()) {
                    alerts.add(mapAlert(rs));
                }
            }
        } catch (SQLException e) {
            logger.error("Failed to load recent transaction alerts", e);
        }
        return alerts;
    }

    public Optional<TransactionAlertRecord> findByIdWithEvidence(String alertId) {
        Optional<TransactionAlertRecord> alert = findById(alertId);
        alert.ifPresent(record -> record.setDocumentEvidence(loadEvidence(alertId)));
        return alert;
    }

    private Optional<TransactionAlertRecord> findById(String alertId) {
        try (Connection connection = DatabaseConfig.getDataSource().getConnection();
                PreparedStatement ps = connection.prepareStatement(SELECT_BY_ID)) {
            ps.setString(1, alertId);
            try (ResultSet rs = ps.executeQuery()) {
                if (rs.next()) {
                    return Optional.of(mapAlert(rs));
                }
            }
        } catch (SQLException e) {
            logger.error("Failed to load transaction alert {}", alertId, e);
        }
        return Optional.empty();
    }

    private List<DocumentEvidenceRecord> loadEvidence(String alertId) {
        List<DocumentEvidenceRecord> evidence = new ArrayList<>();
        try (Connection connection = DatabaseConfig.getDataSource().getConnection();
                PreparedStatement ps = connection.prepareStatement(SELECT_EVIDENCE)) {
            ps.setString(1, alertId);
            try (ResultSet rs = ps.executeQuery()) {
                while (rs.next()) {
                    evidence.add(mapEvidence(rs));
                }
            }
        } catch (SQLException e) {
            logger.error("Failed to load document evidence for alert {}", alertId, e);
        }
        return evidence;
    }

    private TransactionAlertRecord mapAlert(ResultSet rs) throws SQLException {
        return TransactionAlertRecord.builder()
                .alertId(rs.getString("alert_id"))
                .transactionId(rs.getString("transaction_id"))
                .customerId(rs.getString("customer_id"))
                .alertType(rs.getString("alert_type"))
                .severity(rs.getString("severity"))
                .recommendedAction(rs.getString("recommended_action"))
                .status(rs.getString("status"))
                .createdAt(readDateTime(rs.getTimestamp("created_at")))
                .resolvedAt(readDateTime(rs.getTimestamp("resolved_at")))
                .deviationDetails(readJsonMap(rs, "deviation_details"))
                .supportingEvidence(readJsonList(rs, "supporting_evidence"))
                .documentEvidence(new ArrayList<>())
                .build();
    }

    private DocumentEvidenceRecord mapEvidence(ResultSet rs) throws SQLException {
        return DocumentEvidenceRecord.builder()
                .evidenceId(rs.getObject("evidence_id") != null ? rs.getLong("evidence_id") : null)
                .alertId(rs.getString("alert_id"))
                .documentType(rs.getString("document_type"))
                .documentPath(rs.getString("document_path"))
                .excerpt(rs.getString("excerpt"))
                .relevanceScore(rs.getObject("relevance_score") != null ? rs.getDouble("relevance_score") : null)
                .extractedAt(readDateTime(rs.getTimestamp("extracted_at")))
                .build();
    }

    private Map<String, Object> readJsonMap(ResultSet rs, String column) {
        String payload;
        try {
            payload = rs.getString(column);
        } catch (SQLException e) {
            logger.warn("Unable to read column {}", column, e);
            return Collections.emptyMap();
        }
        if (payload == null || payload.isBlank()) {
            return Collections.emptyMap();
        }
        try {
            return OBJECT_MAPPER.readValue(payload, MAP_TYPE);
        } catch (Exception e) {
            logger.warn("Failed to parse JSON map for column {}", column, e);
            return Collections.emptyMap();
        }
    }

    private List<String> readJsonList(ResultSet rs, String column) {
        String payload;
        try {
            payload = rs.getString(column);
        } catch (SQLException e) {
            logger.warn("Unable to read column {}", column, e);
            return Collections.emptyList();
        }
        if (payload == null || payload.isBlank()) {
            return Collections.emptyList();
        }
        try {
            return OBJECT_MAPPER.readValue(payload, LIST_TYPE);
        } catch (Exception e) {
            logger.warn("Failed to parse JSON list for column {}", column, e);
            return Collections.emptyList();
        }
    }

    private LocalDateTime readDateTime(Timestamp timestamp) {
        return timestamp != null ? timestamp.toLocalDateTime() : null;
    }
}
