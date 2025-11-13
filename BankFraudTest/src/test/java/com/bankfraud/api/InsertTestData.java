package com.bankfraud.api;

import com.bankfraud.config.DatabaseConfig;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.Timestamp;
import java.time.LocalDateTime;

/**
 * Utility to insert test data for REST API validation.
 */
public class InsertTestData {

    private static final Logger logger = LoggerFactory.getLogger(InsertTestData.class);

    public static void main(String[] args) {
        try {
            insertTestCustomers();
            insertTestTransactions();
            insertTestAlerts();
            insertTestEvidence();
            logger.info("Test data inserted successfully");
        } catch (Exception e) {
            logger.error("Failed to insert test data", e);
            System.exit(1);
        }
    }

    private static void insertTestCustomers() throws Exception {
        String sql = "INSERT INTO customers (customer_id, first_name, last_name, email, account_created_date) "
                + "VALUES (?, ?, ?, ?, ?) "
                + "ON CONFLICT (customer_id) DO NOTHING";

        try (Connection conn = DatabaseConfig.getDataSource().getConnection();
                PreparedStatement ps = conn.prepareStatement(sql)) {

            ps.setString(1, "CUST-789");
            ps.setString(2, "Test");
            ps.setString(3, "Customer789");
            ps.setString(4, "test789@example.com");
            ps.setDate(5, java.sql.Date.valueOf(LocalDateTime.now().minusYears(2).toLocalDate()));
            ps.addBatch();

            ps.setString(1, "CUST-790");
            ps.setString(2, "Test");
            ps.setString(3, "Customer790");
            ps.setString(4, "test790@example.com");
            ps.setDate(5, java.sql.Date.valueOf(LocalDateTime.now().minusYears(1).toLocalDate()));
            ps.addBatch();

            ps.setString(1, "CUST-791");
            ps.setString(2, "Test");
            ps.setString(3, "Customer791");
            ps.setString(4, "test791@example.com");
            ps.setDate(5, java.sql.Date.valueOf(LocalDateTime.now().minusMonths(6).toLocalDate()));
            ps.addBatch();

            int[] results = ps.executeBatch();
            logger.info("Inserted {} test customers", results.length);
        }
    }

    private static void insertTestTransactions() throws Exception {
        String sql = "INSERT INTO transactions (transaction_id, customer_id, transaction_date, amount, "
                + "merchant_name, merchant_category, transaction_type, source_system) "
                + "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                + "ON CONFLICT (transaction_id) DO NOTHING";

        try (Connection conn = DatabaseConfig.getDataSource().getConnection();
                PreparedStatement ps = conn.prepareStatement(sql)) {

            // Transaction 1
            ps.setString(1, "TXN-12345");
            ps.setString(2, "CUST-789");
            ps.setTimestamp(3, Timestamp.valueOf(LocalDateTime.now().minusHours(2)));
            ps.setDouble(4, 25000.00);
            ps.setString(5, "Online Retailer XYZ");
            ps.setString(6, "RETAIL");
            ps.setString(7, "PURCHASE");
            ps.setString(8, "TEST_SYSTEM");
            ps.addBatch();

            // Transaction 2
            ps.setString(1, "TXN-12346");
            ps.setString(2, "CUST-790");
            ps.setTimestamp(3, Timestamp.valueOf(LocalDateTime.now().minusHours(1)));
            ps.setDouble(4, 500.00);
            ps.setString(5, "Gas Station ABC");
            ps.setString(6, "GAS");
            ps.setString(7, "PURCHASE");
            ps.setString(8, "TEST_SYSTEM");
            ps.addBatch();

            // Transaction 3
            ps.setString(1, "TXN-12347");
            ps.setString(2, "CUST-791");
            ps.setTimestamp(3, Timestamp.valueOf(LocalDateTime.now().minusMinutes(30)));
            ps.setDouble(4, 1200.00);
            ps.setString(5, "International Store DEF");
            ps.setString(6, "INTERNATIONAL");
            ps.setString(7, "PURCHASE");
            ps.setString(8, "TEST_SYSTEM");
            ps.addBatch();

            int[] results = ps.executeBatch();
            logger.info("Inserted {} test transactions", results.length);
        }
    }

    private static void insertTestAlerts() throws Exception {
        String sql = "INSERT INTO transaction_alerts (alert_id, transaction_id, customer_id, alert_type, "
                + "severity, recommended_action, status, deviation_details, supporting_evidence, created_at) "
                + "VALUES (?, ?, ?, ?, ?, ?, ?, ?::jsonb, ?::jsonb, ?) "
                + "ON CONFLICT (alert_id) DO NOTHING";

        try (Connection conn = DatabaseConfig.getDataSource().getConnection();
                PreparedStatement ps = conn.prepareStatement(sql)) {

            // Alert 1
            ps.setString(1, "TEST-ALERT-001");
            ps.setString(2, "TXN-12345");
            ps.setString(3, "CUST-789");
            ps.setString(4, "AMOUNT_ANOMALY");
            ps.setString(5, "HIGH");
            ps.setString(6, "BLOCK");
            ps.setString(7, "PENDING");
            ps.setString(8, "{\"z_score\": 4.2, \"expected_range\": \"$1K-$5K\", \"actual_amount\": \"$25K\"}");
            ps.setString(9, "[\"Large deviation from customer profile\", \"Unusual transaction time\"]");
            ps.setTimestamp(10, Timestamp.valueOf(LocalDateTime.now().minusHours(2)));
            ps.addBatch();

            // Alert 2
            ps.setString(1, "TEST-ALERT-002");
            ps.setString(2, "TXN-12346");
            ps.setString(3, "CUST-790");
            ps.setString(4, "VELOCITY_CHECK");
            ps.setString(5, "MEDIUM");
            ps.setString(6, "REVIEW");
            ps.setString(7, "PENDING");
            ps.setString(8, "{\"transaction_count\": 15, \"time_window\": \"1 hour\", \"threshold\": 5}");
            ps.setString(9, "[\"Multiple transactions in short time\"]");
            ps.setTimestamp(10, Timestamp.valueOf(LocalDateTime.now().minusHours(1)));
            ps.addBatch();

            // Alert 3
            ps.setString(1, "TEST-ALERT-003");
            ps.setString(2, "TXN-12347");
            ps.setString(3, "CUST-791");
            ps.setString(4, "PROFILE_DEVIATION");
            ps.setString(5, "LOW");
            ps.setString(6, "MONITOR");
            ps.setString(7, "PENDING");
            ps.setString(8, "{\"merchant_type\": \"overseas\", \"customer_history\": \"domestic_only\"}");
            ps.setString(9, "[\"First international transaction\"]");
            ps.setTimestamp(10, Timestamp.valueOf(LocalDateTime.now().minusMinutes(30)));
            ps.addBatch();

            int[] results = ps.executeBatch();
            logger.info("Inserted {} test alerts", results.length);
        }
    }

    private static void insertTestEvidence() throws Exception {
        String sql = "INSERT INTO document_evidence (alert_id, customer_id, transaction_id, "
                + "document_type, document_path, excerpt, relevance_score, extracted_at) "
                + "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                + "ON CONFLICT DO NOTHING";

        try (Connection conn = DatabaseConfig.getDataSource().getConnection();
                PreparedStatement ps = conn.prepareStatement(sql)) {

            // Evidence for Alert 1
            ps.setString(1, "TEST-ALERT-001");
            ps.setString(2, "CUST-789");
            ps.setString(3, "TXN-12345");
            ps.setString(4, "invoice");
            ps.setString(5, "/docs/invoices/inv_12345.pdf");
            ps.setString(6, "Invoice amount $25,000 significantly exceeds customer's typical range");
            ps.setDouble(7, 0.92);
            ps.setTimestamp(8, Timestamp.valueOf(LocalDateTime.now().minusHours(2)));
            ps.addBatch();

            ps.setString(1, "TEST-ALERT-001");
            ps.setString(2, "CUST-789");
            ps.setString(3, "TXN-12345");
            ps.setString(4, "contract");
            ps.setString(5, "/docs/contracts/contract_789.pdf");
            ps.setString(6, "Customer contract specifies monthly transaction limit of $10K");
            ps.setDouble(7, 0.88);
            ps.setTimestamp(8, Timestamp.valueOf(LocalDateTime.now().minusHours(2)));
            ps.addBatch();

            // Evidence for Alert 2
            ps.setString(1, "TEST-ALERT-002");
            ps.setString(2, "CUST-790");
            ps.setString(3, "TXN-12346");
            ps.setString(4, "transaction_log");
            ps.setString(5, "/logs/txn_log_790.txt");
            ps.setString(6, "15 transactions detected within 60-minute window");
            ps.setDouble(7, 0.95);
            ps.setTimestamp(8, Timestamp.valueOf(LocalDateTime.now().minusHours(1)));
            ps.addBatch();

            // Evidence for Alert 3
            ps.setString(1, "TEST-ALERT-003");
            ps.setString(2, "CUST-791");
            ps.setString(3, "TXN-12347");
            ps.setString(4, "customer_profile");
            ps.setString(5, "/profiles/profile_791.json");
            ps.setString(6, "Customer profile indicates domestic-only transaction history");
            ps.setDouble(7, 0.85);
            ps.setTimestamp(8, Timestamp.valueOf(LocalDateTime.now().minusMinutes(30)));
            ps.addBatch();

            int[] results = ps.executeBatch();
            logger.info("Inserted {} test evidence records", results.length);
        }
    }
}
