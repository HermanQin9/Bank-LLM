package com.bankfraud.integration;

import com.bankfraud.config.DatabaseConfig;
import com.bankfraud.model.Transaction;
import com.bankfraud.reader.CsvDataReader;
import com.bankfraud.repository.TransactionRepository;
import com.bankfraud.service.DataIngestionService;
import com.bankfraud.service.TransactionNormalizer;
import org.junit.jupiter.api.*;
import org.junit.jupiter.api.io.TempDir;
import org.testcontainers.containers.PostgreSQLContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;

import java.io.IOException;
import java.math.BigDecimal;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDateTime;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Integration tests for the complete data ingestion pipeline.
 * Uses Testcontainers to spin up a real PostgreSQL database.
 * 
 * @author Banking Platform Team
 * @version 1.0
 */
@Testcontainers
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class DataIngestionIntegrationTest {

    @Container
    static PostgreSQLContainer<?> postgres = new PostgreSQLContainer<>("postgres:15-alpine")
            .withDatabaseName("testdb")
            .withUsername("test")
            .withPassword("test");

    private static TransactionRepository transactionRepository;

    @TempDir
    Path tempDir;

    @BeforeAll
    static void beforeAll() {
        // Wait for container to be ready
        assertTrue(postgres.isRunning(), "PostgreSQL container should be running");

        // Set database connection properties for tests
        System.setProperty("db.url", postgres.getJdbcUrl());
        System.setProperty("db.username", postgres.getUsername());
        System.setProperty("db.password", postgres.getPassword());

        // Force DatabaseConfig to reload with test properties
        DatabaseConfig.closeDataSource();

        transactionRepository = new TransactionRepository();

        // Create tables
        createTables();
    }

    @AfterAll
    static void afterAll() {
        DatabaseConfig.closeDataSource();
    }

    private static void createTables() {
        String createTableSQL = """
                CREATE TABLE IF NOT EXISTS transactions (
                    transaction_id VARCHAR(50) PRIMARY KEY,
                    customer_id VARCHAR(50),
                    amount DECIMAL(12, 2) NOT NULL,
                    transaction_date TIMESTAMP NOT NULL,
                    merchant_name VARCHAR(255),
                    merchant_category VARCHAR(50),
                    location VARCHAR(255),
                    fraud_flag BOOLEAN DEFAULT FALSE,
                    source_system VARCHAR(50),
                    status VARCHAR(20),
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                );
                """;

        try (var conn = DatabaseConfig.getDataSource().getConnection();
                var stmt = conn.createStatement()) {
            stmt.execute(createTableSQL);
        } catch (Exception e) {
            fail("Failed to create test tables: " + e.getMessage());
        }
    }

    @Test
    @Order(1)
    @DisplayName("Integration Test: End-to-End CSV Ingestion Pipeline")
    void testCompleteCSVIngestionPipeline() throws IOException {
        // Given: A CSV file with transaction data
        String csvContent = """
                transaction_id,customer_id,amount,transaction_date,merchant_name,merchant_category
                TXN_INT_001,CUST001,250.00,2024-11-01,Amazon,Retail
                TXN_INT_002,CUST002,150.50,2024-11-02,Starbucks,Food
                TXN_INT_003,CUST003,999.99,2024-11-03,Apple Store,Electronics
                """;

        Path csvFile = tempDir.resolve("integration_test.csv");
        Files.writeString(csvFile, csvContent);

        // When: Running the complete ingestion pipeline
        DataIngestionService service = new DataIngestionService();
        DataIngestionService.IngestionResult result = service.ingestFile(csvFile.toString());

        // Then: Pipeline should complete successfully
        assertTrue(result.isSuccess(), "Ingestion should succeed");
        assertEquals(3, result.getRecordsRead(), "Should read 3 records");
        assertEquals(3, result.getRecordsNormalized(), "Should normalize 3 records");
        assertEquals(3, result.getRecordsSaved(), "Should save 3 records");
        assertTrue(result.getDurationMs() > 0, "Duration should be measured");

        // And: Data should be retrievable from database
        var transaction = transactionRepository.findById("TXN_INT_001");
        assertTrue(transaction.isPresent(), "Transaction should be in database");
        assertEquals("CUST001", transaction.get().getCustomerId());
        assertEquals(new BigDecimal("250.00"), transaction.get().getAmount());
        assertEquals("Amazon", transaction.get().getMerchantName());
    }

    @Test
    @Order(2)
    @DisplayName("Integration Test: Database Transaction Queries")
    void testDatabaseQueries() {
        // Given: Transactions already in database from previous test

        // When: Querying by customer ID
        List<Transaction> customerTransactions = transactionRepository.findByCustomerId("CUST001");

        // Then: Should find the customer's transaction
        assertFalse(customerTransactions.isEmpty(), "Should find transactions");
        assertEquals("TXN_INT_001", customerTransactions.get(0).getTransactionId());

        // When: Querying by date range
        LocalDateTime start = LocalDateTime.of(2024, 11, 1, 0, 0);
        LocalDateTime end = LocalDateTime.of(2024, 11, 30, 23, 59);
        List<Transaction> dateRangeTransactions = transactionRepository.findByDateRange(start, end);

        // Then: Should find all transactions in range
        assertTrue(dateRangeTransactions.size() >= 3, "Should find at least 3 transactions");
    }

    @Test
    @Order(3)
    @DisplayName("Integration Test: Batch Insert Performance")
    void testBatchInsertPerformance() {
        // Given: 100 transactions to insert
        List<Transaction> transactions = generateTestTransactions(100);

        // When: Performing batch insert
        long startTime = System.currentTimeMillis();
        int savedCount = transactionRepository.saveBatch(transactions);
        long duration = System.currentTimeMillis() - startTime;

        // Then: Should insert all records quickly
        assertEquals(100, savedCount, "Should save all 100 transactions");
        assertTrue(duration < 5000, "Should complete in less than 5 seconds");

        // And: Records should be queryable
        var transaction = transactionRepository.findById("BATCH_TXN_001");
        assertTrue(transaction.isPresent(), "Batch transaction should be in database");
    }

    @Test
    @Order(4)
    @DisplayName("Integration Test: Update and Delete Operations")
    void testUpdateAndDeleteOperations() {
        // Given: An existing transaction
        var existingTransaction = transactionRepository.findById("TXN_INT_001");
        assertTrue(existingTransaction.isPresent());

        // When: Updating transaction status
        boolean updated = transactionRepository.updateStatus("TXN_INT_001", "APPROVED");

        // Then: Update should succeed
        assertTrue(updated, "Update should succeed");

        // And: Status should be updated in database
        var updatedTransaction = transactionRepository.findById("TXN_INT_001");
        assertTrue(updatedTransaction.isPresent());
        assertEquals("APPROVED", updatedTransaction.get().getStatus());

        // When: Deleting transaction
        boolean deleted = transactionRepository.delete("TXN_INT_001");

        // Then: Delete should succeed
        assertTrue(deleted, "Delete should succeed");

        // And: Transaction should no longer exist
        var deletedTransaction = transactionRepository.findById("TXN_INT_001");
        assertFalse(deletedTransaction.isPresent(), "Transaction should be deleted");
    }

    @Test
    @Order(5)
    @DisplayName("Integration Test: Error Handling with Invalid Data")
    void testErrorHandlingWithInvalidData() throws IOException {
        // Given: CSV with some invalid records
        String csvContent = """
                transaction_id,customer_id,amount,transaction_date
                VALID_TXN,CUST999,100.00,2024-11-01
                ,CUST999,200.00,2024-11-02
                NO_AMOUNT_TXN,CUST999,,2024-11-03
                """;

        Path csvFile = tempDir.resolve("invalid_data.csv");
        Files.writeString(csvFile, csvContent);

        // When: Running ingestion with invalid data
        DataIngestionService service = new DataIngestionService();
        DataIngestionService.IngestionResult result = service.ingestFile(csvFile.toString());

        // Then: Should handle errors gracefully
        assertTrue(result.isSuccess(), "Pipeline should not crash");
        assertEquals(3, result.getRecordsRead(), "Should read all 3 records");
        assertTrue(result.getRecordsNormalized() < 3, "Some records should fail normalization");

        // And: Valid records should still be saved
        assertTrue(result.getRecordsSaved() > 0, "Valid records should be saved");
    }

    /**
     * Helper method to generate test transactions.
     */
    private List<Transaction> generateTestTransactions(int count) {
        List<Transaction> transactions = new java.util.ArrayList<>();

        for (int i = 1; i <= count; i++) {
            Transaction transaction = new Transaction();
            transaction.setTransactionId(String.format("BATCH_TXN_%03d", i));
            transaction.setCustomerId("BATCH_CUST_" + (i % 10));
            transaction.setAmount(new BigDecimal(String.format("%d.%02d", i * 10, i % 100)));
            transaction.setTransactionDate(LocalDateTime.now().minusDays(i));
            transaction.setMerchantName("Merchant " + i);
            transaction.setMerchantCategory("Category " + (i % 5));
            transaction.setSourceSystem("TEST");
            transaction.setStatus("PENDING");
            transaction.setFraudFlag(i % 20 == 0); // 5% fraud rate

            transactions.add(transaction);
        }

        return transactions;
    }
}
