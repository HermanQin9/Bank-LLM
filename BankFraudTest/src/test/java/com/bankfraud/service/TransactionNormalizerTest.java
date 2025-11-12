package com.bankfraud.service;

import com.bankfraud.model.Transaction;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for TransactionNormalizer.
 * Tests data normalization from various formats.
 * 
 * @author Banking Platform Team
 * @version 1.0
 */
class TransactionNormalizerTest {
    
    private TransactionNormalizer normalizer;
    
    @BeforeEach
    void setUp() {
        normalizer = new TransactionNormalizer();
    }
    
    @Test
    void testNormalizeCsvFormat() {
        // Given: Raw CSV-like records
        Map<String, String> rawRecord = new HashMap<>();
        rawRecord.put("transaction_id", "TXN001");
        rawRecord.put("customer_id", "CUST001");
        rawRecord.put("amount", "150.75");
        rawRecord.put("transaction_date", "2024-01-15");
        rawRecord.put("merchant_name", "Amazon");
        rawRecord.put("merchant_category", "Retail");
        
        List<Map<String, String>> rawRecords = List.of(rawRecord);
        
        // When: Normalizing records
        List<Transaction> transactions = normalizer.normalize(rawRecords, "CSV");
        
        // Then: Should produce valid Transaction objects
        assertNotNull(transactions);
        assertEquals(1, transactions.size());
        
        Transaction transaction = transactions.get(0);
        assertEquals("TXN001", transaction.getTransactionId());
        assertEquals("CUST001", transaction.getCustomerId());
        assertEquals(new BigDecimal("150.75"), transaction.getAmount());
        assertEquals("Amazon", transaction.getMerchantName());
        assertEquals("Retail", transaction.getMerchantCategory());
        assertEquals("CSV", transaction.getSourceSystem());
    }
    
    @Test
    void testNormalizeCreditCardFormat() {
        // Given: Credit card dataset format with separate date fields
        Map<String, String> rawRecord = new HashMap<>();
        rawRecord.put("User", "12345");
        rawRecord.put("Card", "9876");
        rawRecord.put("Year", "2024");
        rawRecord.put("Month", "3");
        rawRecord.put("Day", "15");
        rawRecord.put("Time", "14:30:00");
        rawRecord.put("Amount", "$125.50");
        rawRecord.put("Merchant Name", "Starbucks");
        rawRecord.put("Merchant City", "New York");
        rawRecord.put("Merchant State", "NY");
        rawRecord.put("Zip", "10001");
        rawRecord.put("MCC", "5812");
        rawRecord.put("Is Fraud?", "No");
        
        List<Map<String, String>> rawRecords = List.of(rawRecord);
        
        // When: Normalizing records
        List<Transaction> transactions = normalizer.normalize(rawRecords, "CSV");
        
        // Then: Should parse date components and location correctly
        assertNotNull(transactions);
        assertEquals(1, transactions.size());
        
        Transaction transaction = transactions.get(0);
        assertEquals("12345", transaction.getTransactionId());
        assertEquals("9876", transaction.getCustomerId());
        assertEquals(new BigDecimal("125.50"), transaction.getAmount());
        assertEquals("Starbucks", transaction.getMerchantName());
        assertEquals("5812", transaction.getMerchantCategory());
        assertEquals("New York, NY 10001", transaction.getLocation());
        assertFalse(transaction.isFraudFlag());
        
        // Check date parsing
        LocalDateTime expectedDate = LocalDateTime.of(2024, 3, 15, 14, 30, 0);
        assertEquals(expectedDate, transaction.getTransactionDate());
    }
    
    @Test
    void testNormalizeWithFraudFlag() {
        // Given: Records with various fraud flag representations
        Map<String, String> record1 = createBasicRecord();
        record1.put("Is Fraud?", "Yes");
        
        Map<String, String> record2 = createBasicRecord();
        record2.put("is_fraud", "true");
        
        Map<String, String> record3 = createBasicRecord();
        record3.put("fraud_flag", "1");
        
        List<Map<String, String>> rawRecords = List.of(record1, record2, record3);
        
        // When: Normalizing records
        List<Transaction> transactions = normalizer.normalize(rawRecords, "CSV");
        
        // Then: All should be marked as fraud
        assertEquals(3, transactions.size());
        assertTrue(transactions.get(0).isFraudFlag());
        assertTrue(transactions.get(1).isFraudFlag());
        assertTrue(transactions.get(2).isFraudFlag());
    }
    
    @Test
    void testNormalizeWithMissingFields() {
        // Given: Record with only required fields
        Map<String, String> rawRecord = new HashMap<>();
        rawRecord.put("transaction_id", "TXN999");
        rawRecord.put("amount", "99.99");
        rawRecord.put("Year", "2024");
        rawRecord.put("Month", "1");
        rawRecord.put("Day", "1");
        
        List<Map<String, String>> rawRecords = List.of(rawRecord);
        
        // When: Normalizing records
        List<Transaction> transactions = normalizer.normalize(rawRecords, "CSV");
        
        // Then: Should still create transaction with available data
        assertEquals(1, transactions.size());
        
        Transaction transaction = transactions.get(0);
        assertEquals("TXN999", transaction.getTransactionId());
        assertEquals(new BigDecimal("99.99"), transaction.getAmount());
        assertNull(transaction.getCustomerId());
        assertNull(transaction.getMerchantName());
    }
    
    @Test
    void testNormalizeWithInvalidAmount() {
        // Given: Record with invalid amount
        Map<String, String> rawRecord = createBasicRecord();
        rawRecord.put("amount", "invalid");
        
        List<Map<String, String>> rawRecords = List.of(rawRecord);
        
        // When: Normalizing records
        List<Transaction> transactions = normalizer.normalize(rawRecords, "CSV");
        
        // Then: Should skip invalid record
        assertEquals(0, transactions.size());
    }
    
    @Test
    void testNormalizeWithMissingTransactionId() {
        // Given: Record without transaction ID
        Map<String, String> rawRecord = new HashMap<>();
        rawRecord.put("amount", "100.00");
        rawRecord.put("transaction_date", "2024-01-01");
        
        List<Map<String, String>> rawRecords = List.of(rawRecord);
        
        // When: Normalizing records
        List<Transaction> transactions = normalizer.normalize(rawRecords, "CSV");
        
        // Then: Should skip record without ID
        assertEquals(0, transactions.size());
    }
    
    @Test
    void testNormalizeMultipleRecords() {
        // Given: Multiple valid records
        List<Map<String, String>> rawRecords = new ArrayList<>();
        for (int i = 1; i <= 100; i++) {
            Map<String, String> record = createBasicRecord();
            record.put("transaction_id", "TXN" + String.format("%03d", i));
            record.put("amount", String.valueOf(i * 10.5));
            rawRecords.add(record);
        }
        
        // When: Normalizing records
        List<Transaction> transactions = normalizer.normalize(rawRecords, "CSV");
        
        // Then: Should normalize all records
        assertEquals(100, transactions.size());
        assertEquals("TXN001", transactions.get(0).getTransactionId());
        assertEquals("TXN100", transactions.get(99).getTransactionId());
    }
    
    @Test
    void testNormalizeMixedValidAndInvalidRecords() {
        // Given: Mix of valid and invalid records
        List<Map<String, String>> rawRecords = new ArrayList<>();
        
        // Valid record
        rawRecords.add(createBasicRecord());
        
        // Invalid record (missing ID)
        Map<String, String> invalidRecord1 = new HashMap<>();
        invalidRecord1.put("amount", "100.00");
        rawRecords.add(invalidRecord1);
        
        // Valid record
        Map<String, String> validRecord2 = createBasicRecord();
        validRecord2.put("transaction_id", "TXN002");
        rawRecords.add(validRecord2);
        
        // Invalid record (bad amount)
        Map<String, String> invalidRecord2 = createBasicRecord();
        invalidRecord2.put("transaction_id", "TXN003");
        invalidRecord2.put("amount", "not_a_number");
        rawRecords.add(invalidRecord2);
        
        // When: Normalizing records
        List<Transaction> transactions = normalizer.normalize(rawRecords, "CSV");
        
        // Then: Should only normalize valid records
        assertEquals(2, transactions.size());
    }
    
    /**
     * Helper method to create a basic valid record.
     */
    private Map<String, String> createBasicRecord() {
        Map<String, String> record = new HashMap<>();
        record.put("transaction_id", "TXN001");
        record.put("customer_id", "CUST001");
        record.put("amount", "100.00");
        record.put("Year", "2024");
        record.put("Month", "1");
        record.put("Day", "15");
        return record;
    }
}
