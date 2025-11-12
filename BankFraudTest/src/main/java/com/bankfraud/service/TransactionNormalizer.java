package com.bankfraud.service;

import com.bankfraud.model.Transaction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.math.BigDecimal;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeParseException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Service for normalizing transaction data from various formats into standardized Transaction objects.
 * Handles field mapping, data type conversion, and validation.
 * 
 * @author Banking Platform Team
 * @version 1.0
 */
public class TransactionNormalizer {
    
    private static final Logger logger = LoggerFactory.getLogger(TransactionNormalizer.class);
    
    // Common date formats to try parsing
    private static final DateTimeFormatter[] DATE_FORMATTERS = {
        DateTimeFormatter.ISO_LOCAL_DATE_TIME,
        DateTimeFormatter.ISO_LOCAL_DATE,
        DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"),
        DateTimeFormatter.ofPattern("yyyy-MM-dd"),
        DateTimeFormatter.ofPattern("MM/dd/yyyy"),
        DateTimeFormatter.ofPattern("dd/MM/yyyy"),
        DateTimeFormatter.ofPattern("yyyy/MM/dd")
    };
    
    /**
     * Normalizes a list of raw transaction records into Transaction objects.
     * 
     * @param rawRecords list of raw transaction data as maps
     * @param sourceFormat the source format identifier (CSV, JSON, FIXED_WIDTH)
     * @return list of normalized Transaction objects
     */
    public List<Transaction> normalize(List<Map<String, String>> rawRecords, String sourceFormat) {
        logger.info("Starting normalization of {} records from {} format", rawRecords.size(), sourceFormat);
        
        List<Transaction> transactions = new ArrayList<>();
        int successCount = 0;
        int failureCount = 0;
        
        for (int i = 0; i < rawRecords.size(); i++) {
            Map<String, String> rawRecord = rawRecords.get(i);
            
            try {
                Transaction transaction = normalizeRecord(rawRecord, sourceFormat);
                transactions.add(transaction);
                successCount++;
                
                if (successCount % 1000 == 0) {
                    logger.debug("Successfully normalized {} transactions", successCount);
                }
                
            } catch (Exception e) {
                failureCount++;
                logger.warn("Failed to normalize record at index {}: {}. Error: {}", 
                        i, rawRecord, e.getMessage());
            }
        }
        
        logger.info("Normalization complete. Success: {}, Failures: {}", successCount, failureCount);
        
        return transactions;
    }
    
    /**
     * Normalizes a single raw record into a Transaction object.
     * 
     * @param rawRecord raw transaction data
     * @param sourceFormat source format identifier
     * @return normalized Transaction object
     * @throws IllegalArgumentException if required fields are missing or invalid
     */
    private Transaction normalizeRecord(Map<String, String> rawRecord, String sourceFormat) {
        Transaction transaction = new Transaction();
        
        // Extract and set transaction ID
        String transactionId = extractField(rawRecord, "transaction_id", "User", "id");
        if (transactionId == null || transactionId.isEmpty()) {
            throw new IllegalArgumentException("Transaction ID is required");
        }
        transaction.setTransactionId(transactionId);
        
        // Extract and set customer ID
        String customerId = extractField(rawRecord, "customer_id", "Card", "customer");
        transaction.setCustomerId(customerId);
        
        // Extract and set amount
        String amountStr = extractField(rawRecord, "amount", "Amount");
        BigDecimal amount = parseAmount(amountStr);
        if (amount == null) {
            throw new IllegalArgumentException("Invalid amount: " + amountStr);
        }
        transaction.setAmount(amount);
        
        // Extract and set transaction date
        String dateStr = extractField(rawRecord, "transaction_date", "Year", "date", "timestamp");
        LocalDateTime transactionDate = parseDate(dateStr, rawRecord);
        if (transactionDate == null) {
            logger.warn("Could not parse date '{}', using current timestamp", dateStr);
            transactionDate = LocalDateTime.now();
        }
        transaction.setTransactionDate(transactionDate);
        
        // Extract and set merchant name
        String merchantName = extractField(rawRecord, "merchant_name", "Merchant Name", "merchant");
        transaction.setMerchantName(merchantName);
        
        // Extract and set merchant category
        String merchantCategory = extractField(rawRecord, "merchant_category", "MCC", "category");
        transaction.setMerchantCategory(merchantCategory);
        
        // Extract and set location
        String location = extractLocation(rawRecord);
        transaction.setLocation(location);
        
        // Extract and set fraud flag
        String isFraudStr = extractField(rawRecord, "is_fraud", "Is Fraud?", "fraud_flag");
        Boolean isFraud = parseFraudFlag(isFraudStr);
        transaction.setFraudFlag(isFraud != null ? isFraud : false);
        
        // Set source system
        transaction.setSourceSystem(sourceFormat);
        
        // Set status to PENDING by default
        transaction.setStatus("PENDING");
        
        logger.debug("Normalized transaction: ID={}, Amount={}, Date={}", 
                transaction.getTransactionId(), transaction.getAmount(), transaction.getTransactionDate());
        
        return transaction;
    }
    
    /**
     * Extracts a field value from raw record, trying multiple possible field names.
     * 
     * @param rawRecord raw transaction data
     * @param fieldNames possible field names to try
     * @return field value or null if not found
     */
    private String extractField(Map<String, String> rawRecord, String... fieldNames) {
        for (String fieldName : fieldNames) {
            if (rawRecord.containsKey(fieldName)) {
                return rawRecord.get(fieldName);
            }
        }
        return null;
    }
    
    /**
     * Extracts location from raw record, combining city, state, and zip if available.
     * 
     * @param rawRecord raw transaction data
     * @return formatted location string
     */
    private String extractLocation(Map<String, String> rawRecord) {
        String city = extractField(rawRecord, "merchant_city", "Merchant City", "city");
        String state = extractField(rawRecord, "merchant_state", "Merchant State", "state");
        String zip = extractField(rawRecord, "zip", "Zip");
        
        StringBuilder location = new StringBuilder();
        if (city != null && !city.isEmpty()) {
            location.append(city);
        }
        if (state != null && !state.isEmpty()) {
            if (location.length() > 0) location.append(", ");
            location.append(state);
        }
        if (zip != null && !zip.isEmpty()) {
            if (location.length() > 0) location.append(" ");
            location.append(zip);
        }
        
        return location.length() > 0 ? location.toString() : null;
    }
    
    /**
     * Parses amount string to BigDecimal.
     * 
     * @param amountStr amount as string
     * @return BigDecimal amount or null if parsing fails
     */
    private BigDecimal parseAmount(String amountStr) {
        if (amountStr == null || amountStr.trim().isEmpty()) {
            return null;
        }
        
        try {
            // Remove currency symbols and commas
            String cleanAmount = amountStr.replaceAll("[^0-9.-]", "");
            return new BigDecimal(cleanAmount);
        } catch (NumberFormatException e) {
            logger.warn("Failed to parse amount: {}", amountStr);
            return null;
        }
    }
    
    /**
     * Parses date string to LocalDateTime, trying multiple formats.
     * For credit card data, combines Year, Month, Day, Time fields.
     * 
     * @param dateStr date as string
     * @param rawRecord full raw record for extracting individual date components
     * @return LocalDateTime or null if parsing fails
     */
    private LocalDateTime parseDate(String dateStr, Map<String, String> rawRecord) {
        // Try parsing single date string
        if (dateStr != null && !dateStr.trim().isEmpty()) {
            for (DateTimeFormatter formatter : DATE_FORMATTERS) {
                try {
                    return LocalDateTime.parse(dateStr, formatter);
                } catch (DateTimeParseException e) {
                    try {
                        LocalDate date = LocalDate.parse(dateStr, formatter);
                        return date.atStartOfDay();
                    } catch (DateTimeParseException e2) {
                        // Try next formatter
                    }
                }
            }
        }
        
        // Try parsing from separate Year, Month, Day, Time fields (credit card format)
        try {
            String year = extractField(rawRecord, "Year");
            String month = extractField(rawRecord, "Month");
            String day = extractField(rawRecord, "Day");
            String time = extractField(rawRecord, "Time");
            
            if (year != null && month != null && day != null) {
                String combinedDate = String.format("%s-%02d-%02d", 
                        year, Integer.parseInt(month), Integer.parseInt(day));
                
                if (time != null && !time.isEmpty()) {
                    combinedDate += " " + time;
                    return LocalDateTime.parse(combinedDate, 
                            DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));
                } else {
                    return LocalDate.parse(combinedDate, DateTimeFormatter.ISO_LOCAL_DATE).atStartOfDay();
                }
            }
        } catch (Exception e) {
            logger.debug("Could not parse date from separate fields: {}", e.getMessage());
        }
        
        return null;
    }
    
    /**
     * Parses fraud flag from various string representations.
     * 
     * @param fraudStr fraud flag as string
     * @return Boolean fraud flag or null if parsing fails
     */
    private Boolean parseFraudFlag(String fraudStr) {
        if (fraudStr == null || fraudStr.trim().isEmpty()) {
            return null;
        }
        
        String normalized = fraudStr.trim().toLowerCase();
        
        // Check for boolean representations
        if (normalized.equals("true") || normalized.equals("yes") || normalized.equals("1")) {
            return true;
        } else if (normalized.equals("false") || normalized.equals("no") || normalized.equals("0")) {
            return false;
        }
        
        // Check for Yes/No with quotes (from credit card dataset)
        if (normalized.contains("yes")) {
            return true;
        } else if (normalized.contains("no")) {
            return false;
        }
        
        logger.debug("Could not parse fraud flag: {}", fraudStr);
        return null;
    }
}
