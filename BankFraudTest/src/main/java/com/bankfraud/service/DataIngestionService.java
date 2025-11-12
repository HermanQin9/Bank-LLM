package com.bankfraud.service;

import com.bankfraud.model.Transaction;
import com.bankfraud.reader.CsvDataReader;
import com.bankfraud.reader.DataReader;
import com.bankfraud.reader.FixedWidthDataReader;
import com.bankfraud.reader.JsonDataReader;
import com.bankfraud.repository.TransactionRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Service for orchestrating end-to-end data ingestion pipeline.
 * Coordinates data reading, normalization, and database storage.
 * 
 * @author Banking Platform Team
 * @version 1.0
 */
public class DataIngestionService {
    
    private static final Logger logger = LoggerFactory.getLogger(DataIngestionService.class);
    
    private final Map<String, DataReader> readers;
    private final TransactionNormalizer normalizer;
    private final TransactionRepository repository;
    
    /**
     * Constructs DataIngestionService with all required dependencies.
     */
    public DataIngestionService() {
        this.readers = new HashMap<>();
        this.readers.put("CSV", new CsvDataReader());
        this.readers.put("JSON", new JsonDataReader());
        this.readers.put("FIXED_WIDTH", new FixedWidthDataReader());
        
        this.normalizer = new TransactionNormalizer();
        this.repository = new TransactionRepository();
        
        logger.info("DataIngestionService initialized with {} data readers", readers.size());
    }
    
    /**
     * Ingests data from a file, automatically detecting format from extension.
     * 
     * @param filePath absolute path to the data file
     * @return ingestion result with statistics
     */
    public IngestionResult ingestFile(String filePath) {
        logger.info("Starting data ingestion for file: {}", filePath);
        long startTime = System.currentTimeMillis();
        
        IngestionResult result = new IngestionResult();
        result.setFilePath(filePath);
        
        try {
            // Determine file format
            String format = detectFormat(filePath);
            result.setFormat(format);
            logger.info("Detected file format: {}", format);
            
            // Get appropriate reader
            DataReader reader = readers.get(format);
            if (reader == null) {
                throw new IllegalArgumentException("Unsupported file format: " + format);
            }
            
            // Step 1: Read raw data
            logger.info("Step 1/3: Reading data from file");
            List<Map<String, String>> rawRecords = reader.read(filePath);
            result.setRecordsRead(rawRecords.size());
            logger.info("Read {} raw records from file", rawRecords.size());
            
            // Step 2: Normalize data
            logger.info("Step 2/3: Normalizing data");
            List<Transaction> transactions = normalizer.normalize(rawRecords, format);
            result.setRecordsNormalized(transactions.size());
            logger.info("Normalized {} records", transactions.size());
            
            // Step 3: Store in database
            logger.info("Step 3/3: Storing data in database");
            int savedCount = repository.saveBatch(transactions);
            result.setRecordsSaved(savedCount);
            logger.info("Saved {} records to database", savedCount);
            
            result.setSuccess(true);
            
        } catch (IOException e) {
            logger.error("I/O error during data ingestion: {}", e.getMessage(), e);
            result.setSuccess(false);
            result.setErrorMessage("I/O Error: " + e.getMessage());
        } catch (Exception e) {
            logger.error("Unexpected error during data ingestion: {}", e.getMessage(), e);
            result.setSuccess(false);
            result.setErrorMessage("Error: " + e.getMessage());
        }
        
        long endTime = System.currentTimeMillis();
        result.setDurationMs(endTime - startTime);
        
        logger.info("Data ingestion completed in {} ms. Success: {}", result.getDurationMs(), result.isSuccess());
        logger.info("Ingestion statistics: Read={}, Normalized={}, Saved={}", 
                result.getRecordsRead(), result.getRecordsNormalized(), result.getRecordsSaved());
        
        return result;
    }
    
    /**
     * Ingests data with explicit format specification.
     * 
     * @param filePath absolute path to the data file
     * @param format data format (CSV, JSON, FIXED_WIDTH)
     * @return ingestion result with statistics
     */
    public IngestionResult ingestFile(String filePath, String format) {
        logger.info("Starting data ingestion for file: {} with format: {}", filePath, format);
        long startTime = System.currentTimeMillis();
        
        IngestionResult result = new IngestionResult();
        result.setFilePath(filePath);
        result.setFormat(format);
        
        try {
            // Get appropriate reader
            DataReader reader = readers.get(format.toUpperCase());
            if (reader == null) {
                throw new IllegalArgumentException("Unsupported file format: " + format);
            }
            
            // Validate format
            if (!reader.validateFormat(filePath)) {
                throw new IllegalArgumentException("File format validation failed for: " + filePath);
            }
            
            // Step 1: Read raw data
            logger.info("Step 1/3: Reading data from file");
            List<Map<String, String>> rawRecords = reader.read(filePath);
            result.setRecordsRead(rawRecords.size());
            logger.info("Read {} raw records from file", rawRecords.size());
            
            // Step 2: Normalize data
            logger.info("Step 2/3: Normalizing data");
            List<Transaction> transactions = normalizer.normalize(rawRecords, format);
            result.setRecordsNormalized(transactions.size());
            logger.info("Normalized {} records", transactions.size());
            
            // Step 3: Store in database
            logger.info("Step 3/3: Storing data in database");
            int savedCount = repository.saveBatch(transactions);
            result.setRecordsSaved(savedCount);
            logger.info("Saved {} records to database", savedCount);
            
            result.setSuccess(true);
            
        } catch (IOException e) {
            logger.error("I/O error during data ingestion: {}", e.getMessage(), e);
            result.setSuccess(false);
            result.setErrorMessage("I/O Error: " + e.getMessage());
        } catch (Exception e) {
            logger.error("Unexpected error during data ingestion: {}", e.getMessage(), e);
            result.setSuccess(false);
            result.setErrorMessage("Error: " + e.getMessage());
        }
        
        long endTime = System.currentTimeMillis();
        result.setDurationMs(endTime - startTime);
        
        logger.info("Data ingestion completed in {} ms. Success: {}", result.getDurationMs(), result.isSuccess());
        
        return result;
    }
    
    /**
     * Detects file format from file extension.
     * 
     * @param filePath path to analyze
     * @return format identifier (CSV, JSON, FIXED_WIDTH)
     */
    private String detectFormat(String filePath) {
        String lowerPath = filePath.toLowerCase();
        
        if (lowerPath.endsWith(".csv")) {
            return "CSV";
        } else if (lowerPath.endsWith(".json")) {
            return "JSON";
        } else if (lowerPath.endsWith(".txt")) {
            return "FIXED_WIDTH";
        } else {
            throw new IllegalArgumentException("Cannot detect format for file: " + filePath);
        }
    }
    
    /**
     * Result object containing ingestion statistics and status.
     */
    public static class IngestionResult {
        private String filePath;
        private String format;
        private int recordsRead;
        private int recordsNormalized;
        private int recordsSaved;
        private boolean success;
        private String errorMessage;
        private long durationMs;
        
        // Getters and setters
        public String getFilePath() { return filePath; }
        public void setFilePath(String filePath) { this.filePath = filePath; }
        
        public String getFormat() { return format; }
        public void setFormat(String format) { this.format = format; }
        
        public int getRecordsRead() { return recordsRead; }
        public void setRecordsRead(int recordsRead) { this.recordsRead = recordsRead; }
        
        public int getRecordsNormalized() { return recordsNormalized; }
        public void setRecordsNormalized(int recordsNormalized) { this.recordsNormalized = recordsNormalized; }
        
        public int getRecordsSaved() { return recordsSaved; }
        public void setRecordsSaved(int recordsSaved) { this.recordsSaved = recordsSaved; }
        
        public boolean isSuccess() { return success; }
        public void setSuccess(boolean success) { this.success = success; }
        
        public String getErrorMessage() { return errorMessage; }
        public void setErrorMessage(String errorMessage) { this.errorMessage = errorMessage; }
        
        public long getDurationMs() { return durationMs; }
        public void setDurationMs(long durationMs) { this.durationMs = durationMs; }
        
        @Override
        public String toString() {
            return String.format("IngestionResult{file='%s', format='%s', read=%d, normalized=%d, saved=%d, success=%s, duration=%dms}",
                    filePath, format, recordsRead, recordsNormalized, recordsSaved, success, durationMs);
        }
    }
}
