package com.bankfraud.reader;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Implementation of DataReader for fixed-width format files.
 * Supports reading fixed-width text files with predefined column positions.
 * 
 * Column Layout (example):
 * - Transaction ID: positions 0-19 (20 chars)
 * - Customer ID: positions 20-39 (20 chars)
 * - Amount: positions 40-59 (20 chars)
 * - Date: positions 60-79 (20 chars)
 * - Merchant: positions 80-109 (30 chars)
 * 
 * @author Banking Platform Team
 * @version 1.0
 */
public class FixedWidthDataReader implements DataReader {
    
    private static final Logger logger = LoggerFactory.getLogger(FixedWidthDataReader.class);
    private static final String FORMAT_IDENTIFIER = "FIXED_WIDTH";
    
    // Column definitions: field name -> [start_position, length]
    private final Map<String, int[]> columnDefinitions;
    
    /**
     * Constructs FixedWidthDataReader with default column layout.
     */
    public FixedWidthDataReader() {
        this.columnDefinitions = getDefaultColumnDefinitions();
        logger.debug("FixedWidthDataReader initialized with default column definitions");
    }
    
    /**
     * Constructs FixedWidthDataReader with custom column layout.
     * 
     * @param columnDefinitions map of field name to [start_position, length]
     */
    public FixedWidthDataReader(Map<String, int[]> columnDefinitions) {
        this.columnDefinitions = columnDefinitions;
        logger.debug("FixedWidthDataReader initialized with custom column definitions");
    }
    
    /**
     * Returns default column definitions for fixed-width format.
     */
    private Map<String, int[]> getDefaultColumnDefinitions() {
        Map<String, int[]> definitions = new HashMap<>();
        definitions.put("transaction_id", new int[]{0, 20});
        definitions.put("customer_id", new int[]{20, 20});
        definitions.put("amount", new int[]{40, 20});
        definitions.put("transaction_date", new int[]{60, 20});
        definitions.put("merchant_name", new int[]{80, 30});
        return definitions;
    }
    
    /**
     * Reads fixed-width file and extracts fields based on column definitions.
     * 
     * @param filePath absolute path to the fixed-width file
     * @return list of transaction records
     * @throws IOException if file reading fails
     */
    @Override
    public List<Map<String, String>> read(String filePath) throws IOException {
        logger.info("Starting to read fixed-width file: {}", filePath);
        
        if (!validateFormat(filePath)) {
            logger.error("Invalid file format or file does not exist: {}", filePath);
            throw new IOException("Invalid fixed-width file: " + filePath);
        }
        
        List<Map<String, String>> records = new ArrayList<>();
        int recordCount = 0;
        int lineNumber = 0;
        
        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            String line;
            
            // Skip header line if exists
            line = reader.readLine();
            lineNumber++;
            if (line != null && line.trim().startsWith("#")) {
                logger.debug("Skipping header line: {}", line.substring(0, Math.min(50, line.length())));
                line = reader.readLine();
                lineNumber++;
            }
            
            while (line != null) {
                lineNumber++;
                
                if (line.trim().isEmpty()) {
                    logger.debug("Skipping empty line at line number: {}", lineNumber);
                    line = reader.readLine();
                    continue;
                }
                
                try {
                    Map<String, String> record = parseLine(line);
                    records.add(record);
                    recordCount++;
                    
                    if (recordCount % 1000 == 0) {
                        logger.debug("Processed {} records from fixed-width file", recordCount);
                    }
                    
                } catch (StringIndexOutOfBoundsException e) {
                    logger.warn("Line {} is too short to parse: expected {} chars, got {}",
                            lineNumber, getExpectedLineLength(), line.length());
                }
                
                line = reader.readLine();
            }
            
            logger.info("Successfully read {} records from fixed-width file: {}", recordCount, filePath);
            
        } catch (IOException e) {
            logger.error("Failed to read fixed-width file: {}. Error: {}", filePath, e.getMessage(), e);
            throw e;
        }
        
        return records;
    }
    
    /**
     * Parses a single line of fixed-width data into a map.
     * 
     * @param line the line to parse
     * @return map of field names to values
     */
    private Map<String, String> parseLine(String line) {
        Map<String, String> record = new HashMap<>();
        
        for (Map.Entry<String, int[]> entry : columnDefinitions.entrySet()) {
            String fieldName = entry.getKey();
            int startPos = entry.getValue()[0];
            int length = entry.getValue()[1];
            
            if (startPos + length <= line.length()) {
                String value = line.substring(startPos, startPos + length).trim();
                record.put(fieldName, value);
            } else {
                logger.warn("Field '{}' extends beyond line length. Using empty value.", fieldName);
                record.put(fieldName, "");
            }
        }
        
        return record;
    }
    
    /**
     * Calculates the expected line length based on column definitions.
     */
    private int getExpectedLineLength() {
        return columnDefinitions.values().stream()
                .mapToInt(positions -> positions[0] + positions[1])
                .max()
                .orElse(0);
    }
    
    /**
     * Validates that the file exists and has .txt extension.
     * 
     * @param filePath path to validate
     * @return true if valid fixed-width file
     */
    @Override
    public boolean validateFormat(String filePath) {
        if (filePath == null || filePath.trim().isEmpty()) {
            logger.warn("File path is null or empty");
            return false;
        }
        
        if (!Files.exists(Paths.get(filePath))) {
            logger.warn("File does not exist: {}", filePath);
            return false;
        }
        
        if (!filePath.toLowerCase().endsWith(".txt")) {
            logger.warn("File does not have .txt extension: {}", filePath);
            return false;
        }
        
        logger.debug("Fixed-width file format validated: {}", filePath);
        return true;
    }
    
    @Override
    public String getSupportedFormat() {
        return FORMAT_IDENTIFIER;
    }
}
