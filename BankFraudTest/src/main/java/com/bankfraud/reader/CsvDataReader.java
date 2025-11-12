package com.bankfraud.reader;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Implementation of DataReader for CSV format files.
 * Supports standard CSV with headers.
 * 
 * @author Banking Platform Team
 * @version 1.0
 */
public class CsvDataReader implements DataReader {
    
    private static final Logger logger = LoggerFactory.getLogger(CsvDataReader.class);
    private static final String FORMAT_IDENTIFIER = "CSV";
    
    /**
     * Reads CSV file and converts each row to a map of column-value pairs.
     * 
     * @param filePath absolute path to the CSV file
     * @return list of transaction records
     * @throws IOException if file reading fails
     */
    @Override
    public List<Map<String, String>> read(String filePath) throws IOException {
        logger.info("Starting to read CSV file: {}", filePath);
        
        if (!validateFormat(filePath)) {
            logger.error("Invalid file format or file does not exist: {}", filePath);
            throw new IOException("Invalid CSV file: " + filePath);
        }
        
        List<Map<String, String>> records = new ArrayList<>();
        int recordCount = 0;
        
        try (FileReader reader = new FileReader(filePath);
             CSVParser csvParser = new CSVParser(reader, 
                     CSVFormat.DEFAULT.builder()
                             .setHeader()
                             .setSkipHeaderRecord(true)
                             .setTrim(true)
                             .setIgnoreEmptyLines(true)
                             .build())) {
            
            logger.debug("CSV parser initialized with headers: {}", csvParser.getHeaderNames());
            
            for (CSVRecord csvRecord : csvParser) {
                Map<String, String> record = new HashMap<>();
                
                // Convert CSV record to map
                for (String header : csvParser.getHeaderNames()) {
                    String value = csvRecord.get(header);
                    record.put(header, value);
                }
                
                records.add(record);
                recordCount++;
                
                if (recordCount % 1000 == 0) {
                    logger.debug("Processed {} records from CSV file", recordCount);
                }
            }
            
            logger.info("Successfully read {} records from CSV file: {}", recordCount, filePath);
            
        } catch (IOException e) {
            logger.error("Failed to read CSV file: {}. Error: {}", filePath, e.getMessage(), e);
            throw e;
        }
        
        return records;
    }
    
    /**
     * Validates that the file exists and has .csv extension.
     * 
     * @param filePath path to validate
     * @return true if valid CSV file
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
        
        if (!filePath.toLowerCase().endsWith(".csv")) {
            logger.warn("File does not have .csv extension: {}", filePath);
            return false;
        }
        
        logger.debug("CSV file format validated: {}", filePath);
        return true;
    }
    
    @Override
    public String getSupportedFormat() {
        return FORMAT_IDENTIFIER;
    }
}
