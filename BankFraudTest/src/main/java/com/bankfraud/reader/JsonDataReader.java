package com.bankfraud.reader;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Implementation of DataReader for JSON format files.
 * Supports JSON arrays of transaction objects.
 * 
 * @author Banking Platform Team
 * @version 1.0
 */
public class JsonDataReader implements DataReader {
    
    private static final Logger logger = LoggerFactory.getLogger(JsonDataReader.class);
    private static final String FORMAT_IDENTIFIER = "JSON";
    private final ObjectMapper objectMapper;
    
    public JsonDataReader() {
        this.objectMapper = new ObjectMapper();
        logger.debug("JsonDataReader initialized with ObjectMapper");
    }
    
    /**
     * Reads JSON file and converts each object to a map of field-value pairs.
     * Expects JSON array format: [{"field1": "value1", ...}, ...]
     * 
     * @param filePath absolute path to the JSON file
     * @return list of transaction records
     * @throws IOException if file reading or JSON parsing fails
     */
    @Override
    public List<Map<String, String>> read(String filePath) throws IOException {
        logger.info("Starting to read JSON file: {}", filePath);
        
        if (!validateFormat(filePath)) {
            logger.error("Invalid file format or file does not exist: {}", filePath);
            throw new IOException("Invalid JSON file: " + filePath);
        }
        
        List<Map<String, String>> records = new ArrayList<>();
        
        try {
            File jsonFile = new File(filePath);
            JsonNode rootNode = objectMapper.readTree(jsonFile);
            
            if (!rootNode.isArray()) {
                logger.error("JSON file must contain an array of objects: {}", filePath);
                throw new IOException("Invalid JSON structure: expected array");
            }
            
            logger.debug("JSON array contains {} elements", rootNode.size());
            
            int recordCount = 0;
            for (JsonNode node : rootNode) {
                Map<String, String> record = new HashMap<>();
                
                // Convert JSON object fields to map
                node.fields().forEachRemaining(entry -> {
                    String key = entry.getKey();
                    String value = entry.getValue().isNull() ? null : entry.getValue().asText();
                    record.put(key, value);
                });
                
                records.add(record);
                recordCount++;
                
                if (recordCount % 1000 == 0) {
                    logger.debug("Processed {} records from JSON file", recordCount);
                }
            }
            
            logger.info("Successfully read {} records from JSON file: {}", recordCount, filePath);
            
        } catch (IOException e) {
            logger.error("Failed to read JSON file: {}. Error: {}", filePath, e.getMessage(), e);
            throw e;
        }
        
        return records;
    }
    
    /**
     * Validates that the file exists and has .json extension.
     * Also performs basic JSON syntax validation.
     * 
     * @param filePath path to validate
     * @return true if valid JSON file
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
        
        if (!filePath.toLowerCase().endsWith(".json")) {
            logger.warn("File does not have .json extension: {}", filePath);
            return false;
        }
        
        // Attempt to parse JSON to validate syntax
        try {
            objectMapper.readTree(new File(filePath));
            logger.debug("JSON file format validated: {}", filePath);
            return true;
        } catch (IOException e) {
            logger.warn("Invalid JSON syntax in file: {}. Error: {}", filePath, e.getMessage());
            return false;
        }
    }
    
    @Override
    public String getSupportedFormat() {
        return FORMAT_IDENTIFIER;
    }
}
