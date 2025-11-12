package com.bankfraud.reader;

import java.io.IOException;
import java.util.List;
import java.util.Map;

/**
 * Interface for reading transaction data from various file formats.
 * Implementations should support CSV, JSON, and fixed-width formats.
 * 
 * @author Banking Platform Team
 * @version 1.0
 */
public interface DataReader {
    
    /**
     * Reads transaction data from the specified file.
     * 
     * @param filePath absolute path to the data file
     * @return list of transaction records as key-value maps
     * @throws IOException if file cannot be read or is malformed
     */
    List<Map<String, String>> read(String filePath) throws IOException;
    
    /**
     * Validates the file format before reading.
     * 
     * @param filePath absolute path to the data file
     * @return true if file format is valid, false otherwise
     */
    boolean validateFormat(String filePath);
    
    /**
     * Returns the supported file format.
     * 
     * @return format identifier (e.g., "CSV", "JSON", "FIXED_WIDTH")
     */
    String getSupportedFormat();
}
