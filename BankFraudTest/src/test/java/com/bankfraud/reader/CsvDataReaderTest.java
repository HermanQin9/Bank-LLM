package com.bankfraud.reader;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for CsvDataReader.
 * Tests CSV file reading, validation, and error handling.
 * 
 * @author Banking Platform Team
 * @version 1.0
 */
class CsvDataReaderTest {
    
    private CsvDataReader reader;
    
    @TempDir
    Path tempDir;
    
    @BeforeEach
    void setUp() {
        reader = new CsvDataReader();
    }
    
    @Test
    void testReadValidCsvFile() throws IOException {
        // Given: A valid CSV file with headers
        String csvContent = "transaction_id,customer_id,amount,transaction_date\n" +
                            "TXN001,CUST001,100.50,2024-01-15\n" +
                            "TXN002,CUST002,250.75,2024-01-16\n";
        Path csvFile = tempDir.resolve("test.csv");
        Files.writeString(csvFile, csvContent);
        
        // When: Reading the CSV file
        List<Map<String, String>> records = reader.read(csvFile.toString());
        
        // Then: Should read all records correctly
        assertNotNull(records);
        assertEquals(2, records.size());
        
        Map<String, String> firstRecord = records.get(0);
        assertEquals("TXN001", firstRecord.get("transaction_id"));
        assertEquals("CUST001", firstRecord.get("customer_id"));
        assertEquals("100.50", firstRecord.get("amount"));
        assertEquals("2024-01-15", firstRecord.get("transaction_date"));
    }
    
    @Test
    void testReadEmptyCsvFile() throws IOException {
        // Given: An empty CSV file (only headers)
        String csvContent = "transaction_id,customer_id,amount\n";
        Path csvFile = tempDir.resolve("empty.csv");
        Files.writeString(csvFile, csvContent);
        
        // When: Reading the empty CSV file
        List<Map<String, String>> records = reader.read(csvFile.toString());
        
        // Then: Should return empty list
        assertNotNull(records);
        assertEquals(0, records.size());
    }
    
    @Test
    void testReadCsvWithSpecialCharacters() throws IOException {
        // Given: CSV with special characters and quotes
        String csvContent = "name,description,amount\n" +
                            "\"Test, Inc.\",\"Product with \"\"quotes\"\"\",123.45\n";
        Path csvFile = tempDir.resolve("special.csv");
        Files.writeString(csvFile, csvContent);
        
        // When: Reading the CSV file
        List<Map<String, String>> records = reader.read(csvFile.toString());
        
        // Then: Should handle special characters correctly
        assertNotNull(records);
        assertEquals(1, records.size());
        assertEquals("Test, Inc.", records.get(0).get("name"));
    }
    
    @Test
    void testValidateFormatValidFile() throws IOException {
        // Given: A valid CSV file
        Path csvFile = tempDir.resolve("valid.csv");
        Files.writeString(csvFile, "header\nvalue\n");
        
        // When: Validating the file format
        boolean isValid = reader.validateFormat(csvFile.toString());
        
        // Then: Should return true
        assertTrue(isValid);
    }
    
    @Test
    void testValidateFormatNonExistentFile() {
        // Given: A non-existent file path
        String nonExistentPath = tempDir.resolve("nonexistent.csv").toString();
        
        // When: Validating the file format
        boolean isValid = reader.validateFormat(nonExistentPath);
        
        // Then: Should return false
        assertFalse(isValid);
    }
    
    @Test
    void testValidateFormatInvalidExtension() throws IOException {
        // Given: A file with wrong extension
        Path txtFile = tempDir.resolve("data.txt");
        Files.writeString(txtFile, "data");
        
        // When: Validating the file format
        boolean isValid = reader.validateFormat(txtFile.toString());
        
        // Then: Should return false
        assertFalse(isValid);
    }
    
    @Test
    void testValidateFormatNullPath() {
        // When: Validating null path
        boolean isValid = reader.validateFormat(null);
        
        // Then: Should return false
        assertFalse(isValid);
    }
    
    @Test
    void testReadNonExistentFile() {
        // Given: A non-existent file path
        String nonExistentPath = tempDir.resolve("nonexistent.csv").toString();
        
        // When & Then: Should throw IOException
        assertThrows(IOException.class, () -> reader.read(nonExistentPath));
    }
    
    @Test
    void testGetSupportedFormat() {
        // When: Getting supported format
        String format = reader.getSupportedFormat();
        
        // Then: Should return CSV
        assertEquals("CSV", format);
    }
}
